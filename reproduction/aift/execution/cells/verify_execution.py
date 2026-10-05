from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import pandas as pd
EXPECTED_COUNTS = {'HASH': 25, 'MASK': 4, 'GENERALIZE': 3, 'KEEP': 60}
BOUNDARY = [None, '', 'a', 'abc', 'abcd', 'abcde', '0000012345', '한글🙂은행계좌', 'a\nb', 'a\x00b', '-1001', '0', '999', '1000']
PARAMETERS = {'KEEP': {}, 'HASH': {'algorithm': 'SHA-256'}, 'MASK': {'mask_char': '*', 'mask_last': 4, 'short_value_mask_length': 4}, 'GENERALIZE': {'strategy': 'integer_interval', 'interval_width': 1000, 'interval_origin': 0, 'separator': '~', 'non_numeric': 'stringify'}}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()

def relative_file(root, relative):
    candidate = root / relative
    require(candidate.is_file(), f'Missing bundle file: {relative}')
    require(candidate.resolve().is_relative_to(root.resolve()), f'File escapes bundle: {relative}')
    return candidate

def read_json(root, relative):
    return json.loads(relative_file(root, relative).read_text(encoding='utf-8'))

def syntax(code):
    return ast.dump(ast.parse(code), include_attributes=False)

def selected_snippet(method, column):
    value = f'record[{column!r}]'
    if method == 'KEEP':
        return ''
    if method == 'HASH':
        return f'import hashlib\nhashed_{column} = hashlib.sha256(str({value}).encode()).hexdigest()'
    if method == 'MASK':
        return f"masked_{column} = str({value})[:-4] + '****' if len(str({value})) > 4 else '****'"
    return 'lambda x: f"{(int(x)//1000)*1000}~{(int(x)//1000)*1000+999}" if pd.notna(x) and str(x).isdigit() else str(x)'

def expected_function(columns):
    body = ['def pseudonymize_data(df):', '    df = df.copy()']
    for column, item in columns.items():
        method = item['action']['action']
        if method == 'KEEP':
            continue
        if method == 'HASH':
            inner = 'lambda x: (hashlib.sha256(str(x).encode()).hexdigest()) if pd.notna(x) else None'
        elif method == 'MASK':
            inner = "lambda x: (str(x)[:-4] + '****' if len(str(x)) > 4 else '****') if pd.notna(x) else None"
        else:
            inner = selected_snippet(method, column)
        callback = f'lambda _pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs: ({inner})(_pseudragon_value, *_pseudragon_args, **_pseudragon_kwargs) if pd.notna(_pseudragon_value) else _pseudragon_value'
        body.extend([f'    if {column!r} in df.columns:', f'        df[{column!r}] = df[{column!r}].apply({callback})'])
    body.append('    return df')
    return ast.parse('\n'.join(body)).body[0]

def load_checked_function(code, columns, table):
    nodes = [node for node in ast.parse(code).body if isinstance(node, ast.FunctionDef) and node.name == 'pseudonymize_data']
    require(len(nodes) == 1, f'{table}: exactly one transformation function required')
    actual = nodes[0]
    if actual.body and isinstance(actual.body[0], ast.Expr) and isinstance(actual.body[0].value, ast.Constant) and isinstance(actual.body[0].value.value, str):
        actual.body = actual.body[1:]
    require(ast.dump(actual, include_attributes=False) == ast.dump(expected_function(columns), include_attributes=False), f'{table}: generated function disagrees with the bounded policy syntax')
    namespace = {'pd': pd, 'hashlib': hashlib, '__builtins__': {'str': str, 'int': int, 'len': len}}
    module = ast.Module(body=[actual], type_ignores=[])
    exec(compile(module, '<retained-transformation>', 'exec'), namespace)
    return namespace['pseudonymize_data']

def scalar_reference(value, method):
    if pd.isna(value) or method == 'KEEP':
        return value
    text = str(value)
    if method == 'HASH':
        return hashlib.new('sha256', text.encode('utf-8')).hexdigest()
    if method == 'MASK':
        visible = max(0, len(text) - 4)
        return text[:visible] + '*' * 4
    if method == 'GENERALIZE':
        if not text.isdigit():
            return text
        low = int(text) - int(text) % 1000
        return '{}~{}'.format(low, low + 999)
    raise ValueError(f'Unsupported method: {method}')

def reference_frame(frame, columns):
    expected = frame.copy(deep=True)
    for column, item in columns.items():
        method = item['action']['action']
        if method != 'KEEP':
            expected[column] = pd.Series([scalar_reference(value, method) for value in frame[column]], index=frame.index)
    return expected

def compare_frames(expected, actual, label):
    require(isinstance(actual, pd.DataFrame), f'{label}: result is not a DataFrame')
    require(list(expected.columns) == list(actual.columns), f'{label}: columns changed')
    require(expected.index.equals(actual.index), f'{label}: index changed')
    for column in expected:
        for row, (wanted, observed) in enumerate(zip(expected[column], actual[column])):
            a_missing, b_missing = (bool(pd.isna(wanted)), bool(pd.isna(observed)))
            require(a_missing and b_missing or (not a_missing and (not b_missing) and (wanted == observed)), f'{label}: mismatch in {column}, row {row}')

def run_frame(transform, frame, columns, label):
    before = frame.copy(deep=True)
    actual = transform(frame)
    compare_frames(reference_frame(before, columns), actual, label)
    pd.testing.assert_frame_equal(before, frame, check_exact=True)
    return actual

def verify(bundle_root):
    root = bundle_root.resolve()
    base = 'data/execution/'
    manifest_path = relative_file(root, base + 'repaired_policy_manifest.json')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    policies = manifest['policies']
    protocol = read_json(root, base + 'protocol.json')
    summary = read_json(root, base + 'summary.json')
    recorded_runs = read_json(root, base + 'repair_results/standalone_runs.json')
    links = read_json(root, base + 'repair_results/selection_code_output_links.json')
    require(len(policies) == 9, 'Expected nine policy tables')
    require(protocol['boundary']['matched_prior_values'] == BOUNDARY, 'Boundary protocol differs from the declared 14 values')
    require(len(recorded_runs) == 27 and len(links) == 9, 'Expected 27 historical records and nine output links')
    require(summary['semantic_actions'] == EXPECTED_COUNTS, 'Historical method counts differ')
    require(summary['run_id'] == manifest['run_id'], 'Historical run IDs disagree')
    link_by_table = {link['table']: link for link in links}
    require(set(link_by_table) == set(policies), 'Historical link tables differ')
    require({row['table'] for row in recorded_runs} == set(policies), 'Historical execution tables differ')
    inputs_present = [(root / (base + f'inputs/{table}.json')).is_file() for table in policies]
    require(all(inputs_present) or not any(inputs_present), 'Optional original-input export is partial: require all nine tables')
    counts = Counter()
    tables = []
    total_original = total_boundary = total_missing = checked_snippets = 0
    historical_hashes_recomputed = 0
    manifest_hash = sha_bytes(manifest_path.read_bytes())
    for table, policy in sorted(policies.items()):
        require(re.fullmatch('[A-Za-z0-9_]+', table) is not None, 'Invalid policy table identifier')
        columns = policy['columns']
        for column, item in columns.items():
            require(re.fullmatch('[A-Za-z_][A-Za-z0-9_]*', column) is not None, f'{table}: invalid column identifier')
            action = item['action']
            method = action['action']
            require(method in PARAMETERS, f'{table}.{column}: unsupported operation')
            require(action['parameters'] == PARAMETERS[method], f'{table}.{column}: method parameters disagree with retained contract')
            require(syntax(action.get('code_snippet', '')) == syntax(selected_snippet(method, column)), f'{table}.{column}: selected snippet disagrees with declared operation')
            counts[method] += 1
            checked_snippets += 1
        program = relative_file(root, base + f'replayed_programs/{table}.py')
        code = program.read_text(encoding='utf-8')
        transform = load_checked_function(code, columns, table)
        link = link_by_table[table]
        require(link['run_id'] == manifest['run_id'], f'{table}: linked run ID differs')
        require(link['packaged_function_sha256'] == sha_bytes(program.read_bytes()), f'{table}: packaged function bytes differ from packaging output link')
        require(link['packaged_policy_manifest_sha256'] == manifest_hash, f'{table}: packaged policy manifest differs from packaging output link')
        records = [row for row in recorded_runs if row['table'] == table]
        require(len(records) == 3 and {row['repeat'] for row in records} == {1, 2, 3}, f'{table}: historical repeats differ')
        output_hashes = []
        for record in sorted(records, key=lambda row: row['repeat']):
            require(record['returncode'] == 0 and record['exact_csv_match'] and record['passed'], f'{table}: failed historical execution record')
            require(record['rows'] == 1000 and record['columns'] == len(columns) and (record['cells'] == 1000 * len(columns)), f'{table}: historical record dimensions disagree')
            require(record['output_sha256'] == record['oracle_csv_sha256'] == link['oracle_csv_sha256'], f'{table}: historical output hashes differ')
            output_hashes.append(record['output_sha256'])
        require(output_hashes == link['output_sha256_by_repeat'] and len(set(output_hashes)) == 1, f'{table}: repeated hashes differ')
        table_result = {'table': table, 'columns': len(columns), 'program_sha256': sha_bytes(program.read_bytes()), 'metadata_and_function_structure': 'PASS'}
        for kind, values in (('boundary', BOUNDARY), ('missing', [None, float('nan'), pd.NA])):
            frame = pd.DataFrame({column: pd.Series(values, dtype=object) for column in columns})
            frame['__UNSELECTED_SENTINEL__'] = pd.Series(values, dtype=object)
            result = run_frame(transform, frame, columns, f'{table}/{kind}')
            if kind == 'missing':
                require(bool(result.isna().all().all()), f'{table}: missing inputs changed')
                total_missing += len(values) * len(columns)
            else:
                total_boundary += len(values) * len(columns)
            table_result[kind + '_cells'] = len(values) * len(columns)
        if all(inputs_present):
            payload = relative_file(root, base + f'inputs/{table}.json').read_text(encoding='utf-8')
            original = pd.read_json(io.StringIO(payload), orient='table')
            require(set(original.columns) == set(columns) and len(original) == 1000, f'{table}: original-input dimensions differ')
            output = run_frame(transform, original, columns, f'{table}/original')
            total_original += original.size
            expected = reference_frame(original, columns)
            csv_hash = sha_bytes(expected.to_csv(index=False).encode('utf-8-sig'))
            require(csv_hash == sha_bytes(output.to_csv(index=False).encode('utf-8-sig')), f'{table}: fresh in-memory CSV serialization differs from scalar reference')
            require(csv_hash == link['oracle_csv_sha256'], f'{table}: serialized input replay differs from historical CSV hash')
            historical_hashes_recomputed += 3
            table_result['original_cells'] = int(original.size)
            table_result['recomputed_historical_csv_sha256'] = csv_hash
        tables.append(table_result)
    require(dict(counts) == EXPECTED_COUNTS and checked_snippets == 92, 'Recomputed method counts differ')
    require(total_boundary == 1288 and total_missing == 276, 'Boundary or missing-value totals differ')
    require(not all(inputs_present) or total_original == 92000, 'Original-input cell total differs')
    return {'status': 'PASS', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Independent replay of retained transformation functions and historical record checks', 'runtime': {'python': sys.version.split()[0], 'pandas': pd.__version__}, 'fresh_function_replay': {'programs': len(tables), 'policy_columns': checked_snippets, 'method_counts': dict(counts), 'boundary_declared_cells': total_boundary, 'expanded_missing_declared_cells': total_missing, 'boundary_sentinel_cells': 126, 'missing_sentinel_cells': 27, 'original_input_cells': int(total_original), 'original_inputs': 'PASS' if all(inputs_present) else 'SKIPPED: optional exports absent', 'input_frames_unchanged': True}, 'historical_standalone_records': {'recorded_run_id': summary['run_id'], 'recorded_completed_at_utc': summary.get('completed_at_utc'), 'records_checked': len(recorded_runs), 'recorded_unique_original_cells': sum((row['cells'] for row in recorded_runs if row['repeat'] == 1)), 'recorded_cell_comparisons_including_repeats': sum((row['cells'] for row in recorded_runs)), 'csv_hashes_independently_recomputed': historical_hashes_recomputed, 'fresh_standalone_executions': 0, 'database_execution_hashes_recomputed': False}, 'tables': tables, 'limits': ['No new standalone process, original database connection, model call, or human approval.', 'The historical database-path-substituted script hashes are retained, not reconstructed.', 'Syntax checks cover only the four retained operation forms; no arbitrary-code proof.', 'Output agreement does not establish policy suitability or privacy protection.']}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-root', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.bundle_root)
    except Exception as error:
        result = {'status': 'FAIL', 'error_type': type(error).__name__, 'error': str(error)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
