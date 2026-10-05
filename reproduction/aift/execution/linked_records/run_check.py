from pathlib import Path
from copy import deepcopy
from datetime import datetime, timezone
import ast
import hashlib
import json
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TABLES = ('CD_TRNS_TRAN', 'HF_TRNS_TRAN')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2) + '\n')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def executable_function(text):
    tree = ast.parse(text)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'pseudonymize_data']
    assert len(functions) == 1
    function = deepcopy(functions[0])
    if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str):
        function.body.pop(0)
    assert not any((isinstance(n, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.Try)) for n in ast.walk(function)))
    names = {n.id for n in ast.walk(function) if isinstance(n, ast.Name)}
    assert names <= {'df', 'pd', 'hashlib', 'str', 'len', 'x', '_pseudragon_value', '_pseudragon_args', '_pseudragon_kwargs'}
    return function

def portable_ast(node):
    if isinstance(node, ast.AST):
        return {'node': type(node).__name__, 'fields': {key: portable_ast(value) for key, value in ast.iter_fields(node) if value is not None and value != []}}
    if isinstance(node, list):
        return [portable_ast(value) for value in node]
    return node

def ast_digest(function):
    return hashlib.sha256(json.dumps(portable_ast(function), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def row_pairs(frames):
    return sorted(((a['ROW_ID'], b['ROW_ID']) for a in frames[TABLES[0]] for b in frames[TABLES[1]] if a['CUST_ID'] == b['CUST_ID']))

def aggregates(frames):
    result = {}
    for rows in frames.values():
        for row in rows:
            key = row['CUST_ID']
            result.setdefault(key, {'count': 0, 'sum_WD_MN': 0})
            result[key]['count'] += 1
            result[key]['sum_WD_MN'] += row['WD_MN']
    return result

def run(run_name):
    import pandas as pd
    assert run_name in {'first', 'repeat'}
    destination = HERE / (run_name + '.json')
    assert not destination.exists(), 'Preserve previous outcomes'
    protocol = read(HERE / 'protocol.json')
    for relative, expected in protocol['frozen_files'].items():
        assert sha(HERE / relative) == expected, relative
    binding = read(HERE / 'source_binding.json')
    for row in binding['bindings']:
        portable = HERE / row['portable']
        assert sha(portable) == row['portable_sha256']
        assert ast_digest(executable_function(portable.read_text())) == row['executable_ast_sha256']
    raw = read(HERE / 'inputs.json')
    transformed, frames_unchanged = ({}, True)
    for row in binding['bindings']:
        frame = pd.DataFrame(raw[row['table']], columns=['ROW_ID', 'CUST_ID', 'WD_MN'])
        before = frame.copy(deep=True)
        function = executable_function((HERE / row['portable']).read_text())
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        environment = {'pd': pd, 'hashlib': hashlib}
        exec(compile(module, '<bound-selected-function>', 'exec'), environment)
        after = environment['pseudonymize_data'](frame)
        assert list(after.columns) == list(frame.columns) and after.index.equals(frame.index)
        frames_unchanged = frames_unchanged and frame.equals(before)
        assert after.ROW_ID.equals(before.ROW_ID) and after.WD_MN.equals(before.WD_MN)
        transformed[row['table']] = after.to_dict('records')
    raw_all = raw[TABLES[0]] + raw[TABLES[1]]
    after_all = transformed[TABLES[0]] + transformed[TABLES[1]]
    transformed_identifiers = sum((a['CUST_ID'] != b['CUST_ID'] for a, b in zip(raw_all, after_all)))
    equality_checks = len(raw_all) * (len(raw_all) - 1) // 2
    equality_errors = sum(((raw_all[i]['CUST_ID'] == raw_all[j]['CUST_ID']) != (after_all[i]['CUST_ID'] == after_all[j]['CUST_ID']) for i in range(len(raw_all)) for j in range(i)))
    key_map = {}
    for a, b in zip(raw[TABLES[0]], transformed[TABLES[0]]):
        assert a['CUST_ID'] not in key_map or key_map[a['CUST_ID']] == b['CUST_ID']
        key_map[a['CUST_ID']] = b['CUST_ID']
    assert len(key_map) == len(set(key_map.values())) == 12
    raw_groups = aggregates(raw)
    expected_groups = {key_map[k]: v for k, v in raw_groups.items()}
    original_pairs = row_pairs(raw)
    actual_pairs = row_pairs(transformed)
    assert len(original_pairs) == 144
    groups_match = aggregates(transformed) == expected_groups
    negative = deepcopy(transformed)
    negative[TABLES[1]][0]['CUST_ID'] = key_map['demo-customer-002']
    negative_pairs = row_pairs(negative)
    missing_pairs = sorted(set(original_pairs) - set(negative_pairs))
    extra_pairs = sorted(set(negative_pairs) - set(original_pairs))
    negative_groups = aggregates(negative)
    affected_groups = [key for key in expected_groups if negative_groups.get(key) != expected_groups[key]]
    result = {'protocol_sha256': sha(HERE / 'protocol.json'), 'input_sha256': sha(HERE / 'inputs.json'), 'units': protocol['units'], 'new_model_calls': 0, 'new_participant_decisions': 0, 'positive': {'identifiers_transformed': transformed_identifiers, 'input_frames_unchanged': frames_unchanged, 'customer_equality_pairs_checked': equality_checks, 'customer_equality_errors': equality_errors, 'raw_join_pairs': len(original_pairs), 'output_join_pairs': len(actual_pairs), 'row_pair_identities_match': original_pairs == actual_pairs, 'customer_groups_checked': len(expected_groups), 'counts_and_sums_match': groups_match, 'each_raw_customer_rows': sorted((v['count'] for v in raw_groups.values()))}, 'negative_control': {'join_count_unchanged': len(negative_pairs) == len(original_pairs), 'join_count': len(negative_pairs), 'wrong_links_detected': bool(missing_pairs or extra_pairs), 'missing_correct_row_pairs': missing_pairs, 'extra_wrong_row_pairs': extra_pairs, 'affected_customer_groups': len(affected_groups), 'group_error_detected': negative_groups != expected_groups}, 'output_records': transformed, 'source_bindings': binding['bindings'], 'passed': transformed_identifiers == 84 and frames_unchanged and (equality_errors == 0) and (original_pairs == actual_pairs) and groups_match and (len(negative_pairs) == 144) and (len(missing_pairs) == len(extra_pairs) == 3) and (len(affected_groups) == 2)}
    assert result['passed'], 'Report unexpected failures; do not modify source or fixtures'
    if run_name == 'repeat':
        assert json.loads(json.dumps(result)) == read(HERE / 'first.json'), 'Deterministic repeat mismatch'
    write(destination, result)
    print(json.dumps({'run': run_name, 'passed': result['passed'], 'positive': result['positive'], 'negative_control': result['negative_control']}, indent=2))
if __name__ == '__main__':
    run(sys.argv[1])
