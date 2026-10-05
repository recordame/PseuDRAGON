import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import types
import pandas as pd
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MASKS = frozenset('*#Xx●•')

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def protected(original, output):
    indices = [i for i, char in enumerate(original) if char.isdigit()]
    if not indices:
        return output == original
    if not isinstance(output, str) or not output:
        return False
    if set(output).issubset(MASKS):
        return True
    if len(output) != len(original):
        return False
    hidden = set()
    for i, (old, new) in enumerate(zip(original, output)):
        if new == old:
            continue
        if new not in MASKS:
            return False
        hidden.add(i)
    targets = [set(indices[-3:])]
    if len(indices) > 3:
        targets.append(set(indices[-4:-1]))
    return any((target.issubset(hidden) for target in targets))

def unfreeze(fixture):
    value = fixture['value']
    if value == {'missing': 'nan'}:
        return float('nan')
    if value == {'missing': 'pd.NA'}:
        return pd.NA
    return value

def pack(value):
    if bool(pd.isna(value)):
        return {'kind': 'missing', 'type': type(value).__name__}
    if isinstance(value, str):
        return {'kind': 'string', 'value': value}
    raise TypeError('Unexpected output type: ' + type(value).__name__)

def main():
    protocol = read(BASE / 'protocol.json')
    inputs = read(BASE / 'inputs.json')
    issues = []
    semantic_probes = [('', '', True), ('', '*', False), ('abc', 'abc', True), ('abc', '***', False), ('1', '*', True), ('12', '*2', False), ('12', '**', True), ('12345', '12***', True), ('12345', '1***5', True), ('12345', '***45', False), ('12-34-56', '12-3*-**', True), ('12-34-56', '12-34***', False), ('Ａ１２３４５６', 'Ａ１２３***', True), ('12345', '#', True), ('12345', '98***', False), ('12345', '12345', False)]
    assert all((protected(original, output) == answer for original, output, answer in semantic_probes))
    runtime_hashes = {path: sha(BASE / path) == digest for path, digest in protocol['runtime_hashes'].items()}
    assert all(runtime_hashes.values()), 'Packaged runtime input changed'
    assert sha(BASE / 'inputs.json') == protocol['inputs_sha256']
    binding = read(BASE / 'source_binding_summary.json')
    assert binding['status'] == 'pass'
    assert binding['inputs_sha256'] == sha(BASE / 'inputs.json')
    assert binding['original_source_hashes_verified'] == protocol['source_hashes']
    assert binding['minimal_runtime_hashes'] == protocol['runtime_hashes']
    canonical = lambda value: hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for field in ['raw_candidates', 'compiled_recorded_rows', 'fixtures']:
        assert canonical(inputs[field]) == binding[field + '_canonical_sha256']
    reviewed = set(read(BASE / 'reviewed_raw_hashes.json')['approved_sha256'])
    assert len(inputs['raw_candidates']) == binding['raw_candidate_count'] == 33
    assert len({c['candidate_id'] for c in inputs['raw_candidates']}) == 33
    codes = {}
    for candidate in inputs['raw_candidates']:
        digest = hashlib.sha256(candidate['code'].encode()).hexdigest()
        assert digest == candidate['code_sha256'] and digest in reviewed
        codes[digest] = candidate['code']
    assert len(codes) == 8 and len({(r['table'], r['column']) for r in inputs['raw_candidates']}) == 11
    assignments = ast.parse((BASE / 'original_fixtures.py').read_text()).body
    node = next((n.value for n in assignments if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'VALUES' for t in n.targets))))
    source_fixtures = []
    for n in node.elts:
        if isinstance(n, ast.Constant):
            source_fixtures.append(n.value)
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and (n.func.id == 'float'):
            assert ast.literal_eval(n.args[0]) == 'nan'
            source_fixtures.append({'missing': 'nan'})
        elif isinstance(n, ast.Attribute) and n.attr == 'NA' and (n.value.id == 'pd'):
            source_fixtures.append({'missing': 'pd.NA'})
        else:
            raise AssertionError('Unrecognized fixture expression')
    fixtures = inputs['fixtures']
    assert [f['value'] for f in fixtures] == source_fixtures and len(fixtures) == 17
    assert [f['case_index'] for f in fixtures] == list(range(17))
    assert sum((f['kind'] == 'missing' for f in fixtures)) == 3
    saved_compiled = inputs['compiled_recorded_rows']
    assert len(saved_compiled) == binding['compiled_recorded_candidate_count'] == 66
    namespace = {'__builtins__': {}, 'str': str, 'len': len, 'pd': types.SimpleNamespace(notna=pd.notna), 're': types.SimpleNamespace(sub=re.sub)}
    functions = {digest: eval(compile(ast.parse(code, mode='eval'), '<independent-raw>', 'eval'), namespace) for digest, code in codes.items()}
    expected = {}
    for candidate in inputs['raw_candidates']:
        for fixture in fixtures:
            value = unfreeze(fixture)
            try:
                actual = functions[candidate['code_sha256']](value)
                output, status, error = (pack(actual), 'ok', None)
            except Exception as exc:
                actual, output, status = (None, None, 'runtime_error')
                error = type(exc).__name__ + ': ' + str(exc)
            missing = fixture['kind'] == 'missing'
            expected['raw', candidate['candidate_id'], fixture['case_index']] = {'output': output, 'status': status, 'error': error, 'position_pass': None if missing else status == 'ok' and protected(value, actual), 'missing_pass': status == 'ok' and output['kind'] == 'missing' if missing else None, 'rep': candidate['rep']}
    for candidate in saved_compiled:
        cid = f"{candidate['id']}-c{candidate['candidate_index']}"
        assert len(candidate['checks']) == 17
        for fixture, check in zip(fixtures, candidate['checks']):
            assert check['case_index'] == fixture['case_index']
            actual = check['actual']
            missing = fixture['kind'] == 'missing'
            expected['compiled', cid, fixture['case_index']] = {'output': pack(actual), 'status': 'ok', 'error': None, 'position_pass': None if missing else protected(fixture['value'], actual), 'missing_pass': actual is None if missing else None, 'rep': candidate['rep']}
    summaries = {}
    for kind in ['raw', 'compiled']:
        selected = [(key, row) for key, row in expected.items() if key[0] == kind]
        ids = sorted({key[1] for key, row in selected})
        rows = [row for key, row in selected]
        digit = [row for key, row in selected if fixtures[key[2]]['kind'] == 'nonmissing' and any((c.isdigit() for c in fixtures[key[2]]['value']))]
        per_candidate = {cid: [r for k, r in selected if k[1] == cid] for cid in ids}
        summaries[kind] = {'candidates': len(ids), 'cases': len(rows), 'positional_pass': sum((r['position_pass'] is True for r in rows)), 'positional_total': sum((r['position_pass'] is not None for r in rows)), 'digit_pass': sum((r['position_pass'] is True for r in digit)), 'digit_total': len(digit), 'missing_pass': sum((r['missing_pass'] is True for r in rows)), 'missing_total': sum((r['missing_pass'] is not None for r in rows)), 'all_nonmissing_candidates': sum((all((r['position_pass'] is not False for r in group)) for group in per_candidate.values())), 'all_missing_candidates': sum((all((r['missing_pass'] is not False for r in group)) for group in per_candidate.values())), 'all17_candidates': sum((all((r['position_pass'] is not False and r['missing_pass'] is not False for r in group)) for group in per_candidate.values())), 'runtime_errors': sum((r['status'] != 'ok' for r in rows)), 'per_repetition': {str(rep): {'cases': sum((r['rep'] == rep for r in rows)), 'nonmissing_pass': sum((r['rep'] == rep and r['position_pass'] is True for r in rows)), 'missing_pass': sum((r['rep'] == rep and r['missing_pass'] is True for r in rows))} for rep in [1, 2, 3]}, 'per_fixture': [{'case_index': i, 'input': fixture['value'], 'candidates': len(ids), 'nonmissing_pass': sum((k[2] == i and r['position_pass'] is True for k, r in selected)), 'missing_pass': sum((k[2] == i and r['missing_pass'] is True for k, r in selected))} for i, fixture in enumerate(fixtures)]}
    comparisons = {}
    for name in ['first.json', 'repeat.json']:
        path = BASE / name
        if not path.exists():
            comparisons[name] = {'available': False}
            continue
        artifact = read(path)
        assert artifact['protocol_sha256'] == sha(BASE / 'protocol.json')
        assert artifact['runner_sha256'] == sha(BASE / 'run_raw.py')
        assert artifact['new_model_calls'] == artifact['new_human_approvals'] == 0
        assert artifact['raw_parameter_missing_or_empty'] == 33
        observed = {(r['kind'], r['candidate_id'], r['case_index']): r for r in artifact['rows']}
        assert len(observed) == len(artifact['rows']), 'Duplicate result row'
        differences = []
        if set(observed) != set(expected):
            differences.append({'keys_missing': sorted(set(expected) - set(observed)), 'keys_extra': sorted(set(observed) - set(expected))})
        for key in observed.keys() & expected.keys():
            assert observed[key]['input'] == fixtures[key[2]]['value']
            for field in ['output', 'status', 'error', 'position_pass', 'missing_pass', 'rep']:
                left, right = (observed[key].get(field), expected[key][field])
                if field == 'output' and key[0] == 'compiled' and (right.get('kind') == 'missing'):
                    equal = isinstance(left, dict) and left.get('kind') == 'missing'
                else:
                    equal = left == right
                if not equal:
                    differences.append({'key': key, 'field': field, 'observed': left, 'independent': right})
        metric_map = {'candidates': 'candidates', 'code_input_cases': 'cases', 'nonmissing_pass': 'positional_pass', 'nonmissing_cases': 'positional_total', 'digit_bearing_pass': 'digit_pass', 'digit_bearing_cases': 'digit_total', 'missing_pass': 'missing_pass', 'missing_cases': 'missing_total', 'all_14_nonmissing_pass': 'all_nonmissing_candidates', 'all_3_missing_pass': 'all_missing_candidates', 'all_17_pass': 'all17_candidates', 'runtime_errors': 'runtime_errors', 'per_repetition': 'per_repetition'}
        for kind in ['raw', 'compiled']:
            for reported, calculated in metric_map.items():
                if artifact['summary'][kind][reported] != summaries[kind][calculated]:
                    differences.append({'summary': kind, 'metric': reported})
        comparisons[name] = {'available': True, 'sha256': sha(path), 'rows': len(observed), 'differences': differences}
        if differences:
            issues.append(name + ' differs from independent reconstruction')
    repeated = all(((BASE / n).exists() for n in ['first.json', 'repeat.json']))
    identical = (BASE / 'first.json').read_bytes() == (BASE / 'repeat.json').read_bytes() if repeated else None
    if repeated and (not identical):
        issues.append('first/repeat bytes differ')
    report = {'status': 'pass' if repeated and (not issues) else 'incomplete' if not repeated else 'fail', 'scope': 'Independent unchanged raw execution; compiled stored-output recheck, not new compiled execution.', 'protocol_sha256': sha(BASE / 'protocol.json'), 'inputs_sha256': sha(BASE / 'inputs.json'), 'checker_sha256': sha(Path(__file__)), 'python': sys.version, 'pandas': pd.__version__, 'runtime_hashes_verified': runtime_hashes, 'raw_code_hashes_match_review': True, 'source_binding_summary_sha256': sha(BASE / 'source_binding_summary.json'), 'source_binding_statement_consistent': True, 'original_source_binding_audit_reperformed': False, 'source_binding_scope': 'Original source audit is a retained trusted local statement; this public rerun validates its input hashes and package consistency without reading private sources.', 'fixtures_match_packaged_VALUES_AST': True, 'compiled_records_source': 'Packaged inputs bound by the prior local source audit', 'independent_semantic_probes_passed': len(semantic_probes), 'summary': summaries, 'first_repeat_byte_identical': identical, 'comparisons': comparisons, 'issues': issues}
    (HERE / 'independent_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'independent_outputs.json').write_text(json.dumps([{'kind': key[0], 'candidate_id': key[1], 'case_index': key[2], **row} for key, row in expected.items()], ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'summary': summaries, 'comparisons': comparisons, 'issues': issues}, ensure_ascii=False))
    return 1 if issues else 0
if __name__ == '__main__':
    raise SystemExit(main())
