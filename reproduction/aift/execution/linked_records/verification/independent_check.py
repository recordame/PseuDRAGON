import ast
from copy import deepcopy
from itertools import combinations
import hashlib
import json
from pathlib import Path
import pandas as pd
package = Path(__file__).resolve().parents[1]
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = read(package / 'artifact_manifest.json')['files']
assert all((sha(package / name) == digest for name, digest in manifest.items() if name != 'journal.jsonl'))
binding = read(package / 'source_binding.json')
inputs = read(package / 'inputs.json')

def function(text):
    node = deepcopy(next((n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'pseudonymize_data')))
    if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        node.body.pop(0)
    return node
def portable_ast(node):
    if isinstance(node, ast.AST):
        return {'node': type(node).__name__, 'fields': {key: portable_ast(value) for key, value in ast.iter_fields(node) if value is not None and value != []}}
    if isinstance(node, list):
        return [portable_ast(value) for value in node]
    return node

outputs = {}
for entry in binding['bindings']:
    source = package / entry['portable']
    assert sha(source) == entry['portable_sha256']
    original = function(source.read_text())
    ast_hash = hashlib.sha256(json.dumps(portable_ast(original), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    assert ast_hash == entry['executable_ast_sha256']
    namespace = {'pd': pd, 'hashlib': hashlib}
    module = ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[]))
    exec(compile(module, '<original_function_only>', 'exec'), namespace)
    frame = pd.DataFrame(inputs[entry['table']])
    before = frame.copy(deep=True)
    output = namespace['pseudonymize_data'](frame)
    assert frame.equals(before) and output['ROW_ID'].equals(before['ROW_ID']) and output['WD_MN'].equals(before['WD_MN'])
    outputs[entry['table']] = output.to_dict('records')
original = {r['ROW_ID']: r for table in inputs.values() for r in table}
transformed = {r['ROW_ID']: r for table in outputs.values() for r in table}
assert len(original) == len(transformed) == 84
assert all((original[k]['CUST_ID'] != transformed[k]['CUST_ID'] for k in original))
assert all(((original[a]['CUST_ID'] == original[b]['CUST_ID']) == (transformed[a]['CUST_ID'] == transformed[b]['CUST_ID']) for a, b in combinations(original, 2)))
keys = sorted(set((r['CUST_ID'] for r in original.values())))
assert len(keys) == 12
buckets = {}
for customer in keys:
    member_ids = {k for k, v in original.items() if v['CUST_ID'] == customer}
    pseudonyms = {transformed[k]['CUST_ID'] for k in member_ids}
    assert len(pseudonyms) == 1
    pseudonym = next(iter(pseudonyms))
    after_ids = {k for k, v in transformed.items() if v['CUST_ID'] == pseudonym}
    assert member_ids == after_ids
    assert sum((original[k]['WD_MN'] for k in member_ids)) == sum((transformed[k]['WD_MN'] for k in after_ids))
    buckets[customer] = {'row_ids': sorted(member_ids), 'count': len(member_ids), 'sum': sum((original[k]['WD_MN'] for k in member_ids))}
left = [k for k in original if k.startswith('CD_')]
right = [k for k in original if k.startswith('HF_')]
pairs = lambda data: {(a, b) for a in left for b in right if data[a]['CUST_ID'] == data[b]['CUST_ID']}
expected = pairs(original)
assert len(expected) == 144 and pairs(transformed) == expected
negative = deepcopy(transformed)
negative[right[0]]['CUST_ID'] = next((negative[k]['CUST_ID'] for k in left if original[k]['CUST_ID'] == keys[1]))
wrong = pairs(negative)
assert len(wrong) == 144 and len(wrong - expected) == len(expected - wrong) == 3
for run in ['first', 'repeat']:
    recorded = read(package / (run + '.json'))
    assert recorded['passed'] and recorded['output_records'] == outputs
    assert recorded['positive']['counts_and_sums_match'] and recorded['positive']['customer_groups_checked'] == 12
    assert set(map(tuple, recorded['negative_control']['missing_correct_row_pairs'])) == expected - wrong
    assert set(map(tuple, recorded['negative_control']['extra_wrong_row_pairs'])) == wrong - expected
assert (package / 'first.json').read_bytes() == (package / 'repeat.json').read_bytes()
report = {'status': 'pass', 'method': 'Bound selected-function execution; raw row-set, equality and group-sum invariants; no reference hash implementation', 'packaged_functions_match_bound_AST': True, 'original_source_binding_audit_reperformed': False, 'records': 84, 'customer_groups': 12, 'exact_join_pairs': 144, 'equality_comparisons': 3486, 'wrong_and_missing_control_pairs': [3, 3], 'output_records_match_both_runs': True, 'first_repeat_byte_identical': True, 'immutable_artifact_hashes_verified': True}
(package / 'verification/independent_report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
