import ast
import collections
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import pandas as pd
HERE = Path(__file__).resolve().parent
MASK = set('*#Xx●•')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def decode(value):
    if value == {'missing': 'nan'}:
        return float('nan')
    if value == {'missing': 'pd.NA'}:
        return pd.NA
    return value

def encode(value):
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return {'kind': 'missing', 'type': type(value).__name__}
    if isinstance(value, str):
        return {'kind': 'string', 'value': value}
    return {'kind': 'other', 'type': type(value).__name__, 'repr': repr(value)}

def positional(original, output):
    if output['kind'] != 'string':
        return False
    result = output['value']
    positions = [i for i, char in enumerate(original) if char.isdigit()]
    if not positions:
        return result == original
    if result and all((char in MASK for char in result)):
        return True
    if len(original) != len(result):
        return False
    if any((left != right and right not in MASK for left, right in zip(original, result))):
        return False
    ranges = [positions[-min(3, len(positions)):]]
    if len(positions) > 3:
        ranges.append(positions[-4:-1])
    return any((all((result[i] != original[i] and result[i] in MASK for i in indexes)) for indexes in ranges))

def summarize(rows):
    summaries = {}
    for kind in ['raw', 'compiled']:
        selected = [r for r in rows if r['kind'] == kind]
        c = collections.Counter()
        groups = collections.defaultdict(list)
        by_rep = collections.defaultdict(collections.Counter)
        for row in selected:
            groups[row['candidate_id']].append(row)
            c['code_input_cases'] += 1
            c['runtime_errors'] += row['status'] != 'ok'
            if row['position_pass'] is not None:
                c['nonmissing_cases'] += 1
                c['nonmissing_pass'] += row['position_pass']
                digit = any((ch.isdigit() for ch in row['input']))
                c['digit_bearing_cases' if digit else 'zero_digit_cases'] += 1
                c['digit_bearing_pass' if digit else 'zero_digit_pass'] += row['position_pass']
            else:
                c['missing_cases'] += 1
                c['missing_pass'] += row['missing_pass']
            by_rep[row['rep']]['cases'] += 1
            by_rep[row['rep']]['nonmissing_pass'] += bool(row['position_pass'])
            by_rep[row['rep']]['missing_pass'] += bool(row['missing_pass'])
        c['candidates'] = len(groups)
        for candidate_rows in groups.values():
            c['all_14_nonmissing_pass'] += all((r['position_pass'] for r in candidate_rows if r['position_pass'] is not None))
            c['all_12_digit_bearing_pass'] += all((r['position_pass'] for r in candidate_rows if isinstance(r['input'], str) and any((ch.isdigit() for ch in r['input']))))
            c['all_3_missing_pass'] += all((r['missing_pass'] for r in candidate_rows if r['missing_pass'] is not None))
            c['all_17_pass'] += all((r['position_pass'] if r['position_pass'] is not None else r['missing_pass'] for r in candidate_rows))
        summaries[kind] = dict(c) | {'per_repetition': dict(by_rep)}
    return summaries

def main():
    output = HERE / sys.argv[1]
    assert output.parent == HERE and output.name in ['first.json', 'repeat.json']
    protocol = load(HERE / 'protocol.json')
    assert digest(HERE / 'inputs.json') == protocol['inputs_sha256']
    for path, expected in protocol['runtime_hashes'].items():
        assert digest(HERE / path) == expected, path
    inputs = load(HERE / 'inputs.json')
    source = HERE / 'raw_ast_guard.py'
    spec = importlib.util.spec_from_file_location('frozen_raw_ast_validator', source)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    reviewed = set(load(HERE / 'reviewed_raw_hashes.json')['approved_sha256'])
    rows = []
    for candidate in inputs['raw_candidates']:
        code = candidate['code']
        assert hashlib.sha256(code.encode()).hexdigest() == candidate['code_sha256']
        assert candidate['code_sha256'] in reviewed
        tree = validator.checked_expression(code)
        builtins = {'str': str, 'len': len, 'min': min, 'max': max, 'enumerate': enumerate, 'range': range, 'list': list, 'tuple': tuple, 'reversed': reversed, 'isinstance': isinstance, 'int': int}
        function = eval(compile(tree, '<unchanged_archived_MASK>', 'eval'), {'__builtins__': builtins, 'pd': pd, 're': re})
        for fixture in inputs['fixtures']:
            row = {'kind': 'raw', 'candidate_id': candidate['candidate_id'], 'rep': candidate['rep'], 'case_index': fixture['case_index'], 'input': fixture['value'], 'status': 'ok', 'error': None}
            try:
                row['output'] = encode(function(decode(fixture['value'])))
            except Exception as exc:
                row.update(status='runtime_error', error=type(exc).__name__ + ': ' + str(exc), output={'kind': 'error'})
            row['position_pass'] = positional(fixture['value'], row['output']) if fixture['kind'] != 'missing' else None
            row['missing_pass'] = row['output']['kind'] == 'missing' if fixture['kind'] == 'missing' else None
            rows.append(row)
    for candidate in inputs['compiled_recorded_rows']:
        for fixture, check in zip(inputs['fixtures'], candidate['checks'], strict=True):
            assert fixture['case_index'] == check['case_index']
            encoded = encode(check['actual'])
            rows.append({'kind': 'compiled', 'candidate_id': candidate['id'] + '-c' + str(candidate['candidate_index']), 'rep': candidate['rep'], 'case_index': fixture['case_index'], 'input': fixture['value'], 'output': encoded, 'status': 'ok', 'error': None, 'position_pass': positional(fixture['value'], encoded) if fixture['kind'] != 'missing' else None, 'missing_pass': encoded['kind'] == 'missing' if fixture['kind'] == 'missing' else None})
    result = {'protocol_sha256': digest(HERE / 'protocol.json'), 'runner_sha256': digest(__file__), 'runtime': {'python': sys.version, 'pandas': pd.__version__}, 'rows': rows, 'summary': summarize(rows), 'raw_parameter_missing_or_empty': sum((not r['parameters'] for r in inputs['raw_candidates'])), 'syntax_errors': 0, 'unsupported_reviewed_snippets': 0, 'new_model_calls': 0, 'new_human_approvals': 0}
    for path, expected in protocol['runtime_hashes'].items():
        assert digest(HERE / path) == expected, path
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False))
if __name__ == '__main__':
    main()
