"""Compile an explicitly supplied display rule into a reviewable MASK candidate.

Explicit caller rules and a deliberately bounded source-table parser produce
separate pending-review candidates. Neither path approves legal applicability;
raw model alternatives remain available.
"""
import ast
import copy
import hashlib
import re
import unicodedata


def compile_display_requirement(requirement):
    if not isinstance(requirement, dict) or set(requirement) != {'position', 'minimum_digits', 'source_record'}:
        raise ValueError('Supply position, minimum_digits and source_record explicitly.')
    position, n, source = requirement['position'], requirement['minimum_digits'], requirement['source_record']
    if position not in {'suffix', 'before_last'} or type(n) is not int or not 1 <= n <= 64:
        raise ValueError('Unsupported masking position or digit count.')
    _validate_source_record(source)
    digit_positions = '[j for j, c in enumerate(str(x)) if c.isdigit()]'
    if position == 'suffix':
        selected = f'{digit_positions}[-{n}:]'
    else:
        selected = f'(lambda p: p[-{n+1}:-1] if len(p) > {n} else p)({digit_positions})'
    code = ("lambda x: (lambda s, p: ''.join('*' if i in p else c for i, c in enumerate(s)))"
            f"(str(x), {selected}) if pd.notna(x) else x")
    parameters = {'strategy': 'digit_position_mask', 'position': position, 'minimum_digits': n,
                  'mask_char': '*', 'short_value': 'mask_all_digits', 'preserve_missing': True,
                  'requirement_source': copy.deepcopy(source), 'requirement_origin': 'caller_supplied'}
    return {'method': 'MASK', 'parameters': parameters, 'code_snippet': code,
            'description': f'Mask {n} digits at {position}; preserve separators and missing values. Review the supplied rule before selection.',
            'legal_source': f"{source['source']} [chunk {source['chunk_id']}]",
            'origin': 'constraint_compiled_pending_review'}


def _validate_source_record(source):
    """Check recorded content identity, including composite retrieval windows."""
    if not isinstance(source, dict) or not all(isinstance(source.get(k), str) and source[k]
                                              for k in ['source', 'chunk_id', 'text', 'text_sha256']):
        raise ValueError('The display requirement needs an exact source record.')
    digest = hashlib.sha256(source['text'].encode()).hexdigest()
    chunk_id = hashlib.sha256((source['source'] + '\n' + source['text']).encode()).hexdigest()[:16]
    if digest != source['text_sha256'] or chunk_id != source['chunk_id']:
        raise ValueError('Source text and its recorded identity disagree.')
    if 'component_records' in source:
        components = source['component_records']
        if (not isinstance(components, list) or not components or
                source.get('composition') != 'adjacent_chunks_same_source'):
            raise ValueError('Unsupported source-window composition.')
        for component in components:
            _validate_source_record(component)
            if component['source'] != source['source']:
                raise ValueError('A source window mixes different documents.')
        if '\n'.join(component['text'] for component in components) != source['text']:
            raise ValueError('The source window differs from its retained components.')


def _compact(text):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text))


def source_display_candidates(evidence_records, description, purpose, requested_operation):
    """Compile exact account-display table rows; leave all other text unhandled.

    This recognizes the *syntax* of an explicit masking row, not whether a rule
    is legally applicable. It needs an account-number field description and a
    display purpose. Numeric column IDs, labels and model output play no role.
    Any conflicting or unrecognized account row suppresses compilation. A rule
    with two permitted positions creates two reviewable alternatives.
    """
    if not all(isinstance(x, str) for x in (description, purpose, requested_operation)):
        return []
    if requested_operation.upper() != 'MASK':
        return []
    field = _compact(description)
    account_field = (re.fullmatch(r'[가-힣]{0,12}계좌번호', field) is not None or
                     re.fullmatch(r'(?:(?:bank|customer|deposit|withdrawal|recipient|sender)\s+)*account\s+number',
                                  description.strip(), re.I) is not None)
    if not account_field or not re.search(r'\bdisplay\b|화면|표시', purpose, re.I):
        return []
    if re.search(r'\b(?:not|never|without)\b|표시하지|표시금지|화면출력금지', purpose, re.I):
        return []
    if not isinstance(evidence_records, list):
        return []
    extracted = []
    expression_pattern = re.compile(r'뒷자리([1-9][0-9]?)개숫자이상(?:또는마지막번호앞([1-9][0-9]?)개숫자이상)?')
    before_pattern = re.compile(r'마지막번호앞([1-9][0-9]?)개숫자이상')
    for source in evidence_records:
        try:
            _validate_source_record(source)
        except ValueError:
            # A malformed selected source is an unresolved integrity problem,
            # not permission to take a different, more convenient clause.
            return []
        text = _compact(source['text'])
        rows = re.findall(r'대상정보:([^:]{1,64})마스킹범위:(.*?)(?=예시:|대상정보:|$)', text)
        if text.count('대상정보:계좌번호') != sum(target == '계좌번호' for target, _ in rows):
            return []
        for target, expression in rows:
            if target != '계좌번호':
                continue
            expression = expression.rstrip('*')  # PDF table's example footnote marker.
            match = expression_pattern.fullmatch(expression)
            before = before_pattern.fullmatch(expression)
            if match:
                requirements = [('suffix', int(match.group(1)))]
                if match.group(2):
                    requirements.append(('before_last', int(match.group(2))))
            elif before:
                requirements = [('before_last', int(before.group(1)))]
            else:
                return []
            if any(count > 64 for _, count in requirements):
                return []
            extracted.append((requirements, source, expression))
    if not extracted or any(row[0] != extracted[0][0] for row in extracted[1:]):
        return []
    requirements, source, expression = extracted[0]
    candidates = []
    for position, count in requirements:
        candidate = compile_display_requirement({'position': position, 'minimum_digits': count,
                                                'source_record': source})
        candidate['parameters'].update({
            'requirement_origin': 'source_table_parser_v1',
            'source_rule_expression': expression,
            'field_description': description,
            'purpose': purpose,
            'requested_operation': requested_operation,
            'review_status': 'pending',
            'legal_applicability_review_required': True,
        })
        candidate['description'] = (f'Source-derived proposal: mask at least {count} digits at {position}. '
                                    'Preserves separators and missing values. '
                                    'Expert selection and legal-applicability review are required.')
        candidate['origin'] = 'source_compiled_pending_review'
        candidates.append(candidate)
    return candidates


def validate_display_candidate(candidate):
    """Check template/parameters/source identity; no legal-adjudication claim."""
    parameters = candidate.get('parameters', {})
    if not isinstance(parameters, dict):
        raise ValueError('Method parameters must be an object.')
    if parameters.get('strategy') != 'digit_position_mask':
        if 'requirement_source' in parameters or parameters.get('requirement_origin') in {'caller_supplied', 'source_table_parser_v1'}:
            raise ValueError('Compiled display-rule strategy metadata was changed.')
        return False
    if parameters.get('requirement_origin') == 'source_table_parser_v1':
        proposals = source_display_candidates(
            [parameters.get('requirement_source')], parameters.get('field_description'),
            parameters.get('purpose'), parameters.get('requested_operation'))
        expected = next((c for c in proposals if c['parameters'].get('position') == parameters.get('position')
                         and c['parameters'].get('minimum_digits') == parameters.get('minimum_digits')), None)
        if expected is None:
            raise ValueError('The selected display rule is not supported by its exact source and field context.')
    else:
        expected = compile_display_requirement({'position': parameters.get('position'),
                                                'minimum_digits': parameters.get('minimum_digits'),
                                                'source_record': parameters.get('requirement_source')})
    if candidate.get('method', candidate.get('action', '')).upper() != 'MASK' or parameters != expected['parameters']:
        raise ValueError('Compiled display-rule metadata was changed.')
    try:
        actual_tree = ast.dump(ast.parse(candidate.get('code_snippet', '')), include_attributes=False)
    except (TypeError, SyntaxError) as exc:
        raise ValueError('Invalid compiled display-rule code.') from exc
    if actual_tree != ast.dump(ast.parse(expected['code_snippet']), include_attributes=False):
        raise ValueError('The selected code no longer implements the compiled display rule.')
    return True
