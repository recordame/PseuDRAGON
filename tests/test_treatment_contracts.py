"""Offline treatment approval and generated-output regression tests."""
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from pseudragon.domain.policy_dsl import ActionType, ColumnPolicy, Policy, PolicyAction
from pseudragon.rag.lexical_evidence import evidence_record
from pseudragon.stages.stage4_code_generation import Stage4CodeGeneration
from pseudragon.validation.display_requirement import (
    compile_display_requirement, source_display_candidates, validate_display_candidate,
)
from pseudragon.validation.method_contract import (
    INTERVAL_1000, MethodContractError, normalize_method_metadata, prepare_method_selection,
)


class TreatmentContractTests(unittest.TestCase):
    def setUp(self):
        self.source = evidence_record({'source': 'fixture-policy', 'priority_level': 1,
            'text': '대상정보: 계좌번호 마스킹범위: 뒷자리3개숫자이상 또는 마지막번호앞3개숫자이상 예시: 합성'})
        self.generator = Stage4CodeGeneration()

    def candidate(self, position='suffix'):
        return compile_display_requirement({'position': position, 'minimum_digits': 3,
                                            'source_record': self.source})

    def generate(self, action, snippet, parameters=None, is_pii=True):
        policy = Policy(table_name='synthetic_accounts')
        policy.columns['value'] = ColumnPolicy(column_name='value', pii_type='test', is_pii=is_pii,
            action=PolicyAction(action=action, parameters=parameters or {}, code_snippet=snippet))
        code = self.generator.generate_code({'columns': ['value']}, policy, {}, log_callback=lambda _: None)
        space = {'__name__': 'generated_test'}
        exec(compile(code, '<generated-test>', 'exec'), space)
        return code, space['pseudonymize_data']

    def test_interval_metadata_is_repaired_without_changing_code(self):
        selected = {'method': 'HASH', 'parameters': {}, 'code_snippet': INTERVAL_1000}
        frozen = copy.deepcopy(selected)
        result = normalize_method_metadata(selected, 'value')
        self.assertEqual(selected, frozen)
        self.assertEqual(result['method'], 'GENERALIZE')
        self.assertEqual(result['parameters']['interval_width'], 1000)
        self.assertEqual(result['code_snippet'], INTERVAL_1000)
        self.assertEqual(result['contract_verification'], 'supported_ast_pattern')

    def test_custom_edit_requires_explicit_contract(self):
        selected = {'method': 'KEEP', 'parameters': {}, 'code_snippet': 'lambda x: x'}
        edit = {'code_modified': True, 'code_snippet': 'lambda x: str(x).upper()'}
        with self.assertRaises(MethodContractError):
            prepare_method_selection(selected, 'value', edit)
        edit['declared_contract'] = {'method': 'GENERALIZE', 'parameters': {'operation': 'uppercase'}}
        self.assertEqual(prepare_method_selection(selected, 'value', edit)['contract_verification'],
                         'user_declared_unverified')
        self.assertEqual(selected['code_snippet'], 'lambda x: x')

    def test_parameterless_mask_cannot_be_selected(self):
        with self.assertRaises(MethodContractError):
            normalize_method_metadata({'method': 'MASK', 'parameters': {}, 'code_snippet': 'lambda x: "***"'}, 'value')

    def test_display_candidate_rejects_code_parameter_or_source_tampering(self):
        self.assertTrue(validate_display_candidate(self.candidate()))
        for mutation in ['code', 'parameters', 'source']:
            candidate = self.candidate()
            if mutation == 'code': candidate['code_snippet'] = 'lambda x: x'
            elif mutation == 'parameters': candidate['parameters']['minimum_digits'] = 2
            else: candidate['parameters']['requirement_source']['text'] += ' altered'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_display_candidate(candidate)

    def test_source_parser_keeps_two_reviewable_alternatives(self):
        candidates = source_display_candidates([self.source], 'account number', 'staff display', 'MASK')
        self.assertEqual([c['parameters']['position'] for c in candidates], ['suffix', 'before_last'])
        for c in candidates:
            self.assertTrue(validate_display_candidate(c))
            self.assertEqual(c['parameters']['review_status'], 'pending')
            self.assertTrue(c['parameters']['legal_applicability_review_required'])
        self.assertEqual(source_display_candidates([self.source], 'account number', 'never display', 'MASK'), [])
        self.assertEqual(source_display_candidates([self.source], 'unrelated field', 'staff display', 'MASK'), [])

    def test_source_parser_rejects_conflicting_requirements(self):
        other = evidence_record({'source': 'other-fixture', 'priority_level': 1,
            'text': '대상정보: 계좌번호 마스킹범위: 뒷자리4개숫자이상 예시: 합성'})
        self.assertEqual(source_display_candidates([self.source, other], 'account number', 'staff display', 'MASK'), [])

    def test_generated_masks_preserve_unicode_separators_and_missing_values(self):
        values = ['123-4567', '12', '', '１２３４５', 'AB-가나다', None, np.nan, pd.NA]
        for position, expected in [
            ('suffix', ['123-4***', '**', '', '１２***', 'AB-가나다']),
            ('before_last', ['123-***7', '**', '', '１***５', 'AB-가나다']),
        ]:
            with self.subTest(position=position):
                c = self.candidate(position)
                _, transform = self.generate(ActionType.MASK, c['code_snippet'], c['parameters'])
                frame = pd.DataFrame({'value': pd.Series(values, dtype=object)})
                before = frame.copy(deep=True)
                actual = transform(frame)
                self.assertEqual(actual['value'].iloc[:5].tolist(), expected)
                self.assertTrue(actual['value'].iloc[5:].isna().all())
                pd.testing.assert_frame_equal(frame, before)

    def test_unguarded_selected_lambda_preserves_missing_inputs(self):
        _, transform = self.generate(ActionType.GENERALIZE, 'lambda x: str(x).upper()')
        output = transform(pd.DataFrame({'value': pd.Series(['abc', None, np.nan, pd.NA], dtype=object)}))
        self.assertEqual(output['value'].iloc[0], 'ABC')
        self.assertTrue(output['value'].iloc[1:].isna().all())

    def test_keep_and_delete_contracts(self):
        frame = pd.DataFrame({'value': [None, 'abc'], 'other': [1, 2]})
        _, keep = self.generate(ActionType.KEEP, '', is_pii=False)
        pd.testing.assert_frame_equal(keep(frame), frame)
        _, delete = self.generate(ActionType.DELETE, '')
        pd.testing.assert_frame_equal(delete(frame), frame[['other']])

    def test_generated_imports_do_not_install_packages(self):
        for action in ActionType:
            policy = Policy(table_name='fixture')
            policy.columns['value'] = ColumnPolicy('value', 'test', False, PolicyAction(action))
            imports = self.generator._build_imports(policy)
            ast.parse(imports)
            self.assertNotIn('subprocess', imports)
            self.assertNotIn('pip', imports)
            if action == ActionType.ENCRYPT:
                self.assertIn('from cryptography.fernet import Fernet', imports)

    def test_encryption_key_is_required_and_never_printed(self):
        policy = Policy(table_name='fixture')
        policy.columns['value'] = ColumnPolicy('value', 'test', True, PolicyAction(ActionType.ENCRYPT))
        init = self.generator._build_encryption_init_code(policy)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'PSEUDRAGON_ENCRYPTION_KEY'):
                exec(init, {'os': os, 'Fernet': Fernet})
        key = Fernet.generate_key().decode('ascii')
        output = io.StringIO()
        with patch.dict(os.environ, {'PSEUDRAGON_ENCRYPTION_KEY': key}), contextlib.redirect_stdout(output):
            space = {'os': os, 'Fernet': Fernet}
            exec(init, space)
            self.assertEqual(space['fernet'].decrypt(space['fernet'].encrypt(b'fixture')), b'fixture')
        self.assertEqual(output.getvalue(), '')


if __name__ == '__main__':
    unittest.main()
