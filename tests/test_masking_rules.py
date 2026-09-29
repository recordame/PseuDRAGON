"""User-selected mask behavior and compiler contract boundaries."""

import copy
import unittest

import pandas as pd

from pseudragon.validation.account_mask import (
    masking_candidate,
    masking_preview,
    validate_masking_candidate,
)
from pseudragon.validation.method_contract import MethodContractError, prepare_method_selection


class MaskingRuleTests(unittest.TestCase):
    def generated_transform(self, candidate):
        from pseudragon.domain.policy_dsl import ActionType, ColumnPolicy, Policy, PolicyAction
        from pseudragon.stages.stage4_code_generation import Stage4CodeGeneration

        selected = prepare_method_selection(candidate, "value", 0)
        policy = Policy(table_name="synthetic_mask_inputs")
        policy.columns["value"] = ColumnPolicy(
            column_name="value",
            pii_type="PII",
            is_pii=True,
            action=PolicyAction(
                action=ActionType.MASK,
                parameters=selected["parameters"],
                code_snippet=selected["code_snippet"],
            ),
        )
        code = Stage4CodeGeneration().generate_code(
            {"columns": ["value", "untouched"]}, policy, {}, log_callback=lambda _: None
        )
        namespace = {"__name__": "generated_mask_test"}
        exec(compile(code, "<generated-mask-test>", "exec"), namespace)
        return namespace["pseudonymize_data"]

    def test_generated_programs_apply_all_six_modes_and_preserve_missing(self):
        values = ["12345678", "AB-가나다", "１２３４５", "12", "", None, float("nan"), pd.NA]
        frame = pd.DataFrame(
            {"value": pd.Series(values, dtype=object), "untouched": list(range(len(values)))}
        )
        original = frame.copy(deep=True)
        expectations = {
            "keep_last": ["****5678", "**-가나다", "*２３４５", "**", ""],
            "keep_first": ["1234****", "AB-가**", "１２３４*", "**", ""],
            "mask_last": ["1234****", "AB****", "１****", "**", ""],
            "mask_first": ["****5678", "****나다", "****５", "**", ""],
            "keep_edges": ["1***5678", "A*-가나다", "*****", "**", ""],
            "mask_all": ["********", "******", "*****", "**", ""],
        }
        for mode, expected in expectations.items():
            with self.subTest(mode=mode):
                transform = self.generated_transform(masking_candidate(mode))
                actual = transform(frame)
                self.assertEqual(actual["value"].iloc[:5].tolist(), expected)
                self.assertTrue(actual["value"].iloc[5:].isna().all())
                pd.testing.assert_series_equal(actual["untouched"], original["untouched"])
                pd.testing.assert_frame_equal(frame, original)

    def test_generated_program_preserves_configured_counts_and_mask_characters(self):
        for char in ["•", "'", '"', "\\"]:
            with self.subTest(mask_char=char):
                candidate = masking_candidate(
                    "keep_edges", prefix_count=2, suffix_count=2, mask_char=char
                )
                transform = self.generated_transform(candidate)
                actual = transform(pd.DataFrame({"value": ["AB-1234", "1234"]}))
                self.assertEqual(actual["value"].tolist(), ["AB" + char * 3 + "34", char * 4])

    def test_modes_match_reviewed_examples_and_preserve_missing(self):
        expectations = {
            "keep_last": "******7890",
            "keep_first": "1234******",
            "mask_last": "123456****",
            "mask_first": "****567890",
            "keep_edges": "1*****7890",
            "mask_all": "**********",
        }
        for mode, output in expectations.items():
            with self.subTest(mode=mode):
                candidate = masking_candidate(mode)
                self.assertTrue(validate_masking_candidate(candidate))
                values = masking_preview(
                    candidate, ["1234567890", "12", "", None, float("nan"), pd.NA]
                )
                self.assertEqual(values[:3], [output, "**", ""])
                self.assertIsNone(values[3])
                self.assertTrue(pd.isna(values[4]))
                self.assertIs(values[5], pd.NA)
                checked = prepare_method_selection(candidate, "ANY_FIELD", 0)
                self.assertEqual(checked["parameters"], candidate["parameters"])
                self.assertEqual(checked["contract_verification"], "compiled_masking_rule")

    def test_counts_and_characters_are_explicit_and_keep_modes_hide_boundaries(self):
        cases = [
            ({"mode": "keep_first", "count": 2, "mask_char": "#"}, "AB###"),
            ({"mode": "keep_last", "count": 2, "mask_char": "#"}, "###CD"),
            ({"mode": "mask_first", "count": 2, "mask_char": "#"}, "##-CD"),
            ({"mode": "mask_last", "count": 2, "mask_char": "#"}, "AB-##"),
            ({"mode": "keep_edges", "prefix_count": 1, "suffix_count": 1}, "A***D"),
            ({"mode": "keep_edges", "prefix_count": 0, "suffix_count": 2}, "***CD"),
            ({"mode": "keep_edges", "prefix_count": 2, "suffix_count": 0}, "AB***"),
        ]
        for options, expected in cases:
            with self.subTest(options=options):
                candidate = masking_candidate(**options)
                self.assertEqual(masking_preview(candidate, ["AB-CD"]), [expected])
        for mode in ["keep_first", "keep_last"]:
            self.assertEqual(masking_preview(masking_candidate(mode, count=4), ["1234"]), ["****"])
            self.assertEqual(masking_preview(masking_candidate(mode, count=0), ["1234"]), ["****"])
        self.assertEqual(
            masking_preview(
                masking_candidate("keep_edges", prefix_count=2, suffix_count=2), ["1234"]
            ),
            ["****"],
        )
        self.assertEqual(
            masking_preview(
                masking_candidate("keep_edges", prefix_count=0, suffix_count=0), ["1234"]
            ),
            ["****"],
        )

    def test_unicode_and_quoted_mask_characters_remain_data(self):
        for char in ["•", "'", '"', "\\", "🔒"]:
            with self.subTest(mask_char=char):
                candidate = masking_candidate("keep_first", count=1, mask_char=char)
                self.assertEqual(
                    masking_preview(candidate, ["홍길동", 123]), ["홍" + char * 2, "1" + char * 2]
                )
                self.assertTrue(validate_masking_candidate(candidate))

    def test_invalid_options_fail_before_program_creation(self):
        invalid_options = [
            {"mode": "unknown"},
            {"mode": []},
            *({"mode": "keep_last", "count": value} for value in [-1, 65, True, 4.0, "4", None]),
            {"mode": "mask_last", "count": 0},
            {"mode": "mask_first", "count": 0},
            {"mode": "keep_edges", "prefix_count": 65},
            {"mode": "keep_edges", "suffix_count": -1},
            *(
                {"mode": "mask_all", "mask_char": value}
                for value in ["", "**", " ", "\n", "\x00", None]
            ),
        ]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                masking_candidate(**options)

    def test_forged_or_edited_compiled_candidate_is_rejected(self):
        original = masking_candidate("keep_last", count=3)
        variants = []
        changed = copy.deepcopy(original)
        changed["code_snippet"] = "lambda x: str(x)"
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["parameters"]["count"] = 2
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["masking_rule"]["count"] = 2
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["masking_rule"]["suffix_count"] = 3
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["method"] = "KEEP"
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["action"] = "KEEP"
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["code_snippet"] = None
        variants.append(changed)
        for candidate in variants:
            with self.subTest(candidate=candidate), self.assertRaises(MethodContractError):
                prepare_method_selection(candidate, "FIELD", 0)
        with self.assertRaises(MethodContractError):
            prepare_method_selection(
                original,
                "FIELD",
                {"code_modified": True, "code_snippet": "lambda x: str(x)"},
            )
        self.assertEqual(original["parameters"]["count"], 3)

    def test_equivalent_ast_and_matching_declaration_are_accepted(self):
        candidate = masking_candidate("mask_first", count=2)
        candidate["code_snippet"] = (
            candidate["code_snippet"].replace("'*'", '"*"') + " # approved rule\n"
        )
        checked = prepare_method_selection(
            candidate,
            "FIELD",
            {"declared_contract": {"method": "MASK", "parameters": candidate["parameters"]}},
        )
        self.assertEqual(checked["contract_verification"], "compiled_masking_rule")
        with self.assertRaises(MethodContractError):
            prepare_method_selection(
                candidate,
                "FIELD",
                {"declared_contract": {"method": "KEEP", "parameters": {}}},
            )


if __name__ == "__main__":
    unittest.main()
