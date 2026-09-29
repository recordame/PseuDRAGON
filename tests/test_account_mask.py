"""Account-mask contracts and the Stage 3 shared-rule boundary."""

import json
import unittest
from unittest.mock import patch
import pandas as pd
from pseudragon.validation.account_mask import account_mask_candidate
from pseudragon.validation.method_contract import prepare_method_selection, MethodContractError


class AccountMaskTests(unittest.TestCase):
    def test_templates_preserve_missing_and_mask_short_values(self):
        for mode, expected in [
            ("keep_last_four", "****5678"),
            ("mask_last_four", "1234****"),
            ("mask_all", "********"),
        ]:
            candidate = account_mask_candidate(mode)
            fn = eval(candidate["code_snippet"], {"pd": pd})
            self.assertEqual(fn("12345678"), expected)
            self.assertEqual(fn("12"), "**")
            self.assertEqual(fn(""), "")
            self.assertIsNone(fn(None))
            self.assertTrue(pd.isna(fn(float("nan"))))
            for column in ["DPS_ACNO", "WD_ACNO"]:
                checked = prepare_method_selection(candidate, column, 0)
                self.assertEqual(checked["parameters"], candidate["parameters"])

    def test_live_model_forms_get_truthful_metadata(self):
        candidates = [
            (
                "lambda x: '*' * (len(str(x)) - 4) + str(x)[-4:] if pd.notna(x) and len(str(x)) > 4 else str(x)",
                "visible_suffix",
                4,
            ),
            (
                "lambda x: str(x)[0] + '*' * (len(str(x)) - 1) if pd.notna(x) and len(str(x)) > 0 else str(x)",
                "visible_prefix",
                1,
            ),
        ]
        for code, key, value in candidates:
            checked = prepare_method_selection(
                {"method": "MASK", "parameters": {}, "code_snippet": code}, "DPS_ACNO", 0
            )
            self.assertEqual(checked["parameters"][key], value)
            self.assertEqual(checked["code_snippet"], code)
        with self.assertRaises(MethodContractError):
            prepare_method_selection(
                {"method": "MASK", "parameters": {}, "code_snippet": "lambda x: str(x)"},
                "DPS_ACNO",
                0,
            )

    def test_shared_rule_updates_candidates_and_rejects_mixed_submission(self):
        from web_interface import app as web

        with patch.object(web.session_manager, "_sessions", {}):
            sid = web.session_manager.create_session()
            state = web.session_manager.get_session(sid)
            state.current_step = web.PipelineStep.STAGE3_READY
            state.schemas = {"T": {"DPS_ACNO": {}, "WD_ACNO": {}}}
            state.pii_data = {"T": {c: {"is_pii": True} for c in state.schemas["T"]}}
            state.policy_data = {
                "T": {
                    c: {"recommended_methods": [account_mask_candidate(mode)]}
                    for c, mode in [("DPS_ACNO", "keep_last_four"), ("WD_ACNO", "mask_all")]
                }
            }
            client = web.app.test_client()
            result = client.post(
                "/submit_stage3",
                json={"session_id": sid, "method_T_DPS_ACNO": 0, "method_T_WD_ACNO": 0},
            )
            self.assertEqual(result.status_code, 400)
            self.assertIn("Shared account MASK", result.get_json()["error"])
            self.assertEqual(state.current_step, web.PipelineStep.STAGE3_READY)
            result = client.post(
                "/api/stage3/account-mask", json={"session_id": sid, "mode": "mask_last_four"}
            )
            self.assertEqual(result.status_code, 200)
            rows = result.get_json()["updated"]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["candidate"], rows[1]["candidate"])
            state.knowledge_invalidated = True
            self.assertEqual(
                client.post(
                    "/api/stage3/account-mask", json={"session_id": sid, "mode": "mask_all"}
                ).status_code,
                409,
            )
