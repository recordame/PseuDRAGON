"""Field-mask editing stays pending until the separate approval step."""

import copy
import unittest
from unittest.mock import patch

from pseudragon.validation.account_mask import masking_preview
from pseudragon.validation.method_contract import prepare_method_selection


class MaskEditorApiTests(unittest.TestCase):
    def setUp(self):
        from web_interface import app as web

        self.web = web
        sessions = patch.object(web.session_manager, "_sessions", {})
        sessions.start()
        self.addCleanup(sessions.stop)
        self.sid = web.session_manager.create_session()
        self.state = web.session_manager.get_session(self.sid)
        self.state.current_step = web.PipelineStep.STAGE3_READY
        self.state.schemas = {"T": {c: {} for c in ("NAME", "DPS_ACNO", "WD_ACNO")}}
        self.state.pii_data = {"T": {c: {"is_pii": True} for c in self.state.schemas["T"]}}
        self.state.policy_data = {
            "T": {
                c: {
                    "is_pii": True,
                    "recommended_methods": [
                        {"method": "HASH", "code_snippet": "original hash"},
                        {"method": "MASK", "code_snippet": "original mask"},
                    ],
                }
                for c in self.state.schemas["T"]
            }
        }
        self.client = web.app.test_client()

    def configure(self, column="NAME", **updates):
        payload = dict(
            session_id=self.sid,
            table="T",
            column=column,
            index=1,
            options={"mode": "keep_edges", "prefix_count": 1, "suffix_count": 2, "mask_char": "#"},
        )
        payload.update(updates)
        return self.client.post("/api/stage3/mask", json=payload)

    def test_any_field_compiles_without_approval_or_other_changes(self):
        before = copy.deepcopy(self.state.policy_data)
        result = self.configure()
        self.assertEqual(result.status_code, 200)
        candidate = result.get_json()["updated"][0]["candidate"]
        checked = prepare_method_selection(candidate, "NAME", 1)
        self.assertEqual(checked["contract_verification"], "compiled_masking_rule")
        self.assertEqual(
            masking_preview(checked, ["1234567890", None, "12"]), ["1#######90", None, "##"]
        )
        self.assertEqual(self.state.current_step, self.web.PipelineStep.STAGE3_READY)
        self.assertEqual(self.state.manual_corrections, [])
        self.assertEqual(
            self.state.pending_mask_edits[("T", "NAME", 1)]["code_snippet"], "original mask"
        )
        self.assertEqual(self.state.policy_data["T"]["DPS_ACNO"], before["T"]["DPS_ACNO"])
        self.assertEqual(
            self.state.policy_data["T"]["NAME"]["recommended_methods"][0],
            before["T"]["NAME"]["recommended_methods"][0],
        )
        self.assertEqual(
            before["T"]["NAME"]["recommended_methods"][1]["code_snippet"], "original mask"
        )

    def test_account_copy_is_explicit_and_consistent(self):
        result = self.configure("DPS_ACNO", apply_matching_accounts=True)
        self.assertEqual(result.status_code, 200)
        rows = result.get_json()["updated"]
        self.assertEqual({row["column"] for row in rows}, {"DPS_ACNO", "WD_ACNO"})
        self.assertEqual(rows[0]["candidate"], rows[1]["candidate"])
        self.assertEqual(
            self.state.policy_data["T"]["NAME"]["recommended_methods"][1]["code_snippet"],
            "original mask",
        )

    def test_invalid_edit_is_atomic_and_stale_reviews_rejected(self):
        before = copy.deepcopy(self.state.policy_data)
        for updates in [
            dict(index=0),
            dict(index=True),
            dict(column="UNKNOWN"),
            dict(options={"mode": "mask_last", "count": 0}),
            dict(apply_matching_accounts=True),
            dict(apply_matching_accounts="yes"),
        ]:
            with self.subTest(updates=updates):
                self.assertEqual(self.configure(**updates).status_code, 400)
                self.assertEqual(before, self.state.policy_data)
        self.assertEqual(self.client.post("/api/stage3/mask", json=["bad"]).status_code, 400)
        self.state.knowledge_invalidated = True
        self.assertEqual(self.configure().status_code, 409)
        self.assertEqual(before, self.state.policy_data)
        self.state.knowledge_invalidated = False
        self.state.current_step = self.web.PipelineStep.COMPLETED
        self.assertEqual(self.configure().status_code, 409)
        self.assertEqual(self.configure(session_id="missing").status_code, 409)
