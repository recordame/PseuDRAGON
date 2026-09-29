"""Approved masking contracts persist and are proposed as a complete unit."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pseudragon.rag.expert_preference_manager import ExpertPreferenceManager
from pseudragon.rag.knowledge_store import atomic_json, revision
from pseudragon.validation.account_mask import masking_candidate, masking_preview


class MaskPreferenceReuseTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.feedback = str(Path(directory.name) / "feedback.json")
        self.rules = str(Path(directory.name) / "rules.json")
        for context in [
            patch.dict(os.environ, EXPERT_FEEDBACK_FILE=self.feedback, HEURISTICS_FILE=self.rules),
            patch.object(ExpertPreferenceManager, "_initialize_feedback_embeddings"),
        ]:
            context.start()
            self.addCleanup(context.stop)
        self.manager = ExpertPreferenceManager()

    def approve(self, candidate):
        self.manager.record_selection(
            "TABLE",
            "ACCOUNT",
            "PII",
            "MASK",
            ["MASK", "HASH"],
            0,
            code_snippet=candidate["code_snippet"],
            code_modified=False,
            is_pii=True,
            approved_mask_candidate=candidate,
        )

    def test_selection_without_code_edits_reuses_complete_contract_after_reload(self):
        candidate = masking_candidate("keep_first", count=2, mask_char="#")
        self.approve(candidate)
        loaded = ExpertPreferenceManager()
        fresh = [
            {"method": "HASH", "parameters": {}, "code_snippet": "hash proposal"},
            masking_candidate("mask_all"),
        ]
        original = copy.deepcopy(fresh)
        proposals = loaded.reorder_methods("ACCOUNT", "PII", fresh)
        self.assertEqual(proposals[0]["masking_rule"], candidate["masking_rule"])
        self.assertEqual(proposals[0]["parameters"], candidate["parameters"])
        self.assertEqual(proposals[0]["code_snippet"], candidate["code_snippet"])
        self.assertEqual(masking_preview(proposals[0], ["12345678"]), ["12######"])
        self.assertEqual(fresh, original)
        preference = loaded.get_preference_from_similar("ACCOUNT", "PII")
        self.assertEqual(
            preference["approved_mask_candidate"]["masking_rule"], candidate["masking_rule"]
        )
        self.assertIsNone(preference["code_snippet"])
        proposals[0]["parameters"]["count"] = 50
        self.assertEqual(
            loaded.reorder_methods("ACCOUNT", "PII", fresh)[0]["parameters"]["count"], 2
        )

    def test_updates_keep_history_and_manual_edit_replaces_active_compiled_rule(self):
        self.approve(masking_candidate("keep_first", count=2))
        self.approve(masking_candidate("keep_last", count=3))
        self.manager.record_selection(
            "TABLE",
            "ACCOUNT",
            "PII",
            "MASK",
            ["MASK"],
            0,
            code_snippet="lambda x: 'manual'",
            code_modified=True,
        )
        data = json.loads(Path(self.feedback).read_text())
        entry = data["column_exact"]["account:PII"]
        self.assertNotIn("approved_mask_candidate", entry)
        self.assertEqual(len(entry["mask_approval_history"]), 3)
        self.assertEqual(
            entry["mask_approval_history"][0]["candidate"]["masking_rule"]["mode"], "keep_first"
        )
        self.assertEqual(
            entry["mask_approval_history"][1]["candidate"]["masking_rule"]["mode"], "keep_last"
        )
        self.assertIsNone(entry["mask_approval_history"][2]["candidate"])
        self.assertEqual(
            self.manager.get_custom_code_snippet("ACCOUNT", "PII", "MASK"), "lambda x: 'manual'"
        )

    def test_configured_rules_do_not_transfer_to_similar_names(self):
        self.approve(masking_candidate("keep_last", count=3))
        with patch.object(
            self.manager, "find_similar_column", return_value=("account:PII", "ACCOUNT", 0.99)
        ):
            preference = self.manager.get_preference_from_similar("OTHER_ACCOUNT", "PII")
        self.assertEqual(preference["match_type"], "rag_similar")
        self.assertIsNone(preference["code_snippet"])
        self.assertNotIn("approved_mask_candidate", preference)
        fresh = [masking_candidate("mask_all")]
        self.assertEqual(
            self.manager.apply_reviewed_mask_candidate("OTHER_ACCOUNT", "PII", fresh), fresh
        )
        self.assertEqual(
            self.manager.apply_reviewed_mask_candidate("ACCOUNT", "Non-PII", fresh), fresh
        )

    def test_invalid_candidate_is_rejected_before_any_store_write(self):
        candidate = masking_candidate("keep_first", count=2)
        candidate["code_snippet"] = "lambda x: str(x)"
        with self.assertRaises(ValueError):
            self.approve(candidate)
        self.assertFalse(Path(self.feedback).exists())
        self.assertEqual(self.manager.preferences["statistics"]["total_selections"], 0)

    def test_withdrawal_removes_reuse_and_archives_full_contract(self):
        candidate = masking_candidate("keep_edges", prefix_count=1, suffix_count=2)
        self.approve(candidate)
        self.manager.manage_knowledge(
            "withdraw",
            revision(self.feedback),
            "Changed handling requirement",
            section="column_exact",
            key="account:PII",
        )
        fresh = [masking_candidate("mask_all")]
        self.assertEqual(self.manager.apply_reviewed_mask_candidate("ACCOUNT", "PII", fresh), fresh)
        saved = json.loads(Path(self.feedback).read_text())
        archived = saved["knowledge_history"][-1]["archived"]["column_exact"]["account:PII"]
        self.assertEqual(archived["approved_mask_candidate"]["parameters"], candidate["parameters"])

    def test_corrupted_saved_rule_falls_back_to_new_candidates(self):
        candidate = masking_candidate("keep_last", count=2)
        self.approve(candidate)
        data = json.loads(Path(self.feedback).read_text())
        entry = data["column_exact"]["account:PII"]
        entry["approved_mask_candidate"]["code_snippet"] = "lambda x: str(x)"
        entry["custom_code_snippets"] = {"MASK": "lambda x: 'old manual rule'"}
        atomic_json(self.feedback, data)
        fresh = [masking_candidate("mask_all")]
        self.assertEqual(self.manager.apply_reviewed_mask_candidate("ACCOUNT", "PII", fresh), fresh)
        self.assertIsNone(
            self.manager.get_preference_from_similar("ACCOUNT", "PII")["code_snippet"]
        )


if __name__ == "__main__":
    unittest.main()
