"""Product checks for reviewed reuse, withdrawal, and policy-scope changes."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "unused-offline-test")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LLM_AUDIT_ENABLED="false")
from pseudragon.rag.expert_preference_manager import ExpertPreferenceManager
from pseudragon.rag.knowledge_store import atomic_json, revision


class KnowledgeManagementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.feedback = str(Path(self.directory.name) / "feedback.json")
        self.rules = str(Path(self.directory.name) / "rules.json")
        self.env = patch.dict(
            os.environ, EXPERT_FEEDBACK_FILE=self.feedback, HEURISTICS_FILE=self.rules
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        self.embeddings = patch.object(ExpertPreferenceManager, "_initialize_feedback_embeddings")
        self.embeddings.start()
        self.addCleanup(self.embeddings.stop)
        self.manager = ExpertPreferenceManager()

    def approve(self, name, label=True):
        self.manager.record_selection(
            "TABLE",
            name,
            "PII" if label else "Non-PII",
            "HASH" if label else "KEEP",
            [],
            0,
            is_pii=label,
        )

    def manage(self, action, **kwargs):
        return self.manager.manage_knowledge(
            action, revision(self.feedback), "Reviewed change", **kwargs
        )

    def test_confirmed_defaults_reuse_without_corrections(self):
        self.approve("CUST_NO")
        self.assertEqual(
            self.manager.get_learned_pii_classification("cust_no")["classification"], "PII"
        )
        self.assertIsNone(self.manager.get_learned_pii_classification("UNSEEN"))

    def test_conflicting_approvals_defer(self):
        self.approve("FIELD", True)
        self.approve("FIELD", False)
        self.assertEqual(
            self.manager.get_learned_pii_classification("FIELD")["match_type"], "conflict"
        )

    def test_withdrawal_preserves_history_and_rebuilds(self):
        self.approve("CUST_NO")
        self.approve("ACCOUNT_NO")
        self.manager.trigger_pattern_learning()
        self.manage("withdraw", section="column_exact", key="cust_no:PII")
        self.assertIsNone(self.manager.get_learned_pii_classification("CUST_NO"))
        data = json.loads(Path(self.feedback).read_text())
        self.assertIn("cust_no:PII", data["knowledge_history"][-1]["archived"]["column_exact"])
        self.assertEqual(data["statistics"]["total_selections"], 1)
        rules = json.loads(Path(self.rules).read_text())
        clean = str(Path(self.directory.name) / "clean.json")
        from pseudragon.rag.heuristic_pattern_learner import HeuristicPatternLearner

        learner = HeuristicPatternLearner(self.feedback, clean)
        learner.learn_patterns_from_feedback()
        learner.refine_heuristics()
        semantic = lambda rows: [
            {k: v for k, v in row.items() if k not in {"created_at", "updated_at"}} for row in rows
        ]
        self.assertEqual(
            semantic(rules["heuristics"]),
            semantic(json.loads(Path(clean).read_text())["heuristics"]),
        )

    def test_reset_archives_manual_rules_and_refreshes_retained_manager(self):
        self.approve("FIELD")
        retained = ExpertPreferenceManager()
        manual = {
            "id": "manual",
            "source": "manual",
            "enabled": True,
            "pattern": "^FIELD$",
            "pii_type": "PII",
            "priority": 100,
            "name": "Manual field",
        }
        atomic_json(self.rules, {"version": "2.0", "heuristics": [manual], "statistics": {}})
        self.manage("reset")
        self.assertIsNone(retained.get_learned_pii_classification("FIELD"))
        self.assertEqual(json.loads(Path(self.rules).read_text())["heuristics"], [])
        self.assertEqual(
            json.loads(Path(self.feedback).read_text())["knowledge_history"][-1]["archived"][
                "rules"
            ]["heuristics"],
            [manual],
        )
        retained.record_selection("T", "NEW", "PII", "HASH", [], 0)
        self.assertNotIn("field:PII", json.loads(Path(self.feedback).read_text())["column_exact"])

    def test_stale_revision_and_failed_rebuild_leave_active_store_unchanged(self):
        self.approve("FIELD")
        old = Path(self.feedback).read_bytes()
        with self.assertRaises(ValueError):
            self.manager.manage_knowledge("reset", "stale", "Change")
        from pseudragon.rag.heuristic_pattern_learner import HeuristicPatternLearner

        with patch.object(
            HeuristicPatternLearner,
            "learn_patterns_from_feedback",
            side_effect=RuntimeError("failed"),
        ):
            with self.assertRaises(RuntimeError):
                self.manage("reset")
        self.assertEqual(old, Path(self.feedback).read_bytes())

    def test_failed_activation_restores_both_stores(self):
        self.approve("FIELD")
        self.manager.trigger_pattern_learning()
        before_feedback = json.loads(Path(self.feedback).read_text())
        before_rules = json.loads(Path(self.rules).read_text())
        original_write = atomic_json
        failed = False

        def failing_write(path, data):
            nonlocal failed
            if str(path) == self.feedback and not failed:
                failed = True
                raise OSError("simulated activation failure")
            original_write(path, data)

        with patch(
            "pseudragon.rag.expert_preference_manager.atomic_json", side_effect=failing_write
        ):
            with self.assertRaises(OSError):
                self.manage("reset")
        self.assertEqual(json.loads(Path(self.feedback).read_text()), before_feedback)
        self.assertEqual(json.loads(Path(self.rules).read_text()), before_rules)

    def test_two_retained_managers_do_not_overwrite_new_approvals(self):
        from concurrent.futures import ThreadPoolExecutor

        second = ExpertPreferenceManager()

        def approve(index):
            manager = self.manager if index % 2 else second
            manager.record_selection("T", f"FIELD_{index}", "PII", "HASH", [], 0)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(approve, range(20)))
        self.assertEqual(len(json.loads(Path(self.feedback).read_text())["column_exact"]), 20)

    def test_stage1_conflict_bypasses_rules_and_requests_model(self):
        from unittest.mock import Mock
        from pseudragon.stages.stage1_pii_detection import Stage1PIIDetection

        self.approve("FIELD", True)
        self.approve("FIELD", False)
        stage = Stage1PIIDetection.__new__(Stage1PIIDetection)
        stage.heuristics_enabled = True
        stage.expert_preference_manager = self.manager
        stage.system_prompt = "Test"
        session = Mock(provider="test")
        session.get_stats.return_value = {"estimated_tokens_saved": 0}
        stage.session_manager = Mock()
        stage.session_manager.get_or_create_session.return_value = session
        stage._log = Mock()
        stage._process_single_column_llm = Mock(return_value={"is_pii": True})
        stage._heuristic_pii = Mock(side_effect=AssertionError("Conflict reached static rules"))
        with patch.object(
            self.manager,
            "classify_by_heuristics",
            side_effect=AssertionError("Conflict reached learned rules"),
        ):
            result = stage.identify_pii({"FIELD": {"type": "TEXT"}}, "T")
        stage._process_single_column_llm.assert_called_once()
        self.assertTrue(result["FIELD"]["is_pii"])

    def test_web_management_rejects_busy_and_stale_approvals(self):
        from web_interface import app as web

        self.approve("FIELD")
        with (
            patch.object(web.session_manager, "_sessions", {}),
            patch.object(web.session_manager, "_expert_preference_manager", self.manager),
        ):
            client = web.app.test_client()
            sid = web.session_manager.create_session()
            state = web.session_manager.get_session(sid)
            state.is_running = True
            request = dict(action="reset", revision=revision(self.feedback), reason="New purpose")
            self.assertEqual(
                client.post("/api/expert_feedback/manage", json=request).status_code, 409
            )
            state.is_running = False
            self.assertEqual(
                client.post("/api/expert_feedback/manage", json=request).status_code, 200
            )
            self.assertEqual(
                client.post("/submit_stage3", json={"session_id": sid}).status_code, 409
            )
            response = client.get("/api/expert_feedback").get_json()
            self.assertEqual(response["method_preferences"], [])
            self.assertEqual(response["history"][-1]["action"], "reset")


if __name__ == "__main__":
    unittest.main()
