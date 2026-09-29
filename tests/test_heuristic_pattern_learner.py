"""Offline regressions for candidate rules with sparse conflicting feedback."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "heuristic_pattern_learner", ROOT / "pseudragon/rag/heuristic_pattern_learner.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
HeuristicPatternLearner = MODULE.HeuristicPatternLearner


class HeuristicPatternLearnerTests(unittest.TestCase):
    def setUp(self):
        self.learner = HeuristicPatternLearner()
        self.learner.MAX_CONFLICT_RATE = 0.10

    def candidate(self):
        return {
            "prefix": {
                "^TX_": {
                    "count": 3,
                    "confidence": 0.8,
                    "samples": ["TX_DAY", "TX_STATUS", "TX_BANK"],
                }
            },
            "suffix": {},
            "keyword": {},
        }

    def columns(self, count, prefix):
        return [{"column_name": f"TX_{prefix}{index}"} for index in range(count)]

    def test_sparse_opposite_feedback_preserves_account_classification(self):
        feedback = {"column_exact": {}, "pii_classification": {}}
        for name, label in (
            ("TX_ACCOUNT", "PII"),
            ("TX_DAY", "Non-PII"),
            ("TX_STATUS", "Non-PII"),
            ("TX_BANK", "Non-PII"),
        ):
            feedback["column_exact"][name] = {
                "column_name": name,
                "pii_type": label,
                "total_selections": 1,
            }
        with tempfile.TemporaryDirectory() as folder:
            feedback_path = Path(folder) / "feedback.json"
            feedback_path.write_text(json.dumps(feedback))
            learner = HeuristicPatternLearner(
                str(feedback_path), str(Path(folder) / "heuristics.json")
            )
            learner.MAX_CONFLICT_RATE = 0.10
            rules = learner.learn_patterns_from_feedback()
            result = learner.classify_column("TX_ACCOUNT")
            self.assertEqual(result["classification"], "PII")
            self.assertNotIn("^TX_", [rule["regex"] for rule in rules["heuristics"]])

    def test_rare_counterexample_blocks_one_sided_candidate(self):
        positive, negative = self.learner._resolve_conflicts(
            {}, self.candidate(), self.columns(1, "ACCOUNT"), self.columns(3, "DAY")
        )
        self.assertEqual(positive, {})
        self.assertNotIn("^TX_", negative["prefix"])

    def test_exact_conflict_boundary_retains_majority_candidate(self):
        _, negative = self.learner._resolve_conflicts(
            {}, self.candidate(), self.columns(1, "ACCOUNT"), self.columns(9, "DAY")
        )
        self.assertEqual(negative["prefix"]["^TX_"]["conflict_rate"], 0.10)

    def test_above_conflict_boundary_rejects_majority_candidate(self):
        _, negative = self.learner._resolve_conflicts(
            {}, self.candidate(), self.columns(1, "ACCOUNT"), self.columns(8, "DAY")
        )
        self.assertNotIn("^TX_", negative["prefix"])

    def test_minority_candidate_is_not_retained_without_majority_candidate(self):
        positive, negative = self.learner._resolve_conflicts(
            {}, self.candidate(), self.columns(9, "ACCOUNT"), self.columns(1, "DAY")
        )
        self.assertEqual(positive, {})
        self.assertNotIn("^TX_", negative["prefix"])

    def test_unrelated_counterexample_does_not_block_candidate(self):
        _, negative = self.learner._resolve_conflicts(
            {}, self.candidate(), [{"column_name": "OTHER_ACCOUNT"}], self.columns(3, "DAY")
        )
        self.assertEqual(negative["prefix"]["^TX_"]["conflict_rate"], 0)

    def test_refinement_does_not_reintroduce_rejected_suffix(self):
        feedback = {"column_exact": {}, "pii_classification": {}}
        expected = {"A_NO": "PII", "B_NO": "Non-PII", "C_NO": "Non-PII"}
        for name, label in expected.items():
            feedback["column_exact"][name] = {
                "column_name": name,
                "pii_type": label,
                "total_selections": 1,
            }
        with tempfile.TemporaryDirectory() as folder:
            feedback_path = Path(folder) / "feedback.json"
            feedback_path.write_text(json.dumps(feedback))
            learner = HeuristicPatternLearner(
                str(feedback_path), str(Path(folder) / "heuristics.json")
            )
            learner.MAX_CONFLICT_RATE = 0.10
            rules = learner.learn_patterns_from_feedback()
            self.assertNotIn("_NO$", [rule["regex"] for rule in rules["heuristics"]])
            for name, label in expected.items():
                self.assertEqual(learner.classify_column(name)["classification"], label)

    def test_longer_suffix_refinement_checks_all_feedback(self):
        feedback = {"column_exact": {}, "pii_classification": {}}
        for name, label in (
            ("A_BRANCH_NO", "PII"),
            ("B_BRANCH_NO", "Non-PII"),
            ("C_BRANCH_NO", "Non-PII"),
        ):
            feedback["column_exact"][name] = {
                "column_name": name,
                "pii_type": label,
                "total_selections": 1,
            }
        original = {
            "id": "auto_parent",
            "regex": "_NO$",
            "pii_type": "Non-PII",
            "source": "auto_learned",
            "pattern_type": "suffix",
            "enabled": True,
            "sample_columns": ["B_BRANCH_NO", "C_BRANCH_NO"],
            "priority": 80,
        }
        with tempfile.TemporaryDirectory() as folder:
            feedback_path = Path(folder) / "feedback.json"
            feedback_path.write_text(json.dumps(feedback))
            heuristics_path = Path(folder) / "heuristics.json"
            heuristics_path.write_text(
                json.dumps({"version": "2.0", "heuristics": [original], "statistics": {}})
            )
            learner = HeuristicPatternLearner(str(feedback_path), str(heuristics_path))
            learner.MAX_CONFLICT_RATE = 0.10
            rules = learner.refine_heuristics()
            self.assertNotIn("_BRANCH_NO$", [rule["regex"] for rule in rules["heuristics"]])
            self.assertEqual(rules["heuristics"], [original])

    def test_manual_exact_rules_survive_rebuilding_unchanged(self):
        originals = [
            {
                "id": f"manual_{letter}",
                "regex": f"^{letter}_CODE$",
                "pii_type": "PII",
                "source": "manual",
                "pattern_type": "exact",
                "enabled": True,
                "sample_columns": [f"{letter}_CODE"],
                "priority": 100,
            }
            for letter in ("A", "B")
        ]
        with tempfile.TemporaryDirectory() as folder:
            feedback_path = Path(folder) / "feedback.json"
            feedback_path.write_text(json.dumps({"column_exact": {}, "pii_classification": {}}))
            heuristics_path = Path(folder) / "heuristics.json"
            heuristics_path.write_text(
                json.dumps({"version": "2.0", "heuristics": originals, "statistics": {}})
            )
            learner = HeuristicPatternLearner(str(feedback_path), str(heuristics_path))
            rules = learner.learn_patterns_from_feedback()
            self.assertEqual(rules["heuristics"], originals)

    def test_refinement_without_feedback_preserves_existing_rules(self):
        originals = [
            {
                "id": f"auto_{letter}",
                "regex": f"^{letter}_CODE$",
                "pii_type": "PII",
                "source": "auto_learned",
                "pattern_type": "exact",
                "enabled": True,
                "sample_columns": [f"{letter}_CODE"],
                "priority": 50,
            }
            for letter in ("A", "B")
        ]
        with tempfile.TemporaryDirectory() as folder:
            heuristics_path = Path(folder) / "heuristics.json"
            heuristics_path.write_text(
                json.dumps({"version": "2.0", "heuristics": originals, "statistics": {}})
            )
            learner = HeuristicPatternLearner(
                str(Path(folder) / "missing.json"), str(heuristics_path)
            )
            before = heuristics_path.read_bytes()
            rules = learner.refine_heuristics()
            self.assertEqual(rules["heuristics"], originals)
            self.assertEqual(heuristics_path.read_bytes(), before)

    def test_conflicting_exact_approvals_do_not_create_a_rule(self):
        feedback = {
            "column_exact": {
                "positive": {"column_name": "A_X", "pii_type": "PII", "total_selections": 1},
                "negative": {"column_name": "A_X", "pii_type": "Non-PII", "total_selections": 1},
            },
            "pii_classification": {},
        }
        with tempfile.TemporaryDirectory() as folder:
            feedback_path = Path(folder) / "feedback.json"
            feedback_path.write_text(json.dumps(feedback))
            learner = HeuristicPatternLearner(
                str(feedback_path), str(Path(folder) / "heuristics.json")
            )
            learner.MAX_CONFLICT_RATE = 0.10
            rules = learner.learn_patterns_from_feedback()
            self.assertEqual(rules["heuristics"], [])
            self.assertIsNone(learner.classify_column("A_X"))


if __name__ == "__main__":
    unittest.main()
