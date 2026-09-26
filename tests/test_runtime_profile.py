"""Offline API/profile checks; do not import the web app or initialize models."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
NODE = shutil.which("node")


def extracted_namespace():
    config_tree = ast.parse((ROOT / "config/config.py").read_text())
    config_class = next(
        n for n in config_tree.body if isinstance(n, ast.ClassDef) and n.name == "RAGConfig"
    )
    space = {"os": os}
    exec(
        compile(ast.Module(body=[config_class], type_ignores=[]), "<real RAGConfig>", "exec"), space
    )
    app_tree = ast.parse((ROOT / "web_interface/app.py").read_text())
    names = {"_get_runtime_retrieval_profile", "get_runtime_config"}
    functions = [n for n in app_tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(functions) == 2
    for function in functions:
        function.decorator_list = []
    space.update(
        Tuple=tuple,
        jsonify=lambda value: value,
        _count_evidence_pdfs=lambda key: {
            "institutional_policy": 2,
            "national_law": 3,
            "international_regulation": 1,
        }[key],
        _classify_endpoint_scope=lambda _: "loopback",
        AppConfig=types.SimpleNamespace(PAPER_TITLE="PseuDRAGON"),
        Settings=types.SimpleNamespace(
            STAGE_1_OPENAI_BASE_URL="http://localhost:11434",
            RAG_ENABLED=True,
            HEURISTICS_ENABLED=True,
            LLM_AUDIT_ENABLED=True,
            CHUNK_SIZE=128,
            CHUNK_OVERLAP=30,
            TOP_K_RESULTS=3,
            LEGAL_EMBEDDING_MODEL="legal",
            FEEDBACK_EMBEDDING_MODEL="feedback",
            LLM_STAGE_1="gpt-oss:120b",
            LLM_STAGE_2="gpt-oss:120b",
            LLM_TEMPERATURE=0,
        ),
        HeuristicPatternLearner=types.SimpleNamespace(
            MIN_PATTERN_OCCURRENCES=3, MAX_CONFLICT_RATE=0.2
        ),
    )
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), "<real runtime profile endpoint>", "exec"
        ),
        space,
    )
    return space


class RuntimeProfileTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.space = extracted_namespace()

    def response(self, mode=None):
        if mode is not None:
            os.environ["RAG_RETRIEVAL_MODE"] = mode
        return self.space["get_runtime_config"]()

    def test_default_has_no_active_source_weights(self):
        result = self.response()
        self.assertEqual(result["ranking"]["retrieval_mode"], "source_order")
        self.assertEqual(result["ranking"]["source_order"], [1, 2, 3])
        ranking = json.dumps(result["ranking"])
        self.assertNotIn("weight", ranking)
        self.assertNotIn("alpha", ranking)
        self.assertNotIn("weighted", " ".join(result["workflow"]))

    def test_source_order_does_not_read_legacy_coefficients(self):
        for name in [
            "SEMANTIC_SIMILARITY_WEIGHT",
            "EXPERT_FEEDBACK_WEIGHT",
            "INTERNAL_POLICY_WEIGHT",
            "NATIONAL_LAW_WEIGHT",
            "INTERNATIONAL_REGULATIONS_WEIGHT",
        ]:
            os.environ[name] = "invalid-unused-value"
        with patch.object(
            self.space["RAGConfig"],
            "get_ranking_config",
            side_effect=AssertionError("inactive weights read"),
        ):
            self.assertEqual(self.response()["status"], "success")

    def test_source_order_requires_review_and_preserves_counts(self):
        ranking = self.response()["ranking"]
        self.assertFalse(ranking["automatic_treatment_approval"])
        self.assertTrue(ranking["legal_applicability_review_required"])
        self.assertIn("otherwise try the next", ranking["summary"])
        self.assertEqual(sum(r["document_count"] for r in ranking["source_families"]), 6)
        self.assertEqual([r["priority_level"] for r in ranking["source_families"]], [1, 2, 3])

    def test_legacy_exposes_actual_overrides(self):
        os.environ["SEMANTIC_SIMILARITY_WEIGHT"] = ".9"
        os.environ["INTERNAL_POLICY_WEIGHT"] = ".75"
        ranking = self.response("legacy")["ranking"]
        self.assertEqual(ranking["alpha_semantic_similarity"], 0.9)
        self.assertEqual(ranking["authority_mixing_weight"], 0.1)
        self.assertEqual(ranking["source_families"][1]["weight"], 0.75)
        self.assertIn("legacy", ranking["label"])

    def test_lexical_does_not_present_legacy_legal_formula(self):
        os.environ["RAG_LEXICAL_SOURCE_GAP"] = ".2"
        ranking = self.response("lexical")["ranking"]
        self.assertEqual(ranking["source_gap"], 0.2)
        self.assertIn("source_level", ranking["formula"])
        self.assertNotIn("alpha_semantic_similarity", ranking)
        self.assertEqual(ranking["feedback_ranking"]["alpha_semantic_similarity"], 0.7)
        self.assertTrue(all("weight" not in row for row in ranking["source_families"]))

    def test_invalid_mode_is_an_error(self):
        result, status = self.response("unknown")
        self.assertEqual(status, 500)
        self.assertEqual(result["status"], "error")

    def test_legacy_invalid_coefficient_is_an_error(self):
        os.environ["SEMANTIC_SIMILARITY_WEIGHT"] = "invalid"
        result, status = self.response("legacy")
        self.assertEqual(status, 500)
        self.assertEqual(result["status"], "error")

    def test_initial_html_is_mode_neutral(self):
        text = (ROOT / "web_interface/templates/index.html").read_text()
        cascade = text.split('class="runtime-cascade">', 1)[1].split("</p>", 1)[0]
        self.assertNotIn("weighted", cascade)
        self.assertIn("<span>Retrieval</span>", text)

    def test_actual_javascript_renders_real_endpoint_payloads(self):
        os.environ["RAG_LEXICAL_SOURCE_GAP"] = ".2"
        payloads = {mode: self.response(mode) for mode in ["source_order", "legacy", "lexical"]}
        if NODE is None:
            self.skipTest("Install Node.js to run the UI rendering checks")
        result = subprocess.run(
            [NODE, str(HERE / "test_runtime_profile.js"), str(ROOT)],
            input=json.dumps(payloads),
            text=True,
            capture_output=True,
            check=True,
            timeout=15,
        )
        self.assertIn("5 runtime rendering checks passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
