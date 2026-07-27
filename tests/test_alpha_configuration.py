"""Regression tests for environment and per-run retrieval alpha settings."""

import os
import unittest
from unittest.mock import patch

import numpy as np

from config.config import RAGConfig
from pseudragon.rag.retriever import RAGSystem
from web_interface.app import app


class RAGAlphaConfigurationTest(unittest.TestCase):
    def test_environment_alpha_is_required(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SEMANTIC_SIMILARITY_WEIGHT", None)
            with self.assertRaisesRegex(
                    ValueError,
                    "SEMANTIC_SIMILARITY_WEIGHT must be defined",
            ):
                RAGConfig.get_semantic_similarity_weight()

    def test_per_run_alpha_is_validated(self) -> None:
        ranking = RAGConfig.get_ranking_config(0.2)
        self.assertEqual(ranking["semantic_similarity_weight"], 0.2)
        self.assertEqual(ranking["authority_mixing_weight"], 0.8)

        for invalid in (-0.1, 1.1, "invalid", True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    RAGConfig.get_ranking_config(invalid)

    def test_per_run_alpha_changes_retrieval_order(self) -> None:
        rag = RAGSystem.__new__(RAGSystem)
        rag.embeddings = np.array([[1.0, 0.0], [0.8, 0.6]])
        rag.documents = [
            {"source": "national", "priority_level": 2},
            {"source": "policy", "priority_level": 1},
        ]
        rag.feedback_embeddings = None
        rag.feedback_documents = []
        query = np.array([1.0, 0.0])

        _, similarity_ranked = rag._find_similar_documents(
            query,
            top_k=2,
            exclude_feedback=True,
            semantic_similarity_weight=1.0,
        )
        _, authority_ranked = rag._find_similar_documents(
            query,
            top_k=2,
            exclude_feedback=True,
            semantic_similarity_weight=0.0,
        )

        self.assertEqual(
            similarity_ranked[0]["doc"]["source"],
            "national",
        )
        self.assertEqual(
            authority_ranked[0]["doc"]["source"],
            "policy",
        )

    def test_start_api_passes_validated_alpha_to_run(self) -> None:
        client = app.test_client()

        invalid = client.post(
            "/start",
            json={
                "tables": ["TEST"],
                "semantic_similarity_weight": 1.5,
            },
        )
        self.assertEqual(invalid.status_code, 400)

        with patch.dict(
                os.environ,
                {"SEMANTIC_SIMILARITY_WEIGHT": "0.7"},
        ):
            with patch("web_interface.app.Thread") as thread_class:
                response = client.post(
                    "/start",
                    json={
                        "tables": ["TEST"],
                        "semantic_similarity_weight": 0.2,
                    },
                )
            self.assertEqual(
                os.environ["SEMANTIC_SIMILARITY_WEIGHT"],
                "0.7",
            )

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["semantic_similarity_weight"], 0.2)
        self.assertEqual(body["authority_mixing_weight"], 0.8)
        self.assertEqual(thread_class.call_args.kwargs["args"][-1], 0.2)
        thread_class.return_value.start.assert_called_once()

    def test_runtime_config_and_page_expose_alpha_control(self) -> None:
        client = app.test_client()
        with patch.dict(
                os.environ,
                {"SEMANTIC_SIMILARITY_WEIGHT": "0.65"},
        ):
            runtime = client.get("/api/runtime_config")

        self.assertEqual(runtime.status_code, 200)
        ranking = runtime.get_json()["ranking"]
        self.assertEqual(ranking["alpha_semantic_similarity"], 0.65)
        self.assertEqual(
            ranking["environment_variable"],
            "SEMANTIC_SIMILARITY_WEIGHT",
        )

        page = client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'id="alphaInput"', page.data)
        self.assertIn(b'id="authorityMixPreview"', page.data)


if __name__ == "__main__":
    unittest.main()
