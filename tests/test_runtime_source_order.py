"""Offline regression checks for the real default retrieval and Stage 1 entry."""
from pathlib import Path
import concurrent.futures
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "unused-offline-test")
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", LLM_AUDIT_ENABLED="false")
import numpy as np
from config.config import RAGConfig, Settings
from pseudragon.rag.retriever import (
    RAGSystem,
    RAGError,
    EmbeddingGenerator,
    CacheManager,
    DocumentProcessor,
)
from pseudragon.rag.source_order import SourceOrderAdapter, SourceOrderedRetriever, RELEVANCE_SYSTEM
from pseudragon.rag.lexical_evidence import LegalLexicalIndex
from pseudragon.stages.stage1_pii_detection import Stage1PIIDetection

DOCS = [
    {"source": "institution.pdf", "priority_level": 1, "text": "account display header"},
    {"source": "institution.pdf", "priority_level": 1, "text": "mask account suffix for staff"},
    {"source": "national.pdf", "priority_level": 2, "text": "account display national condition"},
    {
        "source": "international.pdf",
        "priority_level": 3,
        "text": "account display international guidance",
    },
]


def judge(request, accept=True):
    return {
        "passages": [
            {
                "chunk_id": r["chunk_id"],
                "relevant": accept,
                "reason": "Offline fixture relevance judgment",
            }
            for r in request["evidence"]
        ]
    }


def response(value):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=json.dumps(value)), finish_reason="stop"
            )
        ]
    )


class RuntimeSourceOrderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        os.environ.pop("RAG_RETRIEVAL_MODE", None)

    def tearDown(self):
        self.env.stop()

    def test_default_constructor_and_frozen_corpus_do_not_load_embeddings(self):
        with patch.object(
            EmbeddingGenerator, "__init__", side_effect=AssertionError("embedding load")
        ):
            default = RAGSystem(relevance_evaluator=judge)
            rag = RAGSystem.from_documents(DOCS, relevance_evaluator=judge)
        self.assertEqual(RAGConfig.get_retrieval_mode(), "source_order")
        self.assertFalse(default.is_initialized)
        self.assertTrue(rag.is_initialized)
        self.assertIsNone(rag.embedding_generator)
        self.assertIsNone(rag.feedback_embedding_generator)
        self.assertIsNone(rag.embeddings)

    def test_default_real_entry_never_reads_weights_or_dense_rank(self):
        os.environ.update(
            SEMANTIC_SIMILARITY_WEIGHT="invalid-unused", INTERNAL_POLICY_WEIGHT="invalid-unused"
        )
        rag = RAGSystem.from_documents(DOCS, relevance_evaluator=judge)
        with patch.object(
            RAGConfig, "get_ranking_config", side_effect=AssertionError("weight read")
        ), patch.object(
            LegalLexicalIndex, "rank", side_effect=AssertionError("legacy rank")
        ), patch.object(
            rag, "_generate_query_embedding", side_effect=AssertionError("dense query")
        ):
            self.assertEqual(RAGConfig.get_retrieval_config()["source_order"], [1, 2, 3])
            context, sources, details = rag.retrieve_with_details(
                "account display", metadata_filter={"description": "account"}
            )
            self.assertEqual(
                rag.retrieve("account display", metadata_filter={"description": "account"}),
                (context, sources),
            )
        self.assertEqual(sources, ["institution.pdf"])
        self.assertEqual(details.retrieval_result["selected_tier"], 1)
        self.assertNotIn("w=", context)
        self.assertTrue(details.retrieval_result["needs_review"])
        self.assertTrue(details.retrieval_result["legal_applicability_review_required"])
        self.assertFalse(details.retrieval_result["automatic_treatment_approval"])

    def test_exact_adapter_parity_including_task_and_evidence(self):
        seen_production, seen_adapter = [], []

        def j_prod(req):
            seen_production.append(copy.deepcopy(req))
            return judge(req, req["evidence"][0]["priority_level"] == 2)

        def j_old(req):
            seen_adapter.append(copy.deepcopy(req))
            return judge(req, req["evidence"][0]["priority_level"] == 2)

        meta = {"description": "account", "purpose": "staff display", "requested_operation": "MASK"}
        rag = RAGSystem.from_documents(DOCS, relevance_evaluator=j_prod)
        adapter = SourceOrderAdapter(
            SourceOrderedRetriever(DOCS), j_old, "account", "staff display", "MASK"
        )
        got = rag.retrieve_with_details("account decision", metadata_filter=meta)
        expected = adapter.retrieve_with_details("account decision")
        self.assertEqual(got, expected)
        self.assertEqual(got[2].retrieval_result, adapter.last_result)
        self.assertEqual(seen_production, seen_adapter)
        self.assertEqual([r["evidence"][0]["priority_level"] for r in seen_production], [1, 2])

    def test_empty_and_invalid_gate_results_keep_trace_without_source(self):
        for evaluate in (lambda r: judge(r, False), lambda r: {"passages": []}):
            rag = RAGSystem.from_documents(DOCS, relevance_evaluator=evaluate)
            context, sources, details = rag.retrieve_with_details(
                "account", metadata_filter={"description": "account"}
            )
            self.assertEqual((context, sources, details), ("", [], []))
            self.assertEqual(len(details.retrieval_result["trace"]), 3)
            self.assertEqual(details.retrieval_result["status"], "no_relevant_evidence")
            self.assertEqual(rag.last_result, details.retrieval_result)

    def test_empty_corpus_does_not_invoke_evaluator(self):
        rag = RAGSystem.from_documents(
            [], relevance_evaluator=lambda r: self.fail("empty corpus judged")
        )
        _, _, details = rag.retrieve_with_details("account")
        self.assertEqual(len(details.retrieval_result["trace"]), 3)
        self.assertTrue(
            all(t["reason"] == "no_lexical_candidates" for t in details.retrieval_result["trace"])
        )

    def test_source_scope_and_unknown_description(self):
        rag = RAGSystem.from_documents(DOCS, relevance_evaluator=judge)
        _, sources, details = rag.retrieve_with_details(
            "account", metadata_filter={"description": "account", "legal_source_levels": [2]}
        )
        self.assertEqual(sources, ["national.pdf"])
        self.assertEqual(details.retrieval_result["source_order"], [2])
        _, sources, details = rag.retrieve_with_details(
            "account", metadata_filter={"description": "zzzzzzzz"}
        )
        self.assertEqual(sources, [])
        self.assertEqual(len(details.retrieval_result["trace"]), 3)

    def test_evaluator_transport_failure_aborts_and_retains_trace(self):
        def fail(_):
            raise OSError("offline test transport failure")

        rag = RAGSystem.from_documents(DOCS, relevance_evaluator=fail)
        with self.assertRaises(RAGError):
            rag.retrieve_with_details("account")
        self.assertEqual(rag.last_result["status"], "relevance_error_needs_review")
        self.assertEqual(rag.last_result["trace"][0]["action"], "abort")
        self.assertEqual(rag.last_result["evidence"], [])

    def test_session_evaluator_receives_exact_schema_task_without_history(self):
        captured = []

        class FakeSession:
            def __init__(self, **kw):
                self.kw = kw

            def chat(self, **kw):
                captured.append((self.kw, kw))
                return response(judge(json.loads(kw["user_prompt"])))

        rag = RAGSystem.from_documents(DOCS, relevance_session_factory=FakeSession)
        _, _, details = rag.retrieve_with_details(
            "account", metadata_filter={"description": "account"}
        )
        initialization, call = captured[0]
        self.assertEqual(initialization["max_history"], 0)
        self.assertTrue(
            initialization["system_prompt"].startswith(RELEVANCE_SYSTEM + "\nSchema:\n")
        )
        self.assertEqual(call["max_tokens"], 2048)
        self.assertEqual(call["temperature"], 0)
        self.assertEqual(call["response_format"]["json_schema"]["schema"]["required"], ["passages"])
        self.assertEqual(details.retrieval_result["selected_tier"], 1)

    def test_concurrent_returned_and_last_traces_are_per_request(self):
        barrier = threading.Barrier(2)
        rag = RAGSystem.from_documents(DOCS, relevance_evaluator=judge)

        def worker(level):
            _, _, details = rag.retrieve_with_details(
                "account", metadata_filter={"legal_source_levels": [level]}
            )
            barrier.wait(timeout=3)
            return details.retrieval_result["selected_tier"], rag.last_result["selected_tier"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            values = list(pool.map(worker, [1, 3]))
        self.assertEqual(values, [(1, 1), (3, 3)])

    def test_modified_corpus_refreshes_index(self):
        rag = RAGSystem.from_documents(DOCS, relevance_evaluator=judge)
        first = rag.retrieve_with_details("account")[2][0]["evidence_record"]["text"]
        rag.documents[0]["text"] = "account replacement rule"
        second = rag.retrieve_with_details("account")[2][0]["evidence_record"]["text"]
        self.assertNotEqual(first, second)
        self.assertIn("account replacement rule", second)

    def test_explicit_legacy_route_preserves_dense_weighted_selection(self):
        os.environ["RAG_RETRIEVAL_MODE"] = "legacy"
        with patch.object(EmbeddingGenerator, "__init__", return_value=None):
            rag = RAGSystem()
        rag.documents = copy.deepcopy(DOCS)
        rag.embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.5, 0.5]])
        rag.is_initialized = True
        with patch.object(
            rag, "_generate_query_embedding", return_value=np.asarray([1.0, 0.0])
        ), patch.object(rag, "_retrieve_source_order", side_effect=AssertionError("wrong path")):
            context, sources, details = rag.retrieve_with_details("account", top_k=1)
        self.assertEqual(sources, ["institution.pdf"])
        self.assertEqual(details[0]["selection_reason"], "weighted_rank")
        self.assertIsNone(rag.last_result)

    def test_explicit_lexical_route_remains_available(self):
        os.environ["RAG_RETRIEVAL_MODE"] = "lexical"
        with patch.object(EmbeddingGenerator, "__init__", return_value=None):
            rag = RAGSystem()
        rag.documents = copy.deepcopy(DOCS)
        rag.embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [0.5, 0.5]])
        rag.is_initialized = True
        with patch.object(
            rag, "_generate_query_embedding", return_value=np.asarray([1.0, 0.0])
        ), patch.object(rag, "_retrieve_source_order", side_effect=AssertionError("wrong path")):
            _, _, details = rag.retrieve_with_details("account", top_k=1)
        self.assertEqual(details[0]["selection_reason"], "lexical_evidence")

    def test_source_order_uncached_pdf_keeps_legacy_cache_distinct(self):
        rag = RAGSystem(relevance_evaluator=judge)
        with tempfile.TemporaryDirectory() as directory:
            pdf = str(Path(directory) / "test.pdf")
            with patch.object(
                DocumentProcessor, "extract_text_from_pdf", return_value="account display rule"
            ):
                docs, embeddings = rag._process_single_pdf(pdf, 1)
            self.assertTrue(docs)
            self.assertEqual(len(embeddings), 0)
            self.assertFalse(Path(CacheManager.get_cache_path(pdf)).exists())
            self.assertTrue(
                any("source_order_whitespace" in p.name for p in Path(directory).iterdir())
            )

    def test_stage1_empty_retrieval_keeps_trace_without_forcing_negative_label(self):
        class FakeSession:
            def supports_streaming(self):
                return False

            def chat(self, **kw):
                return response(
                    {
                        "is_pii": True,
                        "pii_type": "PII",
                        "confidence": "Medium",
                        "reasoning": "Test positive proposal without retrieved evidence",
                        "evidence_source": "model",
                        "chain_of_thought": {
                            "input_analysis": "test",
                            "key_features": ["test"],
                            "decision_steps": ["test"],
                            "legal_references": [],
                            "final_justification": "test",
                        },
                    }
                )

        obj = Stage1PIIDetection.__new__(Stage1PIIDetection)
        obj.rag = RAGSystem.from_documents(DOCS, relevance_evaluator=lambda r: judge(r, False))
        obj.audit_logger = None
        obj._log = obj._log_cot_reasoning = lambda *args, **kwargs: None
        result = obj._process_single_column_llm(
            FakeSession(), "ACCOUNT", {"type": "VARCHAR", "comment": "account"}, "T"
        )
        self.assertTrue(result["is_pii"])
        self.assertTrue(result["needs_review"])
        self.assertEqual(result["retrieved_evidence"], [])
        self.assertEqual(len(result["retrieval_trace"]["trace"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
