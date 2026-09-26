from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pseudragon.rag.source_order import (
    SourceOrderedRetriever,
    SourceOrderAdapter,
    validate_relevance,
)
from pseudragon.rag.lexical_evidence import LegalLexicalIndex


DOCS = [
    {"source": "institution", "priority_level": 1, "text": "account display header"},
    {"source": "institution", "priority_level": 1, "text": "mask account number suffix"},
    {"source": "national", "priority_level": 2, "text": "account display general rule"},
    {"source": "international", "priority_level": 3, "text": "account display guidance"},
]


def judged(request, accept=False):
    return {
        "passages": [
            {"chunk_id": r["chunk_id"], "relevant": accept, "reason": "Unit-test evaluator"}
            for r in request["evidence"]
        ]
    }


class SourceOrderTests(unittest.TestCase):
    def setUp(self):
        self.r = SourceOrderedRetriever(DOCS)

    def test_relevant_institution_stops(self):
        r = self.r.retrieve("account", "account display", lambda x: judged(x, True))
        self.assertEqual(r["selected_tier"], 1)
        self.assertEqual(len(r["trace"]), 1)
        self.assertFalse(r["automatic_treatment_approval"])

    def test_irrelevant_institution_moves_on_without_polluting_next_tier(self):
        seen = []

        def judge(request):
            levels = {r["priority_level"] for r in request["evidence"]}
            seen.append(levels)
            return judged(request, levels == {2})

        r = self.r.retrieve("account", "account display", judge)
        self.assertEqual(r["selected_tier"], 2)
        self.assertEqual(seen, [{1}, {2}])

    def test_all_irrelevant_leaves_empty_context_and_trace(self):
        adapter = SourceOrderAdapter(self.r, judged, "account")
        context, sources, details = adapter.retrieve_with_details("account display")
        self.assertEqual((context, sources, details), ("", [], []))
        self.assertEqual(len(adapter.last_result["trace"]), 3)

    def test_invalid_gate_cannot_select_passage(self):
        for output in [
            {"passages": []},
            {"passages": [{"chunk_id": "fake", "relevant": True, "reason": "x"}]},
            None,
        ]:
            r = self.r.retrieve("account", "account display", lambda _: output)
            self.assertEqual(r["evidence"], [])

    def test_duplicate_id_and_string_bool_rejected(self):
        records = self.r.candidates("account", 1)
        d = judged({"evidence": records}, True)
        d["passages"][0]["relevant"] = "true"
        self.assertFalse(validate_relevance(d, records)["valid"])
        records = records + [dict(records[0], chunk_id="second")]
        d = judged({"evidence": records}, True)
        d["passages"][1]["chunk_id"] = d["passages"][0]["chunk_id"]
        self.assertFalse(validate_relevance(d, records)["valid"])

    def test_window_preserves_source_boundary_and_merges_overlap(self):
        records = self.r.candidates("account", 1)
        self.assertEqual(len(records), 1)
        self.assertEqual(
            [r["text"] for r in records[0]["component_records"]], [d["text"] for d in DOCS[:2]]
        )
        self.assertEqual(records[0]["text"], "\n".join(d["text"] for d in DOCS[:2]))

    def test_query_scope_is_respected(self):
        r = self.r.retrieve("account", "GDPR의 본문 또는 전문에서 account", lambda x: judged(x, True))
        self.assertEqual(r["selected_tier"], 3)

    def test_explicit_empty_scope_calls_no_judge(self):
        r = self.r.retrieve("account", "account", lambda _: self.fail(), allowed_levels=[])
        self.assertEqual(r["trace"], [])

    def test_zero_match_does_not_invent_evidence(self):
        r = self.r.retrieve("zzzzzz", "zzzzzz", lambda _: self.fail())
        self.assertEqual(r["evidence"], [])

    def test_no_authority_formula_or_scores_in_gate_or_context(self):
        def judge(request):
            for r in request["evidence"]:
                self.assertFalse({"score", "similarity", "weight", "alpha"} & set(r))
            return judged(request, True)

        with patch.object(
            LegalLexicalIndex, "rank", side_effect=AssertionError("weighted rank used")
        ):
            context, _, details = SourceOrderAdapter(
                self.r, judge, "account"
            ).retrieve_with_details("account")
        self.assertNotIn("similarity:", context)
        self.assertEqual(details[0]["selection_reason"], "source_order_relevance")

    def test_input_mutation_does_not_change_index(self):
        import copy

        docs = copy.deepcopy(DOCS)
        retriever = SourceOrderedRetriever(docs)
        docs[0]["text"] = "mutated"
        self.assertEqual(retriever.documents[0]["text"], DOCS[0]["text"])


if __name__ == "__main__":
    unittest.main()
