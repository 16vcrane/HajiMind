import sys
import unittest

sys.path.insert(0, "backend")

from rag_conflicts import detect_document_conflicts


class RagConflictTests(unittest.TestCase):
    def test_same_claim_across_documents_is_not_a_conflict(self):
        result = detect_document_conflicts(
            [
                {"filename": "guide-a.pdf", "page_number": 1, "text": "HajiMind 使用 LangGraph。"},
                {"filename": "guide-b.pdf", "page_number": 2, "text": "HajiMind 使用 LangGraph。"},
            ]
        )

        self.assertFalse(result["has_conflict"])
        self.assertEqual(result["claims"], [])

    def test_conflicting_claims_include_sources_evidence_and_time(self):
        result = detect_document_conflicts(
            [
                {
                    "filename": "architecture-a.pdf",
                    "page_number": 1,
                    "chunk_id": "a-1",
                    "document_date": "2026-01-01",
                    "text": "HajiMind 使用 LangGraph。",
                },
                {
                    "filename": "architecture-b.pdf",
                    "page_number": 3,
                    "chunk_id": "b-3",
                    "document_date": "2026-02-01",
                    "text": "HajiMind 使用 LangChain。",
                },
            ]
        )

        self.assertTrue(result["has_conflict"])
        self.assertEqual(result["sources"], ["architecture-a.pdf", "architecture-b.pdf"])
        self.assertEqual({item["timestamp"] for item in result["claims"]}, {"2026-01-01", "2026-02-01"})
        self.assertTrue(all(item["evidence"] for item in result["claims"]))

    def test_conflicting_claims_in_one_source_are_not_reported_as_multi_document_conflict(self):
        result = detect_document_conflicts(
            [
                {
                    "filename": "draft.pdf",
                    "text": "HajiMind 使用 LangGraph。HajiMind 使用 LangChain。",
                }
            ]
        )

        self.assertFalse(result["has_conflict"])


if __name__ == "__main__":
    unittest.main()
