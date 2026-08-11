import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, ".")

from eval.dataset import load_eval_dataset
from eval.runner import evaluate_dataset


class RagEvaluationTests(unittest.TestCase):
    def test_dataset_loader_distinguishes_gold_and_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            path.write_text(
                '\n'.join(
                    [
                        '{"question":"Q1","gold_chunk_ids":["a"],"expected_answer":"A","dataset_type":"gold"}',
                        '{"question":"Q2","gold_chunk_ids":["b"],"expected_answer":"B","dataset_type":"synthetic"}',
                    ]
                ),
                encoding="utf-8",
            )

            dataset = load_eval_dataset(path)

        self.assertEqual(dataset[0].dataset_type, "gold")
        self.assertEqual(dataset[1].dataset_type, "synthetic")

    def test_evaluator_computes_metrics(self):
        dataset = load_eval_dataset(Path("eval/datasets/sample_gold.jsonl"))

        def retriever(question, top_k):
            if "RAG" in question:
                return {"docs": [{"chunk_id": "sample::p1::l3::0"}]}
            return {"docs": [{"chunk_id": "other"}]}

        report = evaluate_dataset(
            dataset,
            retriever=retriever,
            top_k=1,
            retrieval_mode="dense",
            candidate_k=3,
            rerank_enabled=False,
        )

        self.assertEqual(report["dataset_size"], 2)
        self.assertEqual(len(report["rows"]), 2)
        self.assertIn("summary", report)
        self.assertGreaterEqual(report["rows"][0]["recall_at_k"], 0.0)


if __name__ == "__main__":
    unittest.main()
