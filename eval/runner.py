from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from rag_utils import retrieve_documents  # type: ignore

from eval.dataset import EvalExample, load_eval_dataset
from eval.metrics import hit_rate, mean_reciprocal_rank, ndcg_at_k, precision_at_k, recall_at_k
from eval.reporter import aggregate_rows, write_csv_report, write_json_report


Retriever = Callable[[str, int], dict[str, Any]]


def default_retriever_factory(
    *,
    retrieval_mode: str,
    candidate_k: int,
    rerank_enabled: bool,
    dense_weight: float,
    sparse_weight: float,
    bm25_k1: float | None,
    bm25_b: float | None,
) -> Retriever:
    def _retrieve(question: str, top_k: int) -> dict[str, Any]:
        return retrieve_documents(
            question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            candidate_k=candidate_k,
            rerank_enabled=rerank_enabled,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b,
        )

    return _retrieve


def evaluate_dataset(
    dataset: list[EvalExample],
    *,
    retriever: Retriever,
    top_k: int,
    retrieval_mode: str,
    candidate_k: int,
    rerank_enabled: bool,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    bm25_k1: float | None = None,
    bm25_b: float | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for example in dataset:
        result = retriever(example.question, top_k)
        docs = result.get("docs", []) if isinstance(result, dict) else []
        predicted_chunk_ids = [str(doc.get("chunk_id", "")) for doc in docs if doc.get("chunk_id")]
        row = {
            "question": example.question,
            "dataset_type": example.dataset_type,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "rerank_enabled": rerank_enabled,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
            "bm25_k1": bm25_k1,
            "bm25_b": bm25_b,
            "expected_answer": example.expected_answer,
            "gold_chunk_ids": example.gold_chunk_ids,
            "predicted_chunk_ids": predicted_chunk_ids,
            "recall_at_k": recall_at_k(predicted_chunk_ids, example.gold_chunk_ids, top_k),
            "precision_at_k": precision_at_k(predicted_chunk_ids, example.gold_chunk_ids, top_k),
            "mrr": mean_reciprocal_rank(predicted_chunk_ids, example.gold_chunk_ids),
            "ndcg_at_k": ndcg_at_k(predicted_chunk_ids, example.gold_chunk_ids, top_k),
            "hit_rate": hit_rate(predicted_chunk_ids, example.gold_chunk_ids, top_k),
        }
        rows.append(row)

    aggregates = aggregate_rows(
        rows,
        [
            "dataset_type",
            "retrieval_mode",
            "top_k",
            "candidate_k",
            "rerank_enabled",
            "dense_weight",
            "sparse_weight",
            "bm25_k1",
            "bm25_b",
        ],
    )
    return {
        "dataset_size": len(dataset),
        "rows": rows,
        "summary": aggregates,
    }


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RAG retrieval and rerank evaluation.")
    parser.add_argument("--dataset", default="eval/datasets", help="Dataset file or directory.")
    parser.add_argument("--output-json", default="eval/reports/rag_eval_report.json")
    parser.add_argument("--output-csv", default="eval/reports/rag_eval_report.csv")
    parser.add_argument("--top-k", default="5", help="Comma-separated top_k values.")
    parser.add_argument("--candidate-k", default="", help="Comma-separated candidate_k values.")
    parser.add_argument(
        "--modes",
        default="dense,bm25,hybrid,hybrid_weighted,hybrid_rerank",
        help="Comma-separated retrieval modes.",
    )
    parser.add_argument("--bm25-k1", default="", help="Optional BM25 k1 override.")
    parser.add_argument("--bm25-b", default="", help="Optional BM25 b override.")
    parser.add_argument("--dense-weight", default="0.5")
    parser.add_argument("--sparse-weight", default="0.5")
    parser.add_argument("--bm25-k1-values", default="", help="Comma-separated BM25 k1 values for a sweep.")
    parser.add_argument("--bm25-b-values", default="", help="Comma-separated BM25 b values for a sweep.")
    parser.add_argument("--dense-weights", default="", help="Comma-separated dense weights for AB testing.")
    parser.add_argument("--sparse-weights", default="", help="Comma-separated sparse weights for AB testing.")
    parser.add_argument("--no-rerank", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    dataset = load_eval_dataset(args.dataset)
    if not dataset:
        raise SystemExit(f"No evaluation examples found in {args.dataset}")

    top_ks = _parse_int_list(args.top_k)
    candidate_ks = _parse_int_list(args.candidate_k) if args.candidate_k else [max(top_k * 3, top_k) for top_k in top_ks]
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    bm25_k1_values = _parse_float_list(args.bm25_k1_values) if args.bm25_k1_values else [_parse_float(args.bm25_k1)]
    bm25_b_values = _parse_float_list(args.bm25_b_values) if args.bm25_b_values else [_parse_float(args.bm25_b)]
    dense_weights = _parse_float_list(args.dense_weights) if args.dense_weights else [float(args.dense_weight)]
    sparse_weights = _parse_float_list(args.sparse_weights) if args.sparse_weights else [float(args.sparse_weight)]

    rows: list[dict[str, Any]] = []
    for retrieval_mode in modes:
        for top_k in top_ks:
            for candidate_k in candidate_ks:
                for dense_weight in dense_weights:
                    for sparse_weight in sparse_weights:
                        for bm25_k1 in bm25_k1_values:
                            for bm25_b in bm25_b_values:
                                rerank_enabled = retrieval_mode.endswith("rerank") and not args.no_rerank
                                mode_name = retrieval_mode.replace("_rerank", "")
                                retriever = default_retriever_factory(
                                    retrieval_mode=mode_name,
                                    candidate_k=candidate_k,
                                    rerank_enabled=rerank_enabled,
                                    dense_weight=dense_weight,
                                    sparse_weight=sparse_weight,
                                    bm25_k1=bm25_k1,
                                    bm25_b=bm25_b,
                                )
                                evaluation = evaluate_dataset(
                                    dataset,
                                    retriever=retriever,
                                    top_k=top_k,
                                    retrieval_mode=mode_name,
                                    candidate_k=candidate_k,
                                    rerank_enabled=rerank_enabled,
                                    dense_weight=dense_weight,
                                    sparse_weight=sparse_weight,
                                    bm25_k1=bm25_k1,
                                    bm25_b=bm25_b,
                                )
                                rows.extend(evaluation["rows"])

    report = {
        "dataset_path": str(args.dataset),
        "dataset_size": len(dataset),
        "example_count": len(rows),
        "summary": aggregate_rows(
            rows,
            [
                "dataset_type",
                "retrieval_mode",
                "top_k",
                "candidate_k",
                "rerank_enabled",
                "dense_weight",
                "sparse_weight",
                "bm25_k1",
                "bm25_b",
            ],
        ),
        "config": {
            "top_k": top_ks,
            "candidate_k": candidate_ks,
            "modes": modes,
            "bm25_k1": bm25_k1_values,
            "bm25_b": bm25_b_values,
            "dense_weight": dense_weights,
            "sparse_weight": sparse_weights,
            "rerank_enabled": not args.no_rerank,
        },
        "rows": rows,
    }

    write_json_report(args.output_json, report)
    write_csv_report(args.output_csv, rows)
    print(json.dumps({"json": args.output_json, "csv": args.output_csv, "examples": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
