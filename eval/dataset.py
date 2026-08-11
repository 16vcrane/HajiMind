from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalExample:
    question: str
    gold_chunk_ids: list[str] = field(default_factory=list)
    expected_answer: str = ""
    dataset_type: str = "gold"
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def load_eval_dataset(path: str | Path) -> list[EvalExample]:
    path = Path(path)
    if path.is_dir():
        items: list[EvalExample] = []
        for file_path in sorted(path.glob("*")):
            items.extend(load_eval_dataset(file_path))
        return items
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        items = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(_row_to_example(json.loads(line)))
        return items
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [_row_to_example(row) for row in data]
        if isinstance(data, dict):
            rows = data.get("examples") or data.get("data") or []
            return [_row_to_example(row) for row in rows]
        return []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [_row_to_example(row) for row in reader]
    raise ValueError(f"Unsupported dataset format: {path}")


def _row_to_example(row: dict[str, Any]) -> EvalExample:
    return EvalExample(
        question=str(row.get("question", "")).strip(),
        gold_chunk_ids=_normalize_ids(row.get("gold_chunk_ids") or row.get("gold_chunk_id")),
        expected_answer=str(row.get("expected_answer", "")).strip(),
        dataset_type=str(row.get("dataset_type", "gold")).strip() or "gold",
        metadata={
            key: value
            for key, value in row.items()
            if key not in {"question", "gold_chunk_ids", "gold_chunk_id", "expected_answer", "dataset_type"}
        },
    )
