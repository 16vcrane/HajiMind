from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def write_json_report(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv_report(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]], metric_keys: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(metric) for metric in metric_keys)
        grouped[key].append(row)

    aggregates = []
    for key, group_rows in grouped.items():
        item = {metric_keys[index]: key[index] for index in range(len(metric_keys))}
        item["count"] = len(group_rows)
        for metric in ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "hit_rate"):
            item[metric] = round(sum(float(row.get(metric, 0.0)) for row in group_rows) / len(group_rows), 6)
        aggregates.append(item)
    return {"groups": aggregates}
