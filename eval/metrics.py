from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


def _as_set(values: Iterable[str] | None) -> set[str]:
    return {value for value in (values or []) if value}


def recall_at_k(predicted: list[str], gold: list[str], k: int) -> float:
    gold_set = _as_set(gold)
    if not gold_set:
        return 0.0
    hits = len(_as_set(predicted[:k]) & gold_set)
    return hits / len(gold_set)


def precision_at_k(predicted: list[str], gold: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    gold_set = _as_set(gold)
    hits = len(_as_set(predicted[:k]) & gold_set)
    return hits / k


def hit_rate(predicted: list[str], gold: list[str], k: int) -> float:
    return 1.0 if _as_set(predicted[:k]) & _as_set(gold) else 0.0


def mean_reciprocal_rank(predicted: list[str], gold: list[str]) -> float:
    gold_set = _as_set(gold)
    if not gold_set:
        return 0.0
    for index, item in enumerate(predicted, start=1):
        if item in gold_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(predicted: list[str], gold: list[str], k: int) -> float:
    gold_set = _as_set(gold)
    if not gold_set:
        return 0.0
    dcg = 0.0
    for index, item in enumerate(predicted[:k], start=1):
        if item in gold_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@dataclass
class MetricBundle:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float
