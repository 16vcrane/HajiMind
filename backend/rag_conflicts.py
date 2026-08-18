"""Explainable conflict detection for retrieved knowledge-base documents."""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field


class ConflictClaim(BaseModel):
    claim: str
    source: str
    page_number: str | int | None = None
    chunk_id: str | None = None
    evidence: str
    timestamp: str | None = None


class ConflictResult(BaseModel):
    has_conflict: bool = False
    claims: list[ConflictClaim] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)


_CLAIM_PATTERNS = (
    re.compile(
        r"(?P<subject>[^。；;!?\n]{1,80}?)\s*"
        r"(?P<predicate>使用|采用|支持|是|为)\s*"
        r"(?P<value>[^。；;!?\n]{1,160})"
    ),
    re.compile(
        r"(?P<subject>[A-Za-z][A-Za-z0-9 _./-]{0,80}?)\s+"
        r"(?P<predicate>is|are|uses?|has|supports?)\s+"
        r"(?P<value>[^.?!;\n]{1,160})",
        re.IGNORECASE,
    ),
)
_SOURCE_TIME_KEYS = ("document_date", "published_at", "updated_at", "created_at", "timestamp", "date")


def _normalize(value: str) -> str:
    return re.sub(r"[\s\"'`，,。.;:：()（）\[\]{}]+", "", value).lower()


def _document_timestamp(doc: dict[str, Any]) -> str | None:
    for key in _SOURCE_TIME_KEYS:
        value = doc.get(key)
        if value:
            return str(value)
    metadata = doc.get("metadata")
    if isinstance(metadata, dict):
        for key in _SOURCE_TIME_KEYS:
            value = metadata.get(key)
            if value:
                return str(value)
    return None


def _extract_claims(doc: dict[str, Any]) -> list[tuple[str, str, ConflictClaim]]:
    source = str(doc.get("filename") or doc.get("source") or "Unknown")
    text = str(doc.get("text") or "")
    page_number = doc.get("page_number")
    chunk_id = doc.get("chunk_id")
    timestamp = _document_timestamp(doc)
    extracted: list[tuple[str, str, ConflictClaim]] = []

    for sentence in re.split(r"(?<=[。.!?！？])\s*", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        for pattern in _CLAIM_PATTERNS:
            match = pattern.search(sentence)
            if not match:
                continue
            subject = match.group("subject").strip()
            predicate = match.group("predicate").strip()
            value = match.group("value").strip()
            if not subject or not value:
                continue
            claim = ConflictClaim(
                claim=f"{subject} {predicate} {value}",
                source=source,
                page_number=page_number,
                chunk_id=str(chunk_id) if chunk_id else None,
                evidence=sentence[:240],
                timestamp=timestamp,
            )
            extracted.append((f"{_normalize(subject)}::{_normalize(predicate)}", _normalize(value), claim))
            break
    return extracted


def detect_document_conflicts(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Find incompatible factual claims from different retrieved sources.

    Detection is deliberately conservative: a conflict is reported only when
    equivalent subject/predicate pairs have distinct normalized values across
    separate source documents. It never ranks or resolves the claims.
    """
    grouped: dict[str, dict[str, list[ConflictClaim]]] = defaultdict(lambda: defaultdict(list))
    for doc in docs:
        for claim_key, value_key, claim in _extract_claims(doc):
            grouped[claim_key][value_key].append(claim)

    conflicts: list[ConflictClaim] = []
    sources: set[str] = set()
    for values in grouped.values():
        source_sets = {
            value_key: {claim.source for claim in claims}
            for value_key, claims in values.items()
        }
        distinct_sources = set().union(*source_sets.values()) if source_sets else set()
        if len(values) < 2 or len(distinct_sources) < 2:
            continue
        for claims in values.values():
            for claim in claims:
                conflicts.append(claim)
                sources.add(claim.source)

    return ConflictResult(
        has_conflict=bool(conflicts),
        claims=conflicts,
        sources=sorted(sources),
    ).model_dump(mode="json")
