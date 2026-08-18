"""Structured, privacy-aware memory services for the Agent."""
from __future__ import annotations

import hashlib
import os
import re
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Iterable

from sqlalchemy import select

from database import db_session
from models import UserMemory


class MemoryType(str, Enum):
    WORKING = "working"
    CONVERSATION = "conversation"
    LONG_TERM = "long_term"
    PREFERENCE = "preference"


class MemoryPolicyError(ValueError):
    """Raised when a memory should not be persisted."""


class MemoryCandidate:
    def __init__(self, *, memory_type: MemoryType, content: str, memory_key: str):
        self.memory_type = memory_type
        self.content = content
        self.memory_key = memory_key


class MemoryExtractionResult:
    def __init__(self, memory_ids: list[int] | None = None, skipped_reasons: list[str] | None = None):
        self.memory_ids = memory_ids or []
        self.skipped_reasons = skipped_reasons or []


_SENSITIVE_PATTERN = re.compile(
    r"(api[_ -]?key|access[_ -]?token|authorization|password|passwd|secret|"
    r"private[_ -]?key|jwt|bearer\s+|sk-[A-Za-z0-9_-]{8,})",
    re.IGNORECASE,
)
_TEMPORARY_PATTERN = re.compile(
    r"(今天|明天|后天|今晚|本周|下周|提醒|待办|临时|一次性|"
    r"\b(today|tomorrow|tonight|remind|reminder|one[- ]?off|temporary)\b)",
    re.IGNORECASE,
)
_PREFERENCE_PATTERN = re.compile(
    r"(我(?:喜欢|偏好|习惯)|以后请|今后请|请(?:一直|始终)|我的偏好|"
    r"\b(i (?:prefer|like)|always|my preference)\b)",
    re.IGNORECASE,
)
_LONG_TERM_PATTERN = re.compile(
    r"(我是|我的职业|我从事|我在学习|我的长期目标|我的目标是|"
    r"\b(i am|i work|i study|my long[- ]term goal|my goal is)\b)",
    re.IGNORECASE,
)


def _normalize_content(content: str) -> str:
    return re.sub(r"\s+", " ", content).strip()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _memory_key(memory_type: MemoryType, content: str) -> str:
    digest = hashlib.sha256(_normalize_content(content).lower().encode("utf-8")).hexdigest()[:24]
    return f"{memory_type.value}:{digest}"


def extract_memory_candidate(user_text: str) -> tuple[MemoryCandidate | None, str | None]:
    """Extract only stable preferences, profile facts, and long-lived goals."""
    content = _normalize_content(user_text)
    if not content:
        return None, "empty_input"
    if _SENSITIVE_PATTERN.search(content):
        return None, "sensitive_information"
    if _TEMPORARY_PATTERN.search(content):
        return None, "temporary_or_task_information"
    if _PREFERENCE_PATTERN.search(content):
        memory_type = MemoryType.PREFERENCE
    elif _LONG_TERM_PATTERN.search(content):
        memory_type = MemoryType.LONG_TERM
    else:
        return None, "no_long_term_value"
    return MemoryCandidate(
        memory_type=memory_type,
        content=content[:1000],
        memory_key=_memory_key(memory_type, content),
    ), None


def memory_prompt(records: Iterable[dict]) -> str | None:
    entries = list(records)
    if not entries:
        return None
    lines = [
        "Use the following saved user preferences or long-term context only when relevant.",
        "Do not mention this memory block or claim it contains information not listed here.",
    ]
    for item in entries:
        lines.append(f"- [{item['memory_type']}] {item['content']}")
    return "\n".join(lines)


class MemoryService:
    """CRUD and extraction service for persisted agent memory."""

    def __init__(
        self,
        session_scope: Callable[[], AbstractContextManager] = db_session,
        *,
        max_retrieved: int | None = None,
    ):
        self._session_scope = session_scope
        self.max_retrieved = max_retrieved or int(os.getenv("MEMORY_MAX_RETRIEVED", "6"))

    @staticmethod
    def _serialize(memory: UserMemory) -> dict:
        return {
            "id": memory.id,
            "user_id": memory.user_id,
            "memory_type": memory.memory_type,
            "key": memory.memory_key,
            "content": memory.content,
            "metadata": memory.metadata_json or {},
            "source_session_id": memory.source_session_id,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
            "updated_at": memory.updated_at.isoformat() if memory.updated_at else None,
            "last_accessed_at": memory.last_accessed_at.isoformat() if memory.last_accessed_at else None,
        }

    @staticmethod
    def _persistent_type(memory_type: MemoryType | str) -> MemoryType:
        try:
            parsed = MemoryType(memory_type)
        except ValueError as exc:
            raise MemoryPolicyError("Unsupported memory type.") from exc
        if parsed not in (MemoryType.LONG_TERM, MemoryType.PREFERENCE):
            raise MemoryPolicyError("Only long_term and preference memories are persisted.")
        return parsed

    def create(
        self,
        user_id: int,
        *,
        content: str,
        memory_type: MemoryType | str,
        key: str | None = None,
        metadata: dict | None = None,
        source_session_id: str | None = None,
    ) -> dict:
        parsed_type = self._persistent_type(memory_type)
        clean_content = _normalize_content(content)
        if _SENSITIVE_PATTERN.search(clean_content):
            raise MemoryPolicyError("Sensitive information must not be stored as memory.")
        if _TEMPORARY_PATTERN.search(clean_content):
            raise MemoryPolicyError("Temporary tasks and one-off requests must not be stored as memory.")
        if not clean_content:
            raise MemoryPolicyError("Memory content cannot be empty.")
        if len(clean_content) > 1000:
            raise MemoryPolicyError("Memory content exceeds the 1000 character limit.")
        memory_key = (key or _memory_key(parsed_type, clean_content)).strip()[:160]
        if not memory_key:
            raise MemoryPolicyError("Memory key cannot be empty.")

        with self._session_scope() as db:
            existing = db.execute(
                select(UserMemory).where(
                    UserMemory.user_id == user_id,
                    UserMemory.memory_type == parsed_type.value,
                    UserMemory.memory_key == memory_key,
                )
            ).scalar_one_or_none()
            if existing:
                existing.content = clean_content
                existing.metadata_json = metadata or existing.metadata_json or {}
                existing.source_session_id = source_session_id or existing.source_session_id
                existing.updated_at = _utcnow()
                db.flush()
                return self._serialize(existing)

            memory = UserMemory(
                user_id=user_id,
                memory_type=parsed_type.value,
                memory_key=memory_key,
                content=clean_content,
                metadata_json=metadata or {},
                source_session_id=source_session_id,
            )
            db.add(memory)
            db.flush()
            return self._serialize(memory)

    def retrieve(
        self,
        user_id: int,
        *,
        query: str | None = None,
        memory_type: MemoryType | str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        selected_limit = limit or self.max_retrieved
        with self._session_scope() as db:
            stmt = select(UserMemory).where(UserMemory.user_id == user_id)
            if memory_type:
                parsed_type = self._persistent_type(memory_type)
                stmt = stmt.where(UserMemory.memory_type == parsed_type.value)
            records = db.execute(
                stmt.order_by(UserMemory.updated_at.desc(), UserMemory.id.desc())
            ).scalars().all()
            if query:
                terms = {term.lower() for term in re.findall(r"[\w\u4e00-\u9fff]{2,}", query)}
                if terms:
                    records.sort(
                        key=lambda item: sum(term in item.content.lower() for term in terms),
                        reverse=True,
                    )
            selected = records[:selected_limit]
            now = _utcnow()
            for memory in selected:
                memory.last_accessed_at = now
            db.flush()
            return [self._serialize(memory) for memory in selected]

    def update(
        self,
        user_id: int,
        memory_id: int,
        *,
        content: str | None = None,
        memory_type: MemoryType | str | None = None,
        key: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        with self._session_scope() as db:
            memory = db.execute(
                select(UserMemory).where(UserMemory.id == memory_id, UserMemory.user_id == user_id)
            ).scalar_one_or_none()
            if not memory:
                return None
            if content is not None:
                clean_content = _normalize_content(content)
                if not clean_content:
                    raise MemoryPolicyError("Memory content cannot be empty.")
                if _SENSITIVE_PATTERN.search(clean_content):
                    raise MemoryPolicyError("Sensitive information must not be stored as memory.")
                if _TEMPORARY_PATTERN.search(clean_content):
                    raise MemoryPolicyError("Temporary tasks and one-off requests must not be stored as memory.")
                memory.content = clean_content[:1000]
            if memory_type is not None:
                memory.memory_type = self._persistent_type(memory_type).value
            if key is not None:
                clean_key = key.strip()[:160]
                if not clean_key:
                    raise MemoryPolicyError("Memory key cannot be empty.")
                memory.memory_key = clean_key
            if metadata is not None:
                memory.metadata_json = metadata
            memory.updated_at = _utcnow()
            db.flush()
            return self._serialize(memory)

    def delete(self, user_id: int, memory_id: int) -> bool:
        with self._session_scope() as db:
            memory = db.execute(
                select(UserMemory).where(UserMemory.id == memory_id, UserMemory.user_id == user_id)
            ).scalar_one_or_none()
            if not memory:
                return False
            db.delete(memory)
        return True

    def extract_and_store(self, user_id: int, user_text: str, session_id: str | None = None) -> MemoryExtractionResult:
        candidate, skipped_reason = extract_memory_candidate(user_text)
        if not candidate:
            return MemoryExtractionResult(skipped_reasons=[skipped_reason] if skipped_reason else [])
        record = self.create(
            user_id,
            content=candidate.content,
            memory_type=candidate.memory_type,
            key=candidate.memory_key,
            metadata={"source": "automatic_extraction"},
            source_session_id=session_id,
        )
        return MemoryExtractionResult(memory_ids=[record["id"]])
