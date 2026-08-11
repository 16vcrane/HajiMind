from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


MAX_SUMMARY_LENGTH = 240
SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "token",
    "password",
    "secret",
    "key",
}


class ToolCallTrace(BaseModel):
    tool_name: str
    start_time: str | None = None
    end_time: str | None = None
    latency_ms: int | None = None
    status: str
    input_summary: str | None = None
    output_summary: str | None = None
    error: str | None = None
    attempts: int = 1
    result_count: int | None = None


class TaskExecutionTrace(BaseModel):
    task_id: str
    description: str
    tool: str
    start_time: str | None = None
    end_time: str | None = None
    latency_ms: int | None = None
    status: str
    input_summary: str | None = None
    output_summary: str | None = None
    error: str | None = None
    attempts: int = 0


class AgentTrace(BaseModel):
    request_id: str
    session_id: str
    user_id: int
    intent: str | None = None
    complexity: str | None = None
    route: str | None = None
    plan: dict[str, Any] | None = None
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    rag_trace: dict[str, Any] | None = None
    task_results: dict[str, Any] = Field(default_factory=dict)
    task_traces: list[TaskExecutionTrace] = Field(default_factory=list)
    recovery_attempts: int = 0
    recovery_events: list[dict[str, Any]] = Field(default_factory=list)
    latency: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    final_status: str = "running"


def new_request_id() -> str:
    return str(uuid.uuid4())


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def sanitize_value(value: Any, max_length: int = MAX_SUMMARY_LENGTH) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                clean[key] = "[redacted]"
            else:
                clean[key] = summarize_value(item, max_length=80)
        text = str(clean)
    elif isinstance(value, list):
        text = f"list(len={len(value)})"
    else:
        text = str(value)
    return text[:max_length] + ("..." if len(text) > max_length else "")


def summarize_value(value: Any, max_length: int = MAX_SUMMARY_LENGTH) -> str:
    if isinstance(value, dict):
        if isinstance(value.get("content"), str):
            return sanitize_value(value["content"], max_length=max_length)
        if isinstance(value.get("results"), list):
            return f"results={len(value['results'])}"
        return sanitize_value(value, max_length=max_length)
    if isinstance(value, list):
        return f"items={len(value)}"
    return sanitize_value(value, max_length=max_length)


def result_count(value: Any) -> int | None:
    if isinstance(value, dict):
        if isinstance(value.get("results"), list):
            return len(value["results"])
        if isinstance(value.get("retrieved_chunks"), list):
            return len(value["retrieved_chunks"])
        if isinstance(value.get("initial_retrieved_chunks"), list):
            return len(value["initial_retrieved_chunks"])
    if isinstance(value, list):
        return len(value)
    return None


def tool_call_from_result(tool_result, input_payload: dict[str, Any] | None = None) -> ToolCallTrace:
    trace = tool_result.trace
    return ToolCallTrace(
        tool_name=tool_result.tool_name,
        latency_ms=trace.latency_ms,
        status=str(tool_result.status.value if hasattr(tool_result.status, "value") else tool_result.status),
        input_summary=sanitize_value(input_payload or {}),
        output_summary=summarize_value(tool_result.data),
        error=tool_result.error,
        attempts=trace.attempts,
        result_count=result_count(tool_result.data),
    )


def task_trace_from_task(task, *, started_at: float | None = None) -> TaskExecutionTrace:
    latency_ms = monotonic_ms(started_at) if started_at is not None else None
    return TaskExecutionTrace(
        task_id=task.id,
        description=task.description,
        tool=str(task.tool),
        status=str(task.status.value if hasattr(task.status, "value") else task.status),
        input_summary=sanitize_value(task.input),
        output_summary=summarize_value(task.result),
        error=task.error,
        attempts=task.attempts,
        latency_ms=latency_ms,
    )
