from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

SENSITIVE_KEYS = {"api_key", "authorization", "token", "password", "secret", "key"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ReliabilityConfig:
    max_tool_retries: int = 2
    max_task_retries: int = 2
    max_agent_steps: int = 24
    max_recovery_attempts: int = 3
    retry_backoff_seconds: float = 0.25
    retry_backoff_max_seconds: float = 2.0
    stuck_repeat_threshold: int = 3
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_reset_seconds: float = 30.0


def get_reliability_config() -> ReliabilityConfig:
    return ReliabilityConfig(
        max_tool_retries=_env_int("MAX_TOOL_RETRIES", 2),
        max_task_retries=_env_int("MAX_TASK_RETRIES", 2),
        max_agent_steps=_env_int("MAX_AGENT_STEPS", 24, minimum=1),
        max_recovery_attempts=_env_int("MAX_RECOVERY_ATTEMPTS", 3),
        retry_backoff_seconds=_env_float("RETRY_BACKOFF_SECONDS", 0.25),
        retry_backoff_max_seconds=_env_float("RETRY_BACKOFF_MAX_SECONDS", 2.0),
        stuck_repeat_threshold=_env_int("STUCK_REPEAT_THRESHOLD", 3, minimum=2),
        circuit_breaker_enabled=_env_bool("CIRCUIT_BREAKER_ENABLED", True),
        circuit_breaker_failure_threshold=_env_int(
            "CIRCUIT_BREAKER_FAILURE_THRESHOLD", 3, minimum=1
        ),
        circuit_breaker_reset_seconds=_env_float(
            "CIRCUIT_BREAKER_RESET_SECONDS", 30.0, minimum=0.01
        ),
    )


def retry_delay_seconds(base_seconds: float, attempt_index: int) -> float:
    config = get_reliability_config()
    base = base_seconds if base_seconds > 0 else config.retry_backoff_seconds
    return min(base * (2 ** max(0, attempt_index - 1)), config.retry_backoff_max_seconds)


@dataclass
class CircuitState:
    failure_count: int = 0
    opened_at: float | None = None
    last_error: str | None = None


class CircuitBreaker:
    def __init__(self):
        self._states: dict[str, CircuitState] = {}

    def allow(self, name: str, config: ReliabilityConfig | None = None) -> tuple[bool, str | None]:
        config = config or get_reliability_config()
        if not config.circuit_breaker_enabled:
            return True, None
        state = self._states.get(name)
        if not state or state.opened_at is None:
            return True, None
        elapsed = time.monotonic() - state.opened_at
        if elapsed >= config.circuit_breaker_reset_seconds:
            return True, None
        return False, (
            f"Tool circuit is open for {name}; retry after "
            f"{config.circuit_breaker_reset_seconds - elapsed:.1f}s"
        )

    def record_success(self, name: str) -> None:
        self._states.pop(name, None)

    def record_failure(
        self,
        name: str,
        error: str | None,
        config: ReliabilityConfig | None = None,
    ) -> None:
        config = config or get_reliability_config()
        if not config.circuit_breaker_enabled:
            return
        state = self._states.setdefault(name, CircuitState())
        state.failure_count += 1
        state.last_error = error
        if state.failure_count >= config.circuit_breaker_failure_threshold:
            state.opened_at = time.monotonic()

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self._states.clear()
        else:
            self._states.pop(name, None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            name: {
                "failure_count": state.failure_count,
                "opened": state.opened_at is not None,
                "last_error": state.last_error,
            }
            for name, state in self._states.items()
        }


circuit_breaker = CircuitBreaker()


@dataclass(frozen=True)
class StuckSignal:
    kind: str
    key: str
    count: int
    reason: str


@dataclass(frozen=True)
class RecoveryAction:
    strategy: str
    action: str
    reason: str
    updated_input: dict[str, Any] | None = None
    fallback_tool: str | None = None
    skip_task: bool = False
    user_message: str | None = None


def safe_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else safe_payload(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [safe_payload(item) for item in payload]
    return payload


def payload_signature(payload: Any) -> str:
    try:
        return json.dumps(safe_payload(payload), ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(payload)


def _last_matching(events: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") == event_name]


def _is_stuck(
    execution_trace: list[dict[str, Any]],
    *,
    threshold: int | None = None,
) -> StuckSignal | None:
    config = get_reliability_config()
    threshold = threshold or config.stuck_repeat_threshold
    if len(execution_trace) < threshold:
        return None

    task_starts = _last_matching(execution_trace, "task_started")
    if len(task_starts) >= threshold:
        recent = task_starts[-threshold:]
        keys = [
            event.get("task_signature") or f"{event.get('tool')}:{event.get('task_id')}"
            for event in recent
        ]
        if len(set(keys)) == 1:
            return StuckSignal(
                kind="repeated_task",
                key=str(keys[0]),
                count=threshold,
                reason="same task executed repeatedly",
            )

    tool_starts = _last_matching(execution_trace, "tool_start")
    if len(tool_starts) >= threshold:
        recent = tool_starts[-threshold:]
        call_keys = [
            f"{event.get('tool_name')}:{event.get('input_signature')}"
            for event in recent
        ]
        if len(set(call_keys)) == 1:
            return StuckSignal(
                kind="repeated_tool_call",
                key=str(call_keys[0]),
                count=threshold,
                reason="same tool call repeated",
            )
        query_keys = [
            event.get("query")
            for event in recent
            if event.get("query") is not None
        ]
        if len(query_keys) == threshold and len(set(query_keys)) == 1:
            return StuckSignal(
                kind="repeated_query",
                key=str(query_keys[0]),
                count=threshold,
                reason="same query repeated",
            )

    tool_results = [
        event
        for event in _last_matching(execution_trace, "tool_result")
        if event.get("status") not in ("success", None)
    ]
    if len(tool_results) >= threshold:
        recent = tool_results[-threshold:]
        tools = [event.get("tool_name") for event in recent]
        if len(set(tools)) == 1:
            return StuckSignal(
                kind="consecutive_tool_failure",
                key=str(tools[0]),
                count=threshold,
                reason="tool failed repeatedly",
            )

    planner_ticks = _last_matching(execution_trace, "planner_iteration")
    if len(planner_ticks) >= threshold:
        recent = planner_ticks[-threshold:]
        keys = [
            (
                event.get("pending_count"),
                event.get("completed_count"),
                event.get("failed_count"),
            )
            for event in recent
        ]
        if len(set(keys)) == 1:
            return StuckSignal(
                kind="planner_no_progress",
                key=str(keys[0]),
                count=threshold,
                reason="planner produced no new result",
            )

    return None


def attempt_loop_recovery(
    *,
    task_tool: str,
    payload: dict[str, Any],
    error: str | None = None,
    recovery_attempt: int = 0,
    signal: StuckSignal | None = None,
    config: ReliabilityConfig | None = None,
) -> RecoveryAction:
    config = config or get_reliability_config()
    if recovery_attempt >= config.max_recovery_attempts:
        return RecoveryAction(
            strategy="tell_user_unavailable",
            action="complete_with_message",
            reason="recovery attempts exhausted",
            user_message="无法获得稳定的实时结果，请稍后重试或改用知识库资料。",
        )

    updated = dict(payload)
    query = str(updated.get("query", "")).strip()
    if recovery_attempt == 0 and query:
        updated["query"] = f"{query} 最新 可靠 摘要"
        return RecoveryAction(
            strategy="modify_query",
            action="retry",
            reason=(signal.reason if signal else error) or "retry with adjusted query",
            updated_input=updated,
        )

    top_k = updated.get("top_k")
    if isinstance(top_k, int) and top_k > 1:
        updated["top_k"] = max(1, top_k // 2)
        return RecoveryAction(
            strategy="lower_top_k",
            action="retry",
            reason=(signal.reason if signal else error) or "retry with smaller result set",
            updated_input=updated,
        )

    if task_tool in {"baidu_search", "web_search"}:
        fallback_input = {"query": query or str(payload)}
        return RecoveryAction(
            strategy="switch_to_rag",
            action="fallback_tool",
            reason=(signal.reason if signal else error) or "web search unavailable",
            updated_input=fallback_input,
            fallback_tool="knowledge_base",
        )

    return RecoveryAction(
        strategy="skip_current_task",
        action="skip",
        reason=(signal.reason if signal else error) or "task cannot be recovered",
        skip_task=True,
    )
