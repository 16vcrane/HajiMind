import asyncio
import os
import sys
import time
import unittest
from unittest.mock import patch

from pydantic import BaseModel

sys.path.insert(0, "backend")

from agent_planner import Plan, TaskExecutor
from agent_reliability import (
    CircuitBreaker,
    _is_stuck,
    attempt_loop_recovery,
    get_reliability_config,
)
from tools import BaseTool, ToolExecutionError, ToolRegistry, ToolStatus


class ReliabilityInput(BaseModel):
    query: str = "test"
    top_k: int = 5


class AlwaysFailTool(BaseTool):
    name = "always_fail"
    description = "Always fails with a retryable provider error."
    input_schema = ReliabilityInput
    max_retries = 0

    def __init__(self, error="HTTP 500", *, retryable=True):
        self.calls = 0
        self.error = error
        self.retryable = retryable

    def _invoke(self, _input_data):
        self.calls += 1
        raise ToolExecutionError(self.error, retryable=self.retryable)


class SlowReliabilityTool(BaseTool):
    name = "slow_reliability"
    description = "Sleeps beyond the configured timeout."
    input_schema = ReliabilityInput
    timeout_seconds = 0.01
    max_retries = 0

    def _invoke(self, _input_data):
        time.sleep(0.05)
        return "late"


class SlowBaiduTool(SlowReliabilityTool):
    name = "baidu_search"


class RecordingReliabilityTool(BaseTool):
    name = "baidu_search"
    description = "Records calls for planner recovery tests."
    input_schema = ReliabilityInput
    max_retries = 0

    def __init__(self, calls):
        self.calls = calls

    def _invoke(self, input_data):
        self.calls.append(input_data.model_dump())
        if len(self.calls) == 1:
            raise ToolExecutionError("HTTP 500", retryable=False)
        return {"content": "recovered"}


class FakeModel:
    def invoke(self, _messages):
        return type("Response", (), {"content": "answer"})()


class ReliabilityTests(unittest.TestCase):
    def test_is_stuck_detects_repeated_tool_call(self):
        trace = [
            {
                "event": "tool_start",
                "tool_name": "baidu_search",
                "input_signature": '{"query": "A"}',
            }
            for _ in range(3)
        ]

        signal = _is_stuck(trace, threshold=3)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.kind, "repeated_tool_call")

    def test_is_stuck_detects_consecutive_tool_failures(self):
        trace = [
            {"event": "tool_result", "tool_name": "api", "status": "error"}
            for _ in range(3)
        ]

        signal = _is_stuck(trace, threshold=3)

        self.assertIsNotNone(signal)
        self.assertEqual(signal.kind, "consecutive_tool_failure")

    def test_recovery_changes_query_then_lowers_top_k(self):
        first = attempt_loop_recovery(
            task_tool="baidu_search",
            payload={"query": "A", "top_k": 5},
            recovery_attempt=0,
        )
        second = attempt_loop_recovery(
            task_tool="baidu_search",
            payload={"query": "A", "top_k": 5},
            recovery_attempt=1,
        )

        self.assertEqual(first.strategy, "modify_query")
        self.assertNotEqual(first.updated_input["query"], "A")
        self.assertEqual(second.strategy, "lower_top_k")
        self.assertEqual(second.updated_input["top_k"], 2)

    def test_recovery_switches_to_rag_after_search_options(self):
        action = attempt_loop_recovery(
            task_tool="baidu_search",
            payload={"query": "A", "top_k": 1},
            recovery_attempt=2,
        )

        self.assertEqual(action.strategy, "switch_to_rag")
        self.assertEqual(action.fallback_tool, "knowledge_base")

    def test_reliability_limits_are_configuration_driven(self):
        with patch.dict(
            os.environ,
            {
                "MAX_TOOL_RETRIES": "1",
                "MAX_TASK_RETRIES": "2",
                "MAX_AGENT_STEPS": "7",
                "MAX_RECOVERY_ATTEMPTS": "4",
            },
            clear=False,
        ):
            config = get_reliability_config()

        self.assertEqual(config.max_tool_retries, 1)
        self.assertEqual(config.max_task_retries, 2)
        self.assertEqual(config.max_agent_steps, 7)
        self.assertEqual(config.max_recovery_attempts, 4)

    def test_circuit_breaker_opens_after_failures(self):
        breaker = CircuitBreaker()
        with patch.dict(
            os.environ,
            {
                "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "2",
                "CIRCUIT_BREAKER_RESET_SECONDS": "60",
            },
            clear=False,
        ):
            config = get_reliability_config()
            breaker.record_failure("api", "HTTP 500", config)
            breaker.record_failure("api", "HTTP 500", config)
            allowed, reason = breaker.allow("api", config)

        self.assertFalse(allowed)
        self.assertIn("circuit is open", reason)

    def test_registry_returns_circuit_open_without_calling_tool(self):
        tool = AlwaysFailTool()
        registry = ToolRegistry()
        registry.register(tool)
        with patch.dict(
            os.environ,
            {
                "CIRCUIT_BREAKER_FAILURE_THRESHOLD": "1",
                "CIRCUIT_BREAKER_RESET_SECONDS": "60",
            },
            clear=False,
        ):
            first = registry.execute("always_fail", query="429")
            second = registry.execute("always_fail", query="429")

        self.assertFalse(first.success)
        self.assertFalse(second.success)
        self.assertEqual(second.trace.metadata["error_type"], "circuit_open")
        self.assertEqual(tool.calls, 1)

    def test_api_429_retry_is_bounded_by_configuration(self):
        tool = AlwaysFailTool("HTTP 429", retryable=True)
        tool.max_retries = 5
        with patch.dict(os.environ, {"MAX_TOOL_RETRIES": "1"}, clear=False):
            result = tool.invoke(query="rate limited")

        self.assertFalse(result.success)
        self.assertEqual(result.trace.attempts, 2)
        self.assertEqual(tool.calls, 2)

    def test_api_500_is_returned_as_a_tool_failure(self):
        result = AlwaysFailTool("HTTP 500", retryable=False).invoke(query="provider")

        self.assertFalse(result.success)
        self.assertIn("HTTP 500", result.error)

    def test_unregistered_tool_is_reported_as_unavailable(self):
        registry = ToolRegistry()

        with self.assertRaisesRegex(KeyError, "not registered"):
            registry.execute("unavailable_tool", query="test")

    def test_timeout_is_bounded(self):
        result = SlowReliabilityTool().invoke()
        self.assertEqual(result.status, ToolStatus.TIMEOUT)


class PlannerReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_retry_and_query_recovery(self):
        calls = []
        registry = ToolRegistry()
        registry.register(RecordingReliabilityTool(calls))
        executor = TaskExecutor(model=FakeModel(), tool_registry=registry)
        plan = Plan(
            tasks=[
                {
                    "id": "task_1",
                    "description": "search",
                    "tool": "baidu_search",
                    "input": {"query": "A", "top_k": 5},
                    "max_retries": 0,
                }
            ]
        )

        result = await executor.execute(plan, "A")

        self.assertEqual(result.state.completed_tasks, ["task_1"])
        self.assertGreaterEqual(result.state.recovery_attempts, 1)
        self.assertNotEqual(calls[0]["query"], calls[-1]["query"])

    async def test_repeated_planner_tasks_are_skipped(self):
        calls = []
        registry = ToolRegistry()
        registry.register(RecordingReliabilityTool(calls))
        executor = TaskExecutor(model=FakeModel(), tool_registry=registry)
        plan = Plan(
            tasks=[
                {
                    "id": f"task_{index}",
                    "description": "same search",
                    "tool": "baidu_search",
                    "input": {"query": "same", "top_k": 1},
                }
                for index in range(1, 4)
            ]
        )

        result = await executor.execute(plan, "same")

        self.assertEqual(result.state.plan.tasks[-1].status.value, "skipped")
        self.assertTrue(result.state.recovery_events)

    async def test_cancellation_stops_task_executor(self):
        cancel_event = asyncio.Event()
        registry = ToolRegistry()
        registry.register(SlowBaiduTool())
        executor = TaskExecutor(
            model=FakeModel(),
            tool_registry=registry,
            cancel_event=cancel_event,
        )
        plan = Plan(
            tasks=[
                {
                    "id": "task_1",
                    "description": "slow",
                    "tool": "baidu_search",
                    "timeout_seconds": 1,
                }
            ]
        )

        async def cancel_soon():
            await asyncio.sleep(0.01)
            cancel_event.set()

        run_task = asyncio.create_task(executor.execute(plan, "cancel"))
        cancel_task = asyncio.create_task(cancel_soon())
        result = await run_task
        await cancel_task

        self.assertTrue(result.cancelled)


if __name__ == "__main__":
    unittest.main()
