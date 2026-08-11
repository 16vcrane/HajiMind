import asyncio
import sys
import time
import types
import unittest
from unittest.mock import patch

from pydantic import BaseModel

sys.path.insert(0, "backend")

import agent
from tools import (
    BaseTool,
    KnowledgeBaseTool,
    ToolExecutionError,
    ToolRegistry,
    ToolStatus,
    WeatherTool,
)
from tools.context import get_last_rag_context, reset_tool_call_guards


class ValueInput(BaseModel):
    value: int


class EchoTool(BaseTool):
    name = "echo"
    description = "Echo a numeric value."
    input_schema = ValueInput

    def _invoke(self, input_data: ValueInput):
        return {"value": input_data.value}


class FlakyTool(BaseTool):
    name = "flaky"
    description = "Fails once before succeeding."
    input_schema = ValueInput
    max_retries = 1
    retry_backoff_seconds = 0

    def __init__(self):
        self.calls = 0

    def _invoke(self, input_data: ValueInput):
        self.calls += 1
        if self.calls == 1:
            raise ToolExecutionError("temporary failure", retryable=True)
        return "recovered"


class SlowTool(BaseTool):
    name = "slow"
    description = "Exceeds its timeout."
    input_schema = ValueInput
    timeout_seconds = 0.01

    def _invoke(self, input_data: ValueInput):
        time.sleep(0.05)
        return "late"


class ToolInfrastructureTests(unittest.TestCase):
    def test_success_result_includes_execution_trace(self):
        result = EchoTool().invoke(value=7)

        self.assertTrue(result.success)
        self.assertEqual(result.data, {"value": 7})
        self.assertEqual(result.trace.tool_name, "echo")
        self.assertEqual(result.trace.status, ToolStatus.SUCCESS)
        self.assertGreaterEqual(result.trace.latency_ms, 0)

    def test_validation_error_is_returned_as_a_tool_result(self):
        result = EchoTool().invoke(value="not-an-integer")

        self.assertFalse(result.success)
        self.assertEqual(result.status, ToolStatus.ERROR)
        self.assertEqual(result.trace.metadata["error_type"], "validation_error")

    def test_retry_records_total_attempts(self):
        tool = FlakyTool()
        result = tool.invoke(value=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data, "recovered")
        self.assertEqual(result.trace.attempts, 2)
        self.assertEqual(tool.calls, 2)

    def test_timeout_returns_a_timeout_trace(self):
        result = SlowTool().invoke(value=1)

        self.assertFalse(result.success)
        self.assertEqual(result.status, ToolStatus.TIMEOUT)
        self.assertEqual(result.trace.status, ToolStatus.TIMEOUT)

    def test_registry_registers_and_adapts_tools_for_langchain(self):
        registry = ToolRegistry()
        registry.register(EchoTool())

        self.assertEqual(registry.execute("echo", value=3).data, {"value": 3})
        self.assertEqual(registry.as_langchain_tools()[0].name, "echo")
        with self.assertRaises(ValueError):
            registry.register(EchoTool())

    def test_weather_tool_preserves_existing_configuration_error_content(self):
        with patch.dict(
            "os.environ",
            {"AMAP_WEATHER_API": "", "AMAP_API_KEY": ""},
            clear=False,
        ):
            result = WeatherTool().invoke(location="上海")

        self.assertFalse(result.success)
        self.assertEqual(result.to_agent_content(), "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）")

    def test_knowledge_tool_preserves_rag_result_format_and_trace(self):
        fake_rag = types.ModuleType("rag_pipeline")
        fake_rag.run_rag_graph = lambda query: {
            "docs": [{"filename": "guide.pdf", "page_number": 2, "text": "RAG content"}],
            "rag_trace": {"tool_used": True, "tool_name": "search_knowledge_base"},
        }

        reset_tool_call_guards()
        get_last_rag_context(clear=True)
        with patch.dict(sys.modules, {"rag_pipeline": fake_rag}):
            result = KnowledgeBaseTool().invoke(query="RAG 是什么")

        self.assertTrue(result.success)
        self.assertIn("guide.pdf", result.to_agent_content())
        self.assertEqual(get_last_rag_context()["rag_trace"]["tool_name"], "search_knowledge_base")

    def test_router_failure_context_falls_back_to_existing_agent(self):
        with patch.object(agent.router, "decide", side_effect=RuntimeError("router down")):
            route_context = agent._select_route("test question", [])

        self.assertTrue(route_context["fallback"])
        self.assertEqual(route_context["route_target"], "fallback")
        self.assertIn("router down", route_context["router_error"])


class ToolAsyncInfrastructureTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_invoke_returns_the_same_structured_contract(self):
        result = await EchoTool().async_invoke(value=9)

        self.assertTrue(result.success)
        self.assertEqual(result.data, {"value": 9})

    async def test_streaming_contract_keeps_content_and_done_events(self):
        class FakeAgent:
            async def astream(self, *_args, **_kwargs):
                from langchain_core.messages import AIMessageChunk

                yield AIMessageChunk(content="streamed answer"), {}

        with (
            patch.object(agent, "agent", FakeAgent()),
            patch.object(
                agent,
                "_select_route",
                return_value={
                    "decision": None,
                    "route_target": "fallback",
                    "router_error": "test fallback",
                    "fallback": True,
                },
            ),
            patch.object(agent.storage, "load", return_value=[]),
            patch.object(agent.storage, "save") as save,
            patch.object(agent.memory_service, "retrieve", return_value=[]),
            patch.object(
                agent.memory_service,
                "extract_and_store",
                return_value=types.SimpleNamespace(memory_ids=[], skipped_reasons=[]),
            ),
        ):
            events = [
                event
                async for event in agent.chat_with_agent_stream(
                    "test question",
                    user_id=1,
                    session_id="stream-test",
                )
            ]

        self.assertTrue(any('"type": "content"' in event for event in events))
        self.assertEqual(events[-1], "data: [DONE]\n\n")
        self.assertTrue(save.called)

    async def test_streaming_planner_events_are_forwarded(self):
        class FakePlannerResult:
            cancelled = False
            error = None

            class State:
                def model_dump(self, mode="json"):
                    return {
                        "plan": {"tasks": []},
                        "current_task": None,
                        "task_results": {},
                        "completed_tasks": ["task_1"],
                        "failed_tasks": [],
                        "tool_calls": [
                            {
                                "tool_name": "calendar_time",
                                "status": "success",
                                "latency_ms": 5,
                                "result_count": None,
                            }
                        ],
                        "task_traces": [
                            {
                                "task_id": "task_1",
                                "description": "测试任务",
                                "tool": "synthesis",
                                "status": "completed",
                                "latency_ms": 5,
                            }
                        ],
                        "execution_trace": [],
                    }

            state = State()
            response = "planned answer"

        class FakePlanner:
            async def run(self, *_args, event_handler=None, **_kwargs):
                for event_type in (
                    "planner_started",
                    "task_created",
                    "task_started",
                    "tool_start",
                    "tool_result",
                    "task_completed",
                    "synthesis_started",
                ):
                    event = {
                        "type": event_type,
                        "label": event_type,
                        "task": {
                            "id": "task_1",
                            "description": "测试任务",
                            "tool": "synthesis",
                        },
                    }
                    if event_type.startswith("tool_"):
                        event["tool_name"] = "calendar_time"
                    await event_handler(event)
                return FakePlannerResult()

        with (
            patch.object(agent, "planner", FakePlanner()),
            patch.object(
                agent,
                "_select_route",
                return_value={
                    "decision": None,
                    "route_target": "planner",
                    "router_error": None,
                    "fallback": False,
                },
            ),
            patch.object(agent.storage, "load", return_value=[]),
            patch.object(agent.storage, "save"),
            patch.object(agent.memory_service, "retrieve", return_value=[]),
            patch.object(
                agent.memory_service,
                "extract_and_store",
                return_value=types.SimpleNamespace(memory_ids=[], skipped_reasons=[]),
            ),
        ):
            events = [
                event
                async for event in agent.chat_with_agent_stream(
                    "test planner",
                    user_id=1,
                    session_id="planner-stream-test",
                )
            ]

        joined = "".join(events)
        self.assertIn('"type": "agent_step"', joined)
        self.assertIn('"type": "router_step"', joined)
        self.assertIn('"type": "planner_step"', joined)
        self.assertIn('"type": "planner_started"', joined)
        self.assertIn('"type": "tool_start"', joined)
        self.assertIn('"type": "tool_result"', joined)
        self.assertIn('"type": "task_start"', joined)
        self.assertIn('"type": "task_result"', joined)
        self.assertIn('"type": "task_completed"', joined)
        self.assertIn('"agent_trace"', joined)
        self.assertIn("planned answer", joined)


if __name__ == "__main__":
    unittest.main()
