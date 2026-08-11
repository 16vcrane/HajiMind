import asyncio
import sys
import time
import unittest

from langchain_core.messages import AIMessage
from pydantic import BaseModel

sys.path.insert(0, "backend")

from agent_planner import MultiStepPlanner, Plan, PlannerError, TaskExecutor
from tools import BaseTool, ToolExecutionError, ToolRegistry


class GenericInput(BaseModel):
    query: str | None = None
    operation: str | None = None
    location: str | None = None
    top_k: int | None = None


class RecordingTool(BaseTool):
    description = "Recording test tool."
    input_schema = GenericInput
    timeout_seconds = 5
    max_retries = 0

    def __init__(self, name, calls, *, delay=0, fail=False):
        self.name = name
        self.calls = calls
        self.delay = delay
        self.fail = fail

    def _invoke(self, input_data):
        self.calls.append((self.name, time.perf_counter()))
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise ToolExecutionError(f"{self.name} failed")
        return {"content": f"{self.name} result"}


class FakeStructuredPlanner:
    def __init__(self, plan=None, error=None):
        self.plan = plan
        self.error = error

    def invoke(self, _messages):
        if self.error:
            raise self.error
        return self.plan


class FakePlannerModel:
    def __init__(self, plan=None, error=None):
        self.structured = FakeStructuredPlanner(plan=plan, error=error)

    def with_structured_output(self, _schema):
        return self.structured

    def invoke(self, _messages):
        return AIMessage(content="synthesized answer")


def registry_with_tools(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_task_execution(self):
        calls = []
        plan = Plan(
            tasks=[
                {
                    "id": "task_1",
                    "description": "获取当前日期",
                    "tool": "calendar",
                    "dependencies": [],
                }
            ]
        )
        planner = MultiStepPlanner(
            model=FakePlannerModel(plan),
            tool_registry=registry_with_tools(RecordingTool("calendar_time", calls)),
        )

        result = await planner.run("今天几号")

        self.assertFalse(result.cancelled)
        self.assertEqual(result.state.completed_tasks, ["task_1"])
        self.assertIn("calendar_time result", result.response)
        self.assertEqual(calls[0][0], "calendar_time")
        self.assertEqual(result.state.task_traces[0].status, "completed")
        self.assertIsNotNone(result.state.task_traces[0].start_time)
        self.assertIsNotNone(result.state.task_traces[0].end_time)
        self.assertGreaterEqual(result.state.task_traces[0].latency_ms, 0)
        self.assertEqual(result.state.tool_calls[0].tool_name, "calendar_time")
        self.assertEqual(result.state.tool_calls[0].status, "success")

    async def test_serial_dependencies_execute_in_order(self):
        calls = []
        plan = Plan(
            tasks=[
                {
                    "id": "task_1",
                    "description": "检索知识库",
                    "tool": "knowledge_base",
                    "dependencies": [],
                },
                {
                    "id": "task_2",
                    "description": "综合结果",
                    "tool": "synthesis",
                    "dependencies": ["task_1"],
                },
            ]
        )
        executor = TaskExecutor(
            model=FakePlannerModel(),
            tool_registry=registry_with_tools(RecordingTool("search_knowledge_base", calls)),
        )

        result = await executor.execute(plan, "总结知识库")

        self.assertEqual(result.response, "synthesized answer")
        self.assertEqual(result.state.completed_tasks, ["task_1", "task_2"])
        task_events = [event["event"] for event in result.state.execution_trace]
        self.assertEqual(
            task_events,
            ["task_started", "task_completed", "task_started", "task_completed"],
        )

    async def test_independent_tasks_execute_in_parallel(self):
        calls = []
        plan = Plan(
            tasks=[
                {
                    "id": "task_1",
                    "description": "搜索网络资料",
                    "tool": "baidu_search",
                    "dependencies": [],
                    "input": {"query": "LangGraph", "top_k": 1},
                },
                {
                    "id": "task_2",
                    "description": "获取当前日期",
                    "tool": "calendar",
                    "dependencies": [],
                },
            ]
        )
        events = []
        executor = TaskExecutor(
            model=FakePlannerModel(),
            tool_registry=registry_with_tools(
                RecordingTool("baidu_search", calls, delay=0.08),
                RecordingTool("calendar_time", calls, delay=0.08),
            ),
            event_handler=lambda event: events.append(event),
        )

        started = time.perf_counter()
        result = await executor.execute(plan, "并行任务")
        elapsed = time.perf_counter() - started

        self.assertEqual(set(result.state.completed_tasks), {"task_1", "task_2"})
        self.assertLess(elapsed, 0.15)
        self.assertTrue(any(event["type"] == "parallel_execution_started" for event in events))

    async def test_tool_failure_marks_task_failed(self):
        calls = []
        plan = Plan(
            tasks=[
                {
                    "id": "task_1",
                    "description": "搜索网络资料",
                    "tool": "baidu_search",
                    "dependencies": [],
                    "input": {"query": "LangGraph"},
                    "max_retries": 0,
                }
            ]
        )
        executor = TaskExecutor(
            model=FakePlannerModel(),
            tool_registry=registry_with_tools(
                RecordingTool("baidu_search", calls, fail=True)
            ),
        )

        result = await executor.execute(plan, "失败任务")

        self.assertEqual(result.state.failed_tasks, ["task_1"])
        self.assertEqual(result.state.plan.tasks[0].status.value, "failed")
        self.assertIn("failed", result.state.plan.tasks[0].error)

    async def test_task_timeout_marks_task_failed(self):
        calls = []
        plan = Plan(
            tasks=[
                {
                    "id": "task_1",
                    "description": "慢任务",
                    "tool": "calendar",
                    "dependencies": [],
                    "timeout_seconds": 0.02,
                    "max_retries": 0,
                }
            ]
        )
        executor = TaskExecutor(
            model=FakePlannerModel(),
            tool_registry=registry_with_tools(
                RecordingTool("calendar_time", calls, delay=0.1)
            ),
        )

        result = await executor.execute(plan, "慢任务")

        self.assertEqual(result.state.failed_tasks, ["task_1"])
        self.assertIn("超时", result.state.plan.tasks[0].error)

    async def test_agent_cancellation_stops_planner_execution(self):
        calls = []
        cancel_event = asyncio.Event()
        plan = Plan(
            tasks=[
                {
                    "id": "task_1",
                    "description": "慢任务",
                    "tool": "calendar",
                    "dependencies": [],
                    "timeout_seconds": 1,
                }
            ]
        )

        async def event_handler(event):
            if event["type"] == "task_started":
                cancel_event.set()

        executor = TaskExecutor(
            model=FakePlannerModel(),
            tool_registry=registry_with_tools(
                RecordingTool("calendar_time", calls, delay=0.1)
            ),
            event_handler=event_handler,
            cancel_event=cancel_event,
        )

        result = await executor.execute(plan, "取消任务")

        self.assertTrue(result.cancelled)
        self.assertEqual(result.error, "cancelled")

    async def test_planner_failure_raises_planner_error(self):
        planner = MultiStepPlanner(
            model=FakePlannerModel(error=RuntimeError("bad plan")),
            tool_registry=registry_with_tools(),
        )

        with self.assertRaises(PlannerError):
            await planner.run("无法规划")


if __name__ == "__main__":
    unittest.main()
