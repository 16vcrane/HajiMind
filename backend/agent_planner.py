from __future__ import annotations

import asyncio
import inspect
import json
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, model_validator

from agent_reliability import (
    RecoveryAction,
    _is_stuck,
    attempt_loop_recovery,
    get_reliability_config,
    payload_signature,
    retry_delay_seconds,
)
from agent_observability import (
    TaskExecutionTrace,
    ToolCallTrace,
    monotonic_ms,
    sanitize_value,
    summarize_value,
    task_trace_from_task,
    tool_call_from_result,
    utc_now_iso,
)
from tools import ToolRegistry
from tools.models import ToolResult, ToolStatus


TaskTool = Literal[
    "calendar",
    "calendar_time",
    "baidu_search",
    "web_search",
    "knowledge_base",
    "rag",
    "weather",
    "life_service",
    "synthesis",
]


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class Task(BaseModel):
    id: str
    description: str
    tool: TaskTool
    dependencies: list[str] = Field(default_factory=list)
    input: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=1, ge=0, le=3)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    attempts: int = 0


class Plan(BaseModel):
    tasks: list[Task]

    @model_validator(mode="after")
    def validate_task_graph(self):
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan task ids must be unique")
        known = set(ids)
        for task in self.tasks:
            unknown = [dep for dep in task.dependencies if dep not in known]
            if unknown:
                raise ValueError(f"Task {task.id} has unknown dependencies: {unknown}")
        self._validate_acyclic()
        return self

    def _validate_acyclic(self) -> None:
        graph = {task.id: set(task.dependencies) for task in self.tasks}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError(f"Plan contains a dependency cycle at {task_id}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dep in graph[task_id]:
                visit(dep)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in graph:
            visit(task_id)


class PlannerState(BaseModel):
    plan: Plan | None = None
    current_task: str | None = None
    task_results: dict[str, Any] = Field(default_factory=dict)
    completed_tasks: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    task_traces: list[TaskExecutionTrace] = Field(default_factory=list)
    execution_trace: list[dict[str, Any]] = Field(default_factory=list)
    recovery_attempts: int = 0
    recovery_events: list[dict[str, Any]] = Field(default_factory=list)


class PlannerRunResult(BaseModel):
    response: str
    state: PlannerState
    cancelled: bool = False
    error: str | None = None


PlannerEventHandler = Callable[[dict[str, Any]], Awaitable[None] | None]


class PlannerError(Exception):
    pass


class PlannerCancelled(Exception):
    pass


PLANNER_SYSTEM_PROMPT = """你是 HajiMind 的多步骤任务 Planner。
请把用户问题拆成可执行任务，输出严格结构化 Plan。

可用 tool：
- calendar：当前日期、日期计算、星期、时区转换。
- baidu_search：最新网络资料搜索。
- knowledge_base：用户知识库 / 文档检索。
- weather：天气与天气预报。
- synthesis：综合前置任务结果，生成最终答案或计划。

规则：
- 没有依赖关系的外部工具任务应保持 dependencies=[]，以便并行执行。
- synthesis 任务必须依赖所有需要汇总的前置任务。
- task id 使用 task_1、task_2 这种稳定格式。
- 每个非 synthesis 任务应给出尽量明确的 input。
- mixed 或 complex 任务必须包含 synthesis。
"""


class MultiStepPlanner:
    def __init__(
        self,
        *,
        model,
        tool_registry: ToolRegistry,
        default_task_timeout_seconds: float = 30.0,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.default_task_timeout_seconds = default_task_timeout_seconds

    def create_plan(self, user_text: str, router_decision: dict[str, Any] | None = None) -> Plan:
        prompt = (
            "请根据用户问题制定多步骤执行计划。\n"
            f"用户问题：{user_text}\n"
            f"RouterDecision：{json.dumps(router_decision or {}, ensure_ascii=False)}"
        )
        try:
            structured_model = self.model.with_structured_output(Plan)
            plan = structured_model.invoke(
                [
                    SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            raise PlannerError(f"Planner 生成计划失败: {exc}") from exc
        if not plan.tasks:
            raise PlannerError("Planner 生成了空计划")
        return self._apply_defaults(plan)

    async def run(
        self,
        user_text: str,
        router_decision: dict[str, Any] | None = None,
        *,
        event_handler: PlannerEventHandler | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> PlannerRunResult:
        await self._emit(event_handler, "planner_started", "正在制定计划")
        plan = self.create_plan(user_text, router_decision)
        executor = TaskExecutor(
            model=self.model,
            tool_registry=self.tool_registry,
            event_handler=event_handler,
            cancel_event=cancel_event,
        )
        return await executor.execute(plan, user_text)

    def run_sync(
        self,
        user_text: str,
        router_decision: dict[str, Any] | None = None,
    ) -> PlannerRunResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(user_text, router_decision))
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="planner-sync") as executor:
            future = executor.submit(lambda: asyncio.run(self.run(user_text, router_decision)))
            return future.result()

    def _apply_defaults(self, plan: Plan) -> Plan:
        for task in plan.tasks:
            if task.timeout_seconds == 30.0:
                task.timeout_seconds = self.default_task_timeout_seconds
        return plan

    @staticmethod
    async def _emit(
        event_handler: PlannerEventHandler | None,
        event_type: str,
        label: str,
        **payload: Any,
    ) -> None:
        if not event_handler:
            return
        event = {"type": event_type, "label": label, **payload}
        result = event_handler(event)
        if inspect.isawaitable(result):
            await result


class TaskExecutor:
    def __init__(
        self,
        *,
        model,
        tool_registry: ToolRegistry,
        event_handler: PlannerEventHandler | None = None,
        cancel_event: asyncio.Event | None = None,
    ):
        self.model = model
        self.tool_registry = tool_registry
        self.event_handler = event_handler
        self.cancel_event = cancel_event or asyncio.Event()
        self.active_tasks: set[asyncio.Task] = set()
        self.state = PlannerState()
        self._task_started_at: dict[str, float] = {}
        self._task_start_time: dict[str, str] = {}
        self.reliability_config = get_reliability_config()
        self._agent_steps = 0
        self._task_recovery_attempts: dict[str, int] = {}

    async def execute(self, plan: Plan, user_text: str) -> PlannerRunResult:
        self.state.plan = plan
        self._recover_repeated_plan_tasks(plan)
        for task in plan.tasks:
            await self._emit("task_created", "任务已创建", task=task.model_dump())

        pending: dict[str, Task] = {
            task.id: task
            for task in plan.tasks
            if task.status not in (TaskStatus.SKIPPED, TaskStatus.CANCELLED)
        }
        try:
            while pending:
                self._raise_if_cancelled()
                self._record_planner_iteration(len(pending))
                self._raise_if_stuck()
                ready = [
                    task
                    for task in pending.values()
                    if all(dep in self.state.completed_tasks for dep in task.dependencies)
                ]
                blocked_by_failed = [
                    task
                    for task in pending.values()
                    if any(dep in self.state.failed_tasks for dep in task.dependencies)
                ]
                for task in blocked_by_failed:
                    task.status = TaskStatus.SKIPPED
                    task.error = "dependency failed"
                    self.state.failed_tasks.append(task.id)
                    self._task_start_time[task.id] = utc_now_iso()
                    self.state.task_traces.append(self._task_trace(task, ended=True))
                    self._trace(task, "task_failed", error=task.error)
                    await self._emit(
                        "task_failed",
                        "任务失败",
                        task=task.model_dump(),
                        error=task.error,
                        task_trace=self.state.task_traces[-1].model_dump(mode="json"),
                    )
                    pending.pop(task.id)

                ready = [task for task in ready if task.id in pending]
                if not ready:
                    if pending:
                        raise PlannerError("没有可执行任务，可能存在未满足依赖")
                    break

                if len(ready) > 1:
                    await self._emit(
                        "parallel_execution_started",
                        "并行执行任务",
                        task_ids=[task.id for task in ready],
                    )

                self._consume_agent_steps(len(ready))
                task_futures = [asyncio.create_task(self._run_task(task, user_text)) for task in ready]
                self.active_tasks.update(task_futures)
                try:
                    await asyncio.gather(*task_futures)
                finally:
                    self.active_tasks.difference_update(task_futures)
                for task in ready:
                    pending.pop(task.id, None)
        except PlannerCancelled:
            await self.cancel()
            return PlannerRunResult(
                response="任务已取消。",
                state=self.state,
                cancelled=True,
                error="cancelled",
            )

        final_response = self._final_response()
        return PlannerRunResult(response=final_response, state=self.state)

    async def cancel(self) -> None:
        self.cancel_event.set()
        for task in list(self.active_tasks):
            task.cancel()
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

    async def _run_task(self, task: Task, user_text: str) -> None:
        self._raise_if_cancelled()
        task.status = TaskStatus.RUNNING
        self.state.current_task = task.id
        self._task_started_at[task.id] = time.perf_counter()
        self._task_start_time[task.id] = utc_now_iso()
        self._trace(task, "task_started")
        await self._emit(
            "task_started",
            "任务开始",
            task=task.model_dump(),
            task_trace=self._task_trace(task).model_dump(mode="json"),
        )

        attempts = 0
        max_retries = min(task.max_retries, self.reliability_config.max_task_retries)
        while True:
            self._raise_if_cancelled()
            attempts += 1
            task.attempts = attempts
            try:
                if task.tool == "synthesis":
                    await self._emit("synthesis_started", "正在综合结果", task=task.model_dump())
                result = await self._execute_with_timeout_and_cancellation(
                    task,
                    user_text,
                )
                if task.status in (TaskStatus.SKIPPED, TaskStatus.CANCELLED):
                    return
                if task.id in self.state.completed_tasks:
                    return
                task.result = result
                task.status = TaskStatus.COMPLETED
                self.state.task_results[task.id] = result
                self.state.completed_tasks.append(task.id)
                self.state.task_traces.append(self._task_trace(task, ended=True))
                self._trace(task, "task_completed")
                await self._emit(
                    "task_completed",
                    "任务完成",
                    task=task.model_dump(),
                    task_trace=self.state.task_traces[-1].model_dump(mode="json"),
                )
                return
            except asyncio.CancelledError as exc:
                task.status = TaskStatus.CANCELLED
                task.error = "cancelled"
                self.state.failed_tasks.append(task.id)
                self.state.task_traces.append(self._task_trace(task, ended=True))
                self._trace(task, "task_failed", error=task.error)
                raise PlannerCancelled() from exc
            except asyncio.TimeoutError:
                error = f"任务执行超时（>{task.timeout_seconds:g}s）"
                if attempts <= max_retries:
                    await asyncio.sleep(retry_delay_seconds(0, attempts))
                    continue
                recovered = False
                if self._is_recoverable_error(error):
                    recovered = await self._handle_task_recovery(task, user_text, error)
                if recovered:
                    attempts = 0
                    max_retries = 0
                    continue
                if task.status == TaskStatus.SKIPPED or task.result is not None:
                    return
                await self._mark_failed(task, error)
                return
            except PlannerCancelled:
                task.status = TaskStatus.CANCELLED
                task.error = "cancelled"
                self.state.failed_tasks.append(task.id)
                self.state.task_traces.append(self._task_trace(task, ended=True))
                self._trace(task, "task_failed", error=task.error)
                raise
            except Exception as exc:
                error = str(exc)
                if attempts <= max_retries:
                    await asyncio.sleep(retry_delay_seconds(0, attempts))
                    continue
                recovered = False
                if self._is_recoverable_error(error):
                    recovered = await self._handle_task_recovery(task, user_text, error)
                if recovered:
                    attempts = 0
                    max_retries = 0
                    continue
                if task.status == TaskStatus.SKIPPED or task.result is not None:
                    return
                await self._mark_failed(task, error)
                return

    async def _execute_task_once(self, task: Task, user_text: str) -> Any:
        self._raise_if_cancelled()
        if task.tool == "synthesis":
            return await self._synthesize(task, user_text)

        tool_name = self._resolve_tool_name(task.tool)
        payload = self._build_tool_input(task, user_text)
        signal = self._record_tool_start(task, tool_name, payload)
        if signal:
            recovered = await self._handle_task_recovery(task, user_text, signal.reason, signal=signal)
            if recovered:
                return await self._execute_task_once(task, user_text)
            if task.status == TaskStatus.SKIPPED or task.result is not None:
                return task.result
        tool_started_at = time.perf_counter()
        tool_start_time = utc_now_iso()
        await self._emit(
            "tool_start",
            "工具开始",
            task_id=task.id,
            tool_name=tool_name,
            input_summary=sanitize_value(payload),
            start_time=tool_start_time,
        )
        result: ToolResult = await self.tool_registry.async_execute(tool_name, **payload)
        tool_trace = tool_call_from_result(result, payload)
        tool_trace.start_time = tool_start_time
        tool_trace.end_time = utc_now_iso()
        tool_trace.latency_ms = monotonic_ms(tool_started_at)
        self.state.tool_calls.append(tool_trace)
        self._trace(
            task,
            "tool_result",
            tool_name=tool_name,
            status=result.status.value,
            error=result.error,
            input_signature=payload_signature(payload),
        )
        await self._emit(
            "tool_result",
            "工具完成",
            task_id=task.id,
            tool_name=tool_name,
            tool_trace=tool_trace.model_dump(mode="json"),
        )
        if result.status != ToolStatus.SUCCESS:
            raise PlannerError(result.error or f"{tool_name} 执行失败")
        return {
            "tool_name": result.tool_name,
            "data": result.data,
            "trace": result.trace.model_dump(mode="json"),
            "content": result.to_agent_content(),
        }

    async def _execute_with_timeout_and_cancellation(
        self,
        task: Task,
        user_text: str,
    ) -> Any:
        execution = asyncio.create_task(self._execute_task_once(task, user_text))
        cancellation = asyncio.create_task(self.cancel_event.wait())
        try:
            done, pending = await asyncio.wait(
                {execution, cancellation},
                timeout=task.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation in done and self.cancel_event.is_set():
                execution.cancel()
                raise PlannerCancelled()
            if execution in done:
                return await execution
            execution.cancel()
            raise asyncio.TimeoutError()
        finally:
            for pending_task in (execution, cancellation):
                if not pending_task.done():
                    pending_task.cancel()

    async def _synthesize(self, task: Task, user_text: str) -> str:
        dependency_results = {
            dep: self.state.task_results.get(dep)
            for dep in task.dependencies
        }
        prompt = (
            "请基于以下多步骤任务结果回答用户问题。不要暴露隐藏推理过程。\n"
            f"用户问题：{user_text}\n"
            f"当前综合任务：{task.description}\n"
            f"任务结果：{json.dumps(dependency_results, ensure_ascii=False, default=str)}"
        )
        response = await asyncio.to_thread(
            self.model.invoke,
            [SystemMessage(content="你负责综合工具结果生成最终答案。"), HumanMessage(content=prompt)],
        )
        return getattr(response, "content", str(response))

    async def _mark_failed(self, task: Task, error: str) -> None:
        task.status = TaskStatus.FAILED
        task.error = error
        self.state.failed_tasks.append(task.id)
        self.state.task_traces.append(self._task_trace(task, ended=True))
        self._trace(task, "task_failed", error=error)
        await self._emit(
            "task_failed",
            "任务失败",
            task=task.model_dump(),
            error=error,
            task_trace=self.state.task_traces[-1].model_dump(mode="json"),
        )

    async def _handle_task_recovery(
        self,
        task: Task,
        user_text: str,
        error: str,
        *,
        signal=None,
    ) -> bool:
        attempt = self._task_recovery_attempts.get(task.id, 0)
        payload = self._build_tool_input(task, user_text)
        action = attempt_loop_recovery(
            task_tool=task.tool,
            payload=payload,
            error=error,
            recovery_attempt=attempt,
            signal=signal,
            config=self.reliability_config,
        )
        self._task_recovery_attempts[task.id] = attempt + 1
        self.state.recovery_attempts += 1
        event = self._record_recovery_event(task, action, error)
        await self._emit(
            "planner_step",
            "正在恢复任务",
            task=task.model_dump(),
            recovery=event,
        )

        if action.action == "retry" and action.updated_input is not None:
            task.input = action.updated_input
            return True

        if action.action == "fallback_tool" and action.fallback_tool:
            task.tool = action.fallback_tool  # type: ignore[assignment]
            task.input = action.updated_input or {}
            return True

        if action.action == "complete_with_message" and action.user_message:
            task.result = {"content": action.user_message}
            task.status = TaskStatus.COMPLETED
            if task.id not in self.state.completed_tasks:
                self.state.completed_tasks.append(task.id)
            self.state.task_results[task.id] = task.result
            self.state.task_traces.append(self._task_trace(task, ended=True))
            self._trace(task, "task_completed", recovery_strategy=action.strategy)
            await self._emit(
                "task_completed",
                "任务完成",
                task=task.model_dump(),
                task_trace=self.state.task_traces[-1].model_dump(mode="json"),
            )
            return False

        if action.skip_task:
            task.status = TaskStatus.SKIPPED
            task.error = action.reason
            if task.id not in self.state.failed_tasks:
                self.state.failed_tasks.append(task.id)
            self.state.task_traces.append(self._task_trace(task, ended=True))
            self._trace(task, "task_failed", error=action.reason, recovery_strategy=action.strategy)
            await self._emit(
                "task_failed",
                "任务跳过",
                task=task.model_dump(),
                error=action.reason,
                task_trace=self.state.task_traces[-1].model_dump(mode="json"),
            )
            return False

        return False

    def _record_recovery_event(
        self,
        task: Task,
        action: RecoveryAction,
        error: str,
    ) -> dict[str, Any]:
        event = {
            "event": "recovery_attempted",
            "task_id": task.id,
            "tool": task.tool,
            "strategy": action.strategy,
            "action": action.action,
            "reason": action.reason,
            "error": error,
            "timestamp_ms": round(time.time() * 1000),
        }
        self.state.recovery_events.append(event)
        self.state.execution_trace.append(event)
        return event

    def _build_tool_input(self, task: Task, user_text: str) -> dict[str, Any]:
        if task.input:
            return dict(task.input)
        if task.tool in ("calendar", "calendar_time"):
            return {"operation": "get_current_datetime"}
        if task.tool in ("baidu_search", "web_search"):
            return {"query": user_text, "top_k": 5}
        if task.tool in ("knowledge_base", "rag"):
            return {"query": user_text}
        if task.tool in ("weather", "life_service"):
            return {"operation": "current_weather", "location": user_text}
        return {}

    @staticmethod
    def _resolve_tool_name(tool: str) -> str:
        aliases = {
            "calendar": "calendar_time",
            "calendar_time": "calendar_time",
            "baidu_search": "baidu_search",
            "web_search": "baidu_search",
            "knowledge_base": "search_knowledge_base",
            "rag": "search_knowledge_base",
            "weather": "life_service",
            "life_service": "life_service",
        }
        try:
            return aliases[tool]
        except KeyError as exc:
            raise PlannerError(f"未知任务工具: {tool}") from exc

    def _final_response(self) -> str:
        synthesis_results = [
            result
            for task_id, result in self.state.task_results.items()
            if self._task_by_id(task_id).tool == "synthesis"
        ]
        if synthesis_results:
            return str(synthesis_results[-1])
        if self.state.failed_tasks and not self.state.completed_tasks:
            return "任务执行失败，未能生成结果。"
        return "\n\n".join(
            self._result_to_text(task_id, result)
            for task_id, result in self.state.task_results.items()
        )

    def _task_by_id(self, task_id: str) -> Task:
        if not self.state.plan:
            raise PlannerError("Planner state has no plan")
        for task in self.state.plan.tasks:
            if task.id == task_id:
                return task
        raise PlannerError(f"Unknown task id: {task_id}")

    @staticmethod
    def _result_to_text(task_id: str, result: Any) -> str:
        if isinstance(result, dict) and isinstance(result.get("content"), str):
            return f"{task_id}: {result['content']}"
        return f"{task_id}: {result}"

    def _raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise PlannerCancelled()

    @staticmethod
    def _is_recoverable_error(error: str) -> bool:
        normalized = error.lower()
        return any(
            marker in normalized
            for marker in (
                "timeout",
                "超时",
                "http 429",
                "429",
                "http 500",
                "500",
                "circuit",
                "unavailable",
                "not registered",
            )
        )

    def _consume_agent_steps(self, count: int = 1) -> None:
        self._agent_steps += count
        if self._agent_steps > self.reliability_config.max_agent_steps:
            raise PlannerError(
                f"Agent exceeded MAX_AGENT_STEPS={self.reliability_config.max_agent_steps}"
            )

    def _record_planner_iteration(self, pending_count: int) -> None:
        self.state.execution_trace.append(
            {
                "event": "planner_iteration",
                "pending_count": pending_count,
                "completed_count": len(self.state.completed_tasks),
                "failed_count": len(self.state.failed_tasks),
                "timestamp_ms": round(time.time() * 1000),
            }
        )

    def _raise_if_stuck(self) -> None:
        signal = _is_stuck(
            self.state.execution_trace,
            threshold=self.reliability_config.stuck_repeat_threshold,
        )
        if signal and signal.kind == "planner_no_progress":
            raise PlannerError(f"Planner stuck: {signal.reason}")

    def _record_tool_start(self, task: Task, tool_name: str, payload: dict[str, Any]):
        signature = payload_signature(payload)
        self._trace(
            task,
            "tool_start",
            tool_name=tool_name,
            input_signature=signature,
            query=payload.get("query"),
        )
        return _is_stuck(
            self.state.execution_trace,
            threshold=self.reliability_config.stuck_repeat_threshold,
        )

    def _recover_repeated_plan_tasks(self, plan: Plan) -> None:
        seen: dict[str, int] = {}
        for task in plan.tasks:
            signature = self._task_signature(task)
            seen[signature] = seen.get(signature, 0) + 1
            if seen[signature] < self.reliability_config.stuck_repeat_threshold:
                continue
            task.status = TaskStatus.SKIPPED
            task.error = "repeated planner task skipped"
            if task.id not in self.state.failed_tasks:
                self.state.failed_tasks.append(task.id)
            event = {
                "event": "planner_recovery",
                "task_id": task.id,
                "tool": task.tool,
                "strategy": "skip_repeated_task",
                "reason": task.error,
                "timestamp_ms": round(time.time() * 1000),
            }
            self.state.recovery_events.append(event)
            self.state.execution_trace.append(event)

    @staticmethod
    def _task_signature(task: Task) -> str:
        return payload_signature(
            {
                "description": task.description,
                "tool": task.tool,
                "input": task.input,
                "dependencies": task.dependencies,
            }
        )

    def _trace(self, task: Task, event_type: str, **extra: Any) -> None:
        self.state.execution_trace.append(
            {
                "event": event_type,
                "task_id": task.id,
                "tool": task.tool,
                "status": task.status.value,
                "attempts": task.attempts,
                "task_signature": self._task_signature(task),
                "timestamp_ms": round(time.time() * 1000),
                **extra,
            }
        )

    def _task_trace(self, task: Task, *, ended: bool = False) -> TaskExecutionTrace:
        started_at = self._task_started_at.get(task.id)
        trace = task_trace_from_task(task, started_at=started_at)
        trace.start_time = self._task_start_time.get(task.id)
        if ended:
            trace.end_time = utc_now_iso()
        trace.input_summary = sanitize_value(task.input)
        trace.output_summary = summarize_value(task.result)
        trace.latency_ms = monotonic_ms(started_at) if started_at is not None else None
        return trace

    async def _emit(self, event_type: str, label: str, **payload: Any) -> None:
        if not self.event_handler:
            return
        event = {"type": event_type, "label": label, **payload}
        result = self.event_handler(event)
        if inspect.isawaitable(result):
            await result
