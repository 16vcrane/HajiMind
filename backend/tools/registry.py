from __future__ import annotations

from typing import Iterable

from langchain_core.tools import StructuredTool

from agent_reliability import circuit_breaker, get_reliability_config

from .base import BaseTool
from .models import ToolExecutionTrace, ToolResult, ToolStatus


class ToolRegistry:
    """Registry that owns project tools and exposes LangChain-compatible adapters."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> BaseTool:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Tool '{name}' is not registered") from exc

    def list(self) -> list[BaseTool]:
        return list(self._tools.values())

    def health(self) -> list[dict]:
        return [tool.health() for tool in self.list()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        config = get_reliability_config()
        allowed, reason = circuit_breaker.allow(name, config)
        if not allowed:
            return self._circuit_open_result(name, reason or "tool circuit is open")

        result = self.get(name).invoke(**kwargs)
        self._record_circuit_result(name, result, config)
        return result

    async def async_execute(self, name: str, **kwargs) -> ToolResult:
        config = get_reliability_config()
        allowed, reason = circuit_breaker.allow(name, config)
        if not allowed:
            return self._circuit_open_result(name, reason or "tool circuit is open")

        result = await self.get(name).async_invoke(**kwargs)
        self._record_circuit_result(name, result, config)
        return result

    def as_langchain_tools(self, names: Iterable[str] | None = None) -> list[StructuredTool]:
        selected = self.list() if names is None else [self.get(name) for name in names]
        return [self._as_langchain_tool(tool) for tool in selected]

    def _as_langchain_tool(self, tool: BaseTool) -> StructuredTool:
        def invoke_through_registry(**kwargs):
            return self.execute(tool.name, **kwargs).to_agent_content()

        return StructuredTool.from_function(
            func=invoke_through_registry,
            name=tool.name,
            description=tool.description,
            args_schema=tool.input_schema,
        )

    @staticmethod
    def _record_circuit_result(name: str, result: ToolResult, config) -> None:
        if result.success:
            circuit_breaker.record_success(name)
            return
        error_type = result.trace.metadata.get("error_type")
        if error_type in {"validation_error", "input_type_error", "configuration_error"}:
            return
        circuit_breaker.record_failure(name, result.error, config)

    @staticmethod
    def _circuit_open_result(name: str, reason: str) -> ToolResult:
        trace = ToolExecutionTrace(
            tool_name=name,
            status=ToolStatus.ERROR,
            latency_ms=0,
            attempts=0,
            error=reason,
            metadata={"error_type": "circuit_open"},
        )
        return ToolResult(tool_name=name, status=ToolStatus.ERROR, error=reason, trace=trace)
