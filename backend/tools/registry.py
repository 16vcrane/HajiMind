from __future__ import annotations

from typing import Iterable

from langchain_core.tools import StructuredTool

from .base import BaseTool
from .models import ToolResult


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
        return self.get(name).invoke(**kwargs)

    async def async_execute(self, name: str, **kwargs) -> ToolResult:
        return await self.get(name).async_invoke(**kwargs)

    def as_langchain_tools(self, names: Iterable[str] | None = None) -> list[StructuredTool]:
        selected = self.list() if names is None else [self.get(name) for name in names]
        return [tool.as_langchain_tool() for tool in selected]
