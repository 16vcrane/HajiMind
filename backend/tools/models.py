from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolExecutionTrace(BaseModel):
    tool_name: str
    status: ToolStatus
    latency_ms: int
    attempts: int = 1
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    status: ToolStatus
    data: Any = None
    error: str | None = None
    trace: ToolExecutionTrace

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def to_agent_content(self) -> str:
        """Convert the structured result to the existing text-only agent contract."""
        if not self.success:
            return self.error or f"{self.tool_name} 执行失败"
        if isinstance(self.data, str):
            return self.data
        if isinstance(self.data, dict) and isinstance(self.data.get("content"), str):
            return self.data["content"]
        return str(self.data)
