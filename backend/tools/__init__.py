from .base import BaseTool
from .baidu_search import BaiduSearchTool
from .calendar_time import CalendarTimeTool
from .context import (
    emit_rag_step,
    get_last_rag_context,
    reset_tool_call_guards,
    set_rag_step_queue,
)
from .errors import ToolExecutionError, ToolTimeoutError
from .knowledge import KnowledgeBaseTool
from .life_service import LifeServiceTool
from .models import ToolExecutionTrace, ToolResult, ToolStatus
from .registry import ToolRegistry
from .weather import WeatherTool

tool_registry = ToolRegistry()
tool_registry.register(WeatherTool())
tool_registry.register(KnowledgeBaseTool())
tool_registry.register(BaiduSearchTool())
tool_registry.register(LifeServiceTool())
tool_registry.register(CalendarTimeTool())


def get_agent_tools():
    """Return the existing LangChain tool contract through the unified registry."""
    return tool_registry.as_langchain_tools(
        names=("get_current_weather", "search_knowledge_base")
    )


# Backward-compatible exports for existing imports and external scripts.
get_current_weather = tool_registry.get("get_current_weather").as_langchain_tool()
search_knowledge_base = tool_registry.get("search_knowledge_base").as_langchain_tool()

__all__ = [
    "BaseTool",
    "BaiduSearchTool",
    "CalendarTimeTool",
    "KnowledgeBaseTool",
    "LifeServiceTool",
    "ToolExecutionError",
    "ToolExecutionTrace",
    "ToolRegistry",
    "ToolResult",
    "ToolStatus",
    "ToolTimeoutError",
    "WeatherTool",
    "emit_rag_step",
    "get_agent_tools",
    "get_current_weather",
    "get_last_rag_context",
    "reset_tool_call_guards",
    "search_knowledge_base",
    "set_rag_step_queue",
    "tool_registry",
]
