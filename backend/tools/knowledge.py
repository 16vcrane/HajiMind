from __future__ import annotations

from pydantic import BaseModel

from .base import BaseTool
from .context import reserve_knowledge_tool_call, set_last_rag_context
from .errors import ToolExecutionError


class KnowledgeBaseInput(BaseModel):
    query: str


class KnowledgeBaseTool(BaseTool):
    name = "search_knowledge_base"
    description = "Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."
    input_schema = KnowledgeBaseInput
    timeout_seconds = 45.0
    max_retries = 0

    def _invoke(self, input_data: KnowledgeBaseInput) -> str:
        query = input_data.query.strip()
        if not query:
            raise ToolExecutionError("query参数不能为空")
        if not reserve_knowledge_tool_call():
            return (
                "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
                "Use the existing retrieval result and provide the final answer directly."
            )

        from rag_pipeline import run_rag_graph

        try:
            rag_result = run_rag_graph(query)
        except Exception as exc:
            raise ToolExecutionError(f"知识库检索失败: {exc}") from exc

        docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
        rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
        if rag_trace:
            set_last_rag_context({"rag_trace": rag_trace})

        if not docs:
            return "No relevant documents found in the knowledge base."

        conflict = rag_trace.get("conflict", {}) if isinstance(rag_trace, dict) else {}
        formatted = []
        for index, result in enumerate(docs, 1):
            source = result.get("filename", "Unknown")
            page = result.get("page_number", "N/A")
            text = result.get("text", "")
            formatted.append(f"[{index}] {source} (Page {page}):\n{text}")
        response = "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)
        if conflict.get("has_conflict"):
            conflict_lines = ["知识库中存在来源冲突。"]
            for claim in conflict.get("claims", [])[:6]:
                conflict_lines.extend(
                    [
                        f"来源: {claim.get('source', 'Unknown')}",
                        f"观点: {claim.get('claim', '')}",
                        f"证据: {claim.get('evidence', '')}",
                        f"时间: {claim.get('timestamp') or '未提供'}",
                    ]
                )
            response += "\n\n" + "\n".join(conflict_lines)
        return response
