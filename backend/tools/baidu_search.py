from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .base import BaseTool
from .errors import ToolExecutionError

load_dotenv()


class BaiduSearchInput(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=10)


BaiduSearchProvider = Callable[[str, int, str, str, float], Sequence[dict[str, Any]]]


class BaiduSearchTool(BaseTool):
    """Provider-neutral Baidu search tool pending API contract confirmation."""

    name = "baidu_search"
    description = "Search Baidu for web results with ranked titles, URLs, snippets, and source metadata."
    input_schema = BaiduSearchInput
    timeout_seconds = 10.0
    max_retries = 1

    def __init__(
        self,
        *,
        provider: BaiduSearchProvider | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        self._provider = provider
        if timeout_seconds is not None:
            self.timeout_seconds = timeout_seconds
        if max_retries is not None:
            self.max_retries = max_retries

    def health(self) -> dict[str, Any]:
        health = super().health()
        health["available"] = bool(
            self._provider
            or (os.getenv("BAIDU_SEARCH_API_KEY") and os.getenv("BAIDU_SEARCH_API_URL"))
        )
        health["provider_adapter_configured"] = self._provider is not None
        return health

    def _invoke(self, input_data: BaiduSearchInput) -> dict[str, Any]:
        query = input_data.query.strip()
        if not query:
            raise ToolExecutionError("query参数不能为空")

        api_key = os.getenv("BAIDU_SEARCH_API_KEY")
        api_url = os.getenv("BAIDU_SEARCH_API_URL")
        if not api_key or not api_url:
            raise ToolExecutionError(
                "百度搜索服务未配置（缺少 BAIDU_SEARCH_API_KEY 或 BAIDU_SEARCH_API_URL）",
                metadata={"error_type": "configuration_error"},
            )
        if self._provider is None:
            raise ToolExecutionError(
                "百度搜索 API 接口契约尚未配置。请提供认证方式、请求参数和响应示例后实现 Provider Adapter。",
                metadata={"error_type": "provider_adapter_missing"},
            )

        raw_results = self._provider(
            query,
            input_data.top_k,
            api_url,
            api_key,
            self.timeout_seconds,
        )
        results = [
            self._normalize_result(item, rank)
            for rank, item in enumerate(raw_results[: input_data.top_k], start=1)
        ]
        return {
            "results": results,
            "content": self._format_content(query, results),
        }

    @staticmethod
    def _normalize_result(item: dict[str, Any], rank: int) -> dict[str, Any]:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        if not title or not url:
            raise ToolExecutionError(
                "百度搜索 Provider Adapter 返回的结果缺少 title 或 url",
                metadata={"error_type": "invalid_provider_response"},
            )
        snippet = str(item.get("snippet", "")).strip()
        source = str(item.get("source") or "baidu").strip()
        published_at = item.get("published_at")
        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "source": source,
            "published_at": str(published_at) if published_at else None,
            "rank": rank,
        }

    @staticmethod
    def _format_content(query: str, results: list[dict[str, Any]]) -> str:
        if not results:
            return f"未找到与“{query}”相关的百度搜索结果。"
        lines = [f"百度搜索结果：{query}"]
        for result in results:
            lines.extend(
                [
                    f"[{result['rank']}] {result['title']}",
                    result["url"],
                    result["snippet"],
                ]
            )
        return "\n".join(lines)
