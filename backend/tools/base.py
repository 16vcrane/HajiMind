from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ValidationError

from .errors import ToolExecutionError, ToolTimeoutError
from .models import ToolExecutionTrace, ToolResult, ToolStatus

logger = logging.getLogger(__name__)


class BaseTool(ABC):
    """Common execution, validation, retry, and LangChain bridge for project tools."""

    name: str
    description: str
    input_schema: type[BaseModel]
    timeout_seconds: float = 10.0
    max_retries: int = 0
    retry_backoff_seconds: float = 0.25

    def invoke(self, input_data: BaseModel | dict[str, Any] | None = None, **kwargs: Any) -> ToolResult:
        started_at = time.perf_counter()
        try:
            payload = self._build_payload(input_data, kwargs)
        except TypeError as exc:
            return self._error_result(
                started_at,
                ToolStatus.ERROR,
                str(exc),
                attempts=1,
                metadata={"error_type": "input_type_error"},
            )
        try:
            validated = self.input_schema.model_validate(payload)
        except ValidationError as exc:
            return self._error_result(
                started_at,
                ToolStatus.ERROR,
                str(exc),
                attempts=1,
                metadata={"error_type": "validation_error"},
            )

        attempts = 0
        last_error: ToolExecutionError | None = None
        while attempts <= self.max_retries:
            attempts += 1
            try:
                data = self._invoke_with_timeout(validated)
                return self._success_result(started_at, data, attempts)
            except ToolExecutionError as exc:
                last_error = exc
                if not exc.retryable or attempts > self.max_retries:
                    status = ToolStatus.TIMEOUT if isinstance(exc, ToolTimeoutError) else ToolStatus.ERROR
                    return self._error_result(
                        started_at,
                        status,
                        str(exc),
                        attempts=attempts,
                        metadata=exc.metadata,
                    )
            except Exception as exc:
                last_error = ToolExecutionError(str(exc), retryable=False)
                return self._error_result(
                    started_at,
                    ToolStatus.ERROR,
                    str(last_error),
                    attempts=attempts,
                    metadata={"error_type": type(exc).__name__},
                )

            time.sleep(self.retry_backoff_seconds * (2 ** (attempts - 1)))

        return self._error_result(
            started_at,
            ToolStatus.ERROR,
            str(last_error or "工具执行失败"),
            attempts=attempts,
        )

    async def async_invoke(
        self,
        input_data: BaseModel | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        return await asyncio.to_thread(self.invoke, input_data, **kwargs)

    def as_langchain_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self._langchain_invoke,
            name=self.name,
            description=self.description,
            args_schema=self.input_schema,
        )

    def health(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": True,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }

    @abstractmethod
    def _invoke(self, input_data: BaseModel) -> Any:
        """Run the domain-specific operation and return structured or text data."""

    def _langchain_invoke(self, **kwargs: Any) -> str:
        return self.invoke(**kwargs).to_agent_content()

    def _build_payload(
        self,
        input_data: BaseModel | dict[str, Any] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if input_data is None:
            payload: dict[str, Any] = {}
        elif isinstance(input_data, BaseModel):
            payload = input_data.model_dump()
        elif isinstance(input_data, dict):
            payload = dict(input_data)
        else:
            raise TypeError("input_data must be a Pydantic model, dictionary, or None")
        payload.update(kwargs)
        return payload

    def _invoke_with_timeout(self, input_data: BaseModel) -> Any:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tool-{self.name}")
        future = executor.submit(self._invoke, input_data)
        try:
            return future.result(timeout=self.timeout_seconds)
        except FutureTimeoutError as exc:
            future.cancel()
            raise ToolTimeoutError(f"{self.name} 执行超时（>{self.timeout_seconds:g}s）") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _success_result(self, started_at: float, data: Any, attempts: int) -> ToolResult:
        trace = ToolExecutionTrace(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            latency_ms=self._latency_ms(started_at),
            attempts=attempts,
            metadata=self._trace_metadata(),
        )
        self._log_trace(trace)
        return ToolResult(tool_name=self.name, status=ToolStatus.SUCCESS, data=data, trace=trace)

    def _error_result(
        self,
        started_at: float,
        status: ToolStatus,
        error: str,
        *,
        attempts: int,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        trace = ToolExecutionTrace(
            tool_name=self.name,
            status=status,
            latency_ms=self._latency_ms(started_at),
            attempts=attempts,
            error=error,
            metadata={**self._trace_metadata(), **(metadata or {})},
        )
        self._log_trace(trace)
        return ToolResult(tool_name=self.name, status=status, error=error, trace=trace)

    def _trace_metadata(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }

    @staticmethod
    def _latency_ms(started_at: float) -> int:
        return round((time.perf_counter() - started_at) * 1000)

    @staticmethod
    def _log_trace(trace: ToolExecutionTrace) -> None:
        logger.info("tool_execution=%s", trace.model_dump(mode="json"))
