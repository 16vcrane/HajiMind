from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from pydantic import BaseModel

from .base import BaseTool
from .errors import ToolExecutionError

load_dotenv()


class CalendarTimeInput(BaseModel):
    operation: Literal[
        "get_current_datetime",
        "calculate_date_difference",
        "add_days",
        "subtract_days",
        "get_weekday",
        "convert_timezone",
    ]
    timezone: str | None = None
    datetime_value: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    days: int | None = None
    source_timezone: str | None = None
    target_timezone: str | None = None


class CalendarTimeTool(BaseTool):
    name = "calendar_time"
    description = "Get dates and times, calculate date differences, and convert between timezones."
    input_schema = CalendarTimeInput
    timeout_seconds = 5.0
    max_retries = 0
    fallback_timezone = "UTC"

    def _invoke(self, input_data: CalendarTimeInput) -> dict[str, Any]:
        operation = input_data.operation
        if operation == "get_current_datetime":
            timezone_name = input_data.timezone or self._default_timezone()
            current = datetime.now(self._timezone(timezone_name))
            return self._content(
                f"当前时间：{current.isoformat()}",
                datetime=current.isoformat(),
                timezone=timezone_name,
            )

        if operation == "calculate_date_difference":
            start = self._parse_datetime(
                self._required(input_data.start_datetime, "start_datetime"),
                input_data.timezone,
            )
            end = self._parse_datetime(
                self._required(input_data.end_datetime, "end_datetime"),
                input_data.timezone,
            )
            difference = end - start
            return self._content(
                f"日期差：{difference.days} 天，{difference.total_seconds()} 秒",
                difference_days=difference.days,
                difference_seconds=difference.total_seconds(),
            )

        if operation in {"add_days", "subtract_days"}:
            value = self._parse_datetime(
                self._required(input_data.datetime_value, "datetime_value"),
                input_data.timezone,
            )
            days = self._required(input_data.days, "days")
            delta = timedelta(days=days if operation == "add_days" else -days)
            result = value + delta
            return self._content(
                f"计算结果：{result.isoformat()}",
                datetime=result.isoformat(),
                timezone=self._timezone_name(result),
            )

        if operation == "get_weekday":
            value = self._parse_datetime(
                self._required(input_data.datetime_value, "datetime_value"),
                input_data.timezone,
            )
            weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][
                value.weekday()
            ]
            return self._content(
                f"{value.date().isoformat()} 是{weekday}",
                date=value.date().isoformat(),
                weekday=weekday,
            )

        source_timezone = input_data.source_timezone or self._default_timezone()
        target_timezone = self._required(input_data.target_timezone, "target_timezone")
        value = self._parse_datetime(
            self._required(input_data.datetime_value, "datetime_value"),
            source_timezone,
        )
        converted = value.astimezone(self._timezone(target_timezone))
        return self._content(
            f"时区转换结果：{converted.isoformat()}",
            datetime=converted.isoformat(),
            source_timezone=source_timezone,
            target_timezone=target_timezone,
        )

    def _default_timezone(self) -> str:
        return os.getenv("DEFAULT_TIMEZONE", self.fallback_timezone)

    @staticmethod
    def _required(value: Any, field_name: str) -> Any:
        if value is None:
            raise ToolExecutionError(f"{field_name}参数不能为空")
        return value

    def _parse_datetime(self, value: str, timezone_name: str | None) -> datetime:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ToolExecutionError(
                f"datetime格式错误：{value}。请使用 ISO 8601 格式。"
            ) from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=self._timezone(timezone_name or self._default_timezone())
            )
        return parsed

    @staticmethod
    def _timezone_name(value: datetime) -> str:
        return getattr(value.tzinfo, "key", None) or str(value.tzinfo)

    @staticmethod
    def _content(content: str, **data: Any) -> dict[str, Any]:
        return {"content": content, **data}

    @staticmethod
    def _timezone(timezone_name: str) -> ZoneInfo:
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ToolExecutionError(f"不支持或无效的时区：{timezone_name}") from exc
