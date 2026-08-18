import sys
import time
import unittest
from unittest.mock import patch

import requests

sys.path.insert(0, "backend")

from tools import (
    BaiduSearchTool,
    CalendarTimeTool,
    LifeServiceTool,
    ToolExecutionError,
    ToolStatus,
    get_agent_tools,
    tool_registry,
)


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


def valid_weather_response():
    return {
        "status": "1",
        "lives": [
            {
                "city": "上海",
                "weather": "晴",
                "temperature": "28",
                "humidity": "40",
                "winddirection": "东",
                "windpower": "3",
                "reporttime": "2026-08-11 10:00:00",
            }
        ],
    }


def valid_forecast_response():
    return {
        "status": "1",
        "forecasts": [
            {
                "city": "上海",
                "reporttime": "2026-08-11 10:00:00",
                "casts": [
                    {
                        "dayweather": "晴",
                        "nightweather": "多云",
                        "daytemp": "30",
                        "nighttemp": "24",
                    }
                ],
            }
        ],
    }


class BaiduSearchToolTests(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "BAIDU_SEARCH_API_KEY": "test-key",
            "BAIDU_SEARCH_API_URL": "https://example.test/baidu-search",
        }

    def test_normal_call_returns_normalized_ranked_results_and_trace(self):
        def provider(query, top_k, api_url, api_key, timeout_seconds):
            self.assertEqual(query, "LangGraph")
            self.assertEqual(top_k, 1)
            self.assertEqual(api_url, self.environment["BAIDU_SEARCH_API_URL"])
            self.assertEqual(api_key, self.environment["BAIDU_SEARCH_API_KEY"])
            return [
                {
                    "title": "LangGraph",
                    "url": "https://example.test/langgraph",
                    "snippet": "Agent orchestration",
                    "source": "Baidu",
                    "published_at": "2026-08-10",
                }
            ]

        with patch.dict("os.environ", self.environment, clear=False):
            result = BaiduSearchTool(provider=provider).invoke(query="LangGraph", top_k=1)

        self.assertTrue(result.success)
        self.assertEqual(result.data["results"][0]["rank"], 1)
        self.assertEqual(result.data["results"][0]["published_at"], "2026-08-10")
        self.assertEqual(result.trace.tool_name, "baidu_search")

    def test_invalid_parameters_return_validation_or_tool_errors(self):
        with patch.dict("os.environ", self.environment, clear=False):
            empty_query = BaiduSearchTool(provider=lambda *_: []).invoke(query="")
            invalid_top_k = BaiduSearchTool(provider=lambda *_: []).invoke(
                query="test", top_k=0
            )

        self.assertFalse(empty_query.success)
        self.assertFalse(invalid_top_k.success)
        self.assertEqual(invalid_top_k.trace.metadata["error_type"], "validation_error")

    def test_timeout_is_reported_in_trace(self):
        def slow_provider(*_args):
            time.sleep(0.05)
            return []

        with patch.dict("os.environ", self.environment, clear=False):
            result = BaiduSearchTool(
                provider=slow_provider, timeout_seconds=0.01, max_retries=0
            ).invoke(query="timeout")

        self.assertEqual(result.status, ToolStatus.TIMEOUT)
        self.assertEqual(result.trace.status, ToolStatus.TIMEOUT)

    def test_provider_api_error_is_returned_without_retry(self):
        def failing_provider(*_args):
            raise ToolExecutionError("Baidu API returned 500")

        with patch.dict("os.environ", self.environment, clear=False):
            result = BaiduSearchTool(provider=failing_provider).invoke(query="error")

        self.assertFalse(result.success)
        self.assertIn("Baidu API returned 500", result.error)
        self.assertEqual(result.trace.attempts, 1)

    def test_retry_reinvokes_retryable_provider_failure(self):
        calls = 0

        def flaky_provider(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ToolExecutionError("temporary provider error", retryable=True)
            return [{"title": "Recovered", "url": "https://example.test/recovered"}]

        with patch.dict("os.environ", self.environment, clear=False):
            result = BaiduSearchTool(
                provider=flaky_provider, max_retries=1
            ).invoke(query="retry")

        self.assertTrue(result.success)
        self.assertEqual(calls, 2)
        self.assertEqual(result.trace.attempts, 2)


class LifeServiceToolTests(unittest.TestCase):
    environment = {
        "AMAP_WEATHER_API": "https://example.test/weather",
        "AMAP_API_KEY": "test-key",
    }

    def test_current_weather_and_forecast_reuse_amap_weather_contract(self):
        with (
            patch.dict("os.environ", self.environment, clear=False),
            patch(
                "tools.weather.requests.get",
                side_effect=[
                    FakeResponse(valid_weather_response()),
                    FakeResponse(valid_forecast_response()),
                ],
            ) as get,
        ):
            tool = LifeServiceTool()
            current = tool.invoke(operation="current_weather", location="上海")
            forecast = tool.invoke(operation="forecast", location="上海")

        self.assertTrue(current.success)
        self.assertIn("实时天气", current.to_agent_content())
        self.assertTrue(forecast.success)
        self.assertIn("天气预报", forecast.to_agent_content())
        self.assertEqual(get.call_args_list[0].kwargs["params"]["extensions"], "base")
        self.assertEqual(get.call_args_list[1].kwargs["params"]["extensions"], "all")

    def test_invalid_parameters_return_validation_or_tool_errors(self):
        result = LifeServiceTool().invoke(operation="unsupported", location="上海")

        self.assertFalse(result.success)
        self.assertEqual(result.trace.metadata["error_type"], "validation_error")

    def test_timeout_retries_using_base_tool_infrastructure(self):
        with (
            patch.dict("os.environ", self.environment, clear=False),
            patch(
                "tools.weather.requests.get",
                side_effect=[
                    requests.exceptions.Timeout(),
                    FakeResponse(valid_weather_response()),
                ],
            ),
        ):
            result = LifeServiceTool().invoke(
                operation="current_weather", location="上海"
            )

        self.assertTrue(result.success)
        self.assertEqual(result.trace.attempts, 2)

    def test_amap_api_error_is_returned_in_trace(self):
        with (
            patch.dict("os.environ", self.environment, clear=False),
            patch(
                "tools.weather.requests.get",
                return_value=FakeResponse({"status": "0", "info": "INVALID_USER_KEY"}),
            ),
        ):
            result = LifeServiceTool().invoke(
                operation="current_weather", location="上海"
            )

        self.assertFalse(result.success)
        self.assertIn("INVALID_USER_KEY", result.error)


class SlowCalendarTimeTool(CalendarTimeTool):
    timeout_seconds = 0.01

    def _invoke(self, input_data):
        time.sleep(0.05)
        return super()._invoke(input_data)


class FlakyCalendarTimeTool(CalendarTimeTool):
    max_retries = 1
    retry_backoff_seconds = 0

    def __init__(self):
        self.calls = 0

    def _invoke(self, input_data):
        self.calls += 1
        if self.calls == 1:
            raise ToolExecutionError("temporary calendar service error", retryable=True)
        return super()._invoke(input_data)


class CalendarTimeToolTests(unittest.TestCase):
    def test_supported_datetime_operations_and_timezones(self):
        tool = CalendarTimeTool()

        current = tool.invoke(operation="get_current_datetime", timezone="Asia/Shanghai")
        difference = tool.invoke(
            operation="calculate_date_difference",
            start_datetime="2026-08-10T00:00:00",
            end_datetime="2026-08-12T12:00:00",
            timezone="UTC",
        )
        added = tool.invoke(
            operation="add_days",
            datetime_value="2026-08-11T00:00:00",
            days=2,
            timezone="Asia/Singapore",
        )
        subtracted = tool.invoke(
            operation="subtract_days",
            datetime_value="2026-08-11T00:00:00",
            days=1,
            timezone="UTC",
        )
        weekday = tool.invoke(
            operation="get_weekday",
            datetime_value="2026-08-10T00:00:00",
            timezone="UTC",
        )
        converted = tool.invoke(
            operation="convert_timezone",
            datetime_value="2026-08-11T08:00:00",
            source_timezone="Asia/Shanghai",
            target_timezone="UTC",
        )

        self.assertTrue(current.success)
        self.assertEqual(current.data["timezone"], "Asia/Shanghai")
        self.assertEqual(difference.data["difference_days"], 2)
        self.assertEqual(added.data["datetime"], "2026-08-13T00:00:00+08:00")
        self.assertEqual(subtracted.data["datetime"], "2026-08-10T00:00:00+00:00")
        self.assertEqual(weekday.data["weekday"], "星期一")
        self.assertEqual(converted.data["datetime"], "2026-08-11T00:00:00+00:00")

    def test_invalid_parameters_return_validation_error(self):
        result = CalendarTimeTool().invoke(operation="not-supported")

        self.assertFalse(result.success)
        self.assertEqual(result.trace.metadata["error_type"], "validation_error")

    def test_invalid_timezone_is_returned_as_service_error(self):
        result = CalendarTimeTool().invoke(
            operation="get_current_datetime", timezone="Mars/Olympus"
        )

        self.assertFalse(result.success)
        self.assertIn("无效的时区", result.error)

    def test_timeout_is_reported_in_trace(self):
        result = SlowCalendarTimeTool().invoke(
            operation="get_current_datetime", timezone="UTC"
        )

        self.assertEqual(result.status, ToolStatus.TIMEOUT)

    def test_retry_reinvokes_retryable_failure(self):
        tool = FlakyCalendarTimeTool()
        result = tool.invoke(operation="get_current_datetime", timezone="UTC")

        self.assertTrue(result.success)
        self.assertEqual(tool.calls, 2)
        self.assertEqual(result.trace.attempts, 2)


class ExternalToolRegistryTests(unittest.TestCase):
    def test_external_tools_are_registered_without_changing_agent_tool_list(self):
        registered = {tool.name for tool in tool_registry.list()}
        agent_tools = {tool.name for tool in get_agent_tools()}

        self.assertTrue({"baidu_search", "life_service", "calendar_time"} <= registered)
        self.assertEqual(
            agent_tools, {"get_current_weather", "search_knowledge_base"}
        )


if __name__ == "__main__":
    unittest.main()
