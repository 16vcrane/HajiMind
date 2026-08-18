from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from .base import BaseTool
from .errors import ToolExecutionError

load_dotenv()


class WeatherInput(BaseModel):
    location: str
    extensions: str = "base"


class WeatherTool(BaseTool):
    name = "get_current_weather"
    description = "获取指定城市的实时天气或天气预报。"
    input_schema = WeatherInput
    timeout_seconds = 10.0
    max_retries = 1

    def health(self) -> dict[str, Any]:
        health = super().health()
        health["available"] = bool(
            os.getenv("AMAP_WEATHER_API") and os.getenv("AMAP_API_KEY")
        )
        return health

    def _invoke(self, input_data: WeatherInput) -> str:
        location = input_data.location.strip()
        extensions = input_data.extensions
        if not location:
            raise ToolExecutionError("location参数不能为空")
        if extensions not in ("base", "all"):
            raise ToolExecutionError("extensions参数错误，请输入base或all")

        endpoint = os.getenv("AMAP_WEATHER_API")
        api_key = os.getenv("AMAP_API_KEY")
        if not endpoint or not api_key:
            raise ToolExecutionError("天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）")

        try:
            response = requests.get(
                endpoint,
                params={
                    "key": api_key,
                    "city": location,
                    "extensions": extensions,
                    "output": "json",
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout as exc:
            raise ToolExecutionError("错误：请求天气服务超时", retryable=True) from exc
        except requests.exceptions.RequestException as exc:
            raise ToolExecutionError(f"错误：天气服务请求失败 - {exc}", retryable=True) from exc
        except ValueError as exc:
            raise ToolExecutionError(f"错误：解析天气数据失败 - {exc}") from exc

        if data.get("status") != "1":
            raise ToolExecutionError(f"查询失败：{data.get('info', '未知错误')}")

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                raise ToolExecutionError(f"未查询到 {location} 的天气数据")
            weather = lives[0]
            return (
                f"【{weather.get('city', location)} 实时天气】\n"
                f"天气状况：{weather.get('weather', '未知')}\n"
                f"温度：{weather.get('temperature', '未知')}℃\n"
                f"湿度：{weather.get('humidity', '未知')}%\n"
                f"风向：{weather.get('winddirection', '未知')}\n"
                f"风力：{weather.get('windpower', '未知')}级\n"
                f"更新时间：{weather.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            raise ToolExecutionError(f"未查询到 {location} 的天气预报数据")
        forecast = forecasts[0]
        today = (forecast.get("casts") or [])[0] if forecast.get("casts") else {}
        return "\n".join(
            [
                f"【{forecast.get('city', location)} 天气预报】",
                f"更新时间：{forecast.get('reporttime', '未知')}",
                "",
                "今日天气：",
                f"  白天：{today.get('dayweather', '未知')}",
                f"  夜间：{today.get('nightweather', '未知')}",
                f"  气温：{today.get('nighttemp', '未知')}~{today.get('daytemp', '未知')}℃",
            ]
        )
