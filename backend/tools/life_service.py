from __future__ import annotations

import os
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from .base import BaseTool
from .errors import ToolExecutionError
from .weather import WeatherInput, WeatherTool

load_dotenv()


class LifeServiceInput(BaseModel):
    operation: Literal["current_weather", "forecast"]
    location: str


class LifeServiceTool(BaseTool):
    """Life-service entry point that currently delegates weather to the legacy tool."""

    name = "life_service"
    description = "Get current weather or weather forecasts. POI, traffic, and dining can be added as future operations."
    input_schema = LifeServiceInput
    timeout_seconds = 10.0
    max_retries = 1

    def health(self) -> dict[str, Any]:
        health = super().health()
        health["available"] = bool(
            os.getenv("AMAP_WEATHER_API") and os.getenv("AMAP_API_KEY")
        )
        health["supported_operations"] = ["current_weather", "forecast"]
        return health

    def _invoke(self, input_data: LifeServiceInput) -> dict[str, Any]:
        location = input_data.location.strip()
        if not location:
            raise ToolExecutionError("location参数不能为空")

        extensions = "base" if input_data.operation == "current_weather" else "all"
        weather_tool = WeatherTool()
        weather_tool.timeout_seconds = self.timeout_seconds
        content = weather_tool._invoke(WeatherInput(location=location, extensions=extensions))
        return {
            "operation": input_data.operation,
            "location": location,
            "content": content,
        }
