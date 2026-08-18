from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field


Intent = Literal[
    "simple_qa",
    "knowledge_qa",
    "web_research",
    "weather",
    "calendar",
    "multi_step",
    "mixed",
]
Complexity = Literal["simple", "medium", "complex"]
RouteTarget = Literal[
    "llm",
    "rag",
    "baidu",
    "life_service",
    "calendar",
    "planner",
    "fallback",
]


class RouterDecision(BaseModel):
    intent: Intent
    complexity: Complexity
    requires_rag: bool = False
    requires_web: bool = False
    requires_weather: bool = False
    requires_calendar: bool = False
    requires_planning: bool = False
    reason: str = Field(default="", max_length=500)


ROUTER_SYSTEM_PROMPT = """你是 HajiMind 的任务路由器。
你的唯一职责是根据用户问题输出结构化 RouterDecision，不回答用户问题。

可选 intent：
- simple_qa：通用常识、闲聊、无需外部工具的问题。
- knowledge_qa：需要查询用户上传文档、知识库、项目资料或历史文档的问题。
- web_research：需要最新网络资料、新闻、实时外部信息或互联网搜索的问题。
- weather：天气、气温、预报、生活天气服务。
- calendar：日期、时间、星期、时区、日期计算。
- multi_step：需要拆解、计划、排期、综合多步执行的问题。
- mixed：同一问题同时需要多种能力，例如知识库 + 网络 + 计划。

判断规则：
- 不要只选一个能力；混合任务必须把所有需要的 requires_* 置为 true。
- 涉及“我的知识库、文档、上传资料、项目资料、README、代码审计资料”时 requires_rag=true。
- 涉及“最新、网络、网页、搜索、新闻、当前外部资料”时 requires_web=true。
- 涉及天气或天气预报时 requires_weather=true。
- 涉及日期、时间、星期、时区、日程日期计算时 requires_calendar=true。
- 涉及制定计划、拆解步骤、学习计划、多步骤研究或多个工具组合时 requires_planning=true。
- simple_qa 通常所有 requires_* 都为 false。
- 如果多个 requires_* 为 true，intent 应优先为 mixed，complexity 至少为 medium。
- 如果需要规划或复杂拆解，intent 可为 multi_step 或 mixed。
"""


class AgentRouter:
    def __init__(self, model):
        self.model = model

    def decide(
        self,
        user_text: str,
        history: list[BaseMessage] | None = None,
    ) -> RouterDecision:
        recent_history = history[-6:] if history else []
        prompt = (
            "请分析最后一条用户问题并输出 RouterDecision。\n"
            f"用户问题：{user_text}"
        )
        structured_model = self.model.with_structured_output(RouterDecision)
        return structured_model.invoke(
            [SystemMessage(content=ROUTER_SYSTEM_PROMPT)]
            + recent_history
            + [HumanMessage(content=prompt)]
        )


def resolve_route_target(decision: RouterDecision) -> RouteTarget:
    required_count = sum(
        [
            decision.requires_rag,
            decision.requires_web,
            decision.requires_weather,
            decision.requires_calendar,
        ]
    )
    if (
        decision.requires_planning
        or decision.intent in ("multi_step", "mixed")
        or decision.complexity == "complex"
        or required_count > 1
    ):
        return "planner"
    if decision.requires_rag or decision.intent == "knowledge_qa":
        return "rag"
    if decision.requires_web or decision.intent == "web_research":
        return "baidu"
    if decision.requires_weather or decision.intent == "weather":
        return "life_service"
    if decision.requires_calendar or decision.intent == "calendar":
        return "calendar"
    return "llm"


def route_label(route_target: RouteTarget) -> str:
    return {
        "llm": "LLM",
        "rag": "RAG",
        "baidu": "Baidu",
        "life_service": "LifeService",
        "calendar": "Calendar",
        "planner": "Planner",
        "fallback": "Fallback Agent",
    }[route_target]
