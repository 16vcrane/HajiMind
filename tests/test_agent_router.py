import sys
import unittest

sys.path.insert(0, "backend")

from agent_router import AgentRouter, RouterDecision, resolve_route_target, route_label


class FakeStructuredRouter:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        last_message = messages[-1]
        user_message = (
            last_message.content
            if hasattr(last_message, "content")
            else last_message["content"]
        )
        for marker, decision in sorted(
            self.decisions.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if marker in user_message:
                return RouterDecision(**decision)
        raise AssertionError(f"No fake decision configured for: {user_message}")


class FakeRouterModel:
    def __init__(self, decisions):
        self.structured_router = FakeStructuredRouter(decisions)

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured_router


class AgentRouterTests(unittest.TestCase):
    def setUp(self):
        self.decisions = {
            "你好": {
                "intent": "simple_qa",
                "complexity": "simple",
                "reason": "通用闲聊，无需外部工具。",
            },
            "知识库": {
                "intent": "knowledge_qa",
                "complexity": "medium",
                "requires_rag": True,
                "reason": "用户明确要求查询知识库。",
            },
            "最新": {
                "intent": "web_research",
                "complexity": "medium",
                "requires_web": True,
                "reason": "需要最新网络资料。",
            },
            "天气": {
                "intent": "weather",
                "complexity": "simple",
                "requires_weather": True,
                "reason": "用户询问天气。",
            },
            "星期": {
                "intent": "calendar",
                "complexity": "simple",
                "requires_calendar": True,
                "reason": "用户询问日期星期。",
            },
            "学习计划": {
                "intent": "multi_step",
                "complexity": "complex",
                "requires_planning": True,
                "reason": "需要制定多步骤计划。",
            },
            "知识库和最新网络资料": {
                "intent": "mixed",
                "complexity": "complex",
                "requires_rag": True,
                "requires_web": True,
                "requires_planning": True,
                "reason": "同时需要知识库、网络搜索和计划。",
            },
            "天气和时区": {
                "intent": "mixed",
                "complexity": "medium",
                "requires_weather": True,
                "requires_calendar": True,
                "reason": "同时需要天气和时区能力。",
            },
            "网页资料和日期": {
                "intent": "mixed",
                "complexity": "medium",
                "requires_web": True,
                "requires_calendar": True,
                "reason": "同时需要网络资料和日期计算。",
            },
            "文档排期": {
                "intent": "mixed",
                "complexity": "complex",
                "requires_rag": True,
                "requires_calendar": True,
                "requires_planning": True,
                "reason": "需要读取文档并制定排期。",
            },
        }

    def test_router_uses_structured_llm_output_for_ten_question_types(self):
        model = FakeRouterModel(self.decisions)
        router = AgentRouter(model)

        cases = [
            ("你好，介绍一下你自己", "simple_qa", "llm"),
            ("请查一下我的知识库里 RAG 怎么写", "knowledge_qa", "rag"),
            ("帮我搜索最新 RAG 框架资料", "web_research", "baidu"),
            ("上海今天的天气怎么样", "weather", "life_service"),
            ("2026-08-11 是星期几", "calendar", "calendar"),
            ("帮我做一个学习计划", "multi_step", "planner"),
            (
                "帮我根据我的知识库和最新网络资料总结 RAG，然后制定学习计划。",
                "mixed",
                "planner",
            ),
            ("查天气和时区转换", "mixed", "planner"),
            ("整理网页资料和日期节点", "mixed", "planner"),
            ("根据文档排期完成计划", "mixed", "planner"),
        ]

        for question, expected_intent, expected_route in cases:
            with self.subTest(question=question):
                decision = router.decide(question)
                self.assertEqual(decision.intent, expected_intent)
                self.assertEqual(resolve_route_target(decision), expected_route)

        self.assertIs(model.schema, RouterDecision)
        self.assertEqual(len(model.structured_router.calls), 10)

    def test_mixed_task_keeps_multiple_requirements_instead_of_single_tool(self):
        decision = RouterDecision(
            intent="mixed",
            complexity="complex",
            requires_rag=True,
            requires_web=True,
            requires_planning=True,
            reason="需要知识库、网络资料和计划。",
        )

        self.assertTrue(decision.requires_rag)
        self.assertTrue(decision.requires_web)
        self.assertTrue(decision.requires_planning)
        self.assertEqual(resolve_route_target(decision), "planner")
        self.assertEqual(route_label("planner"), "Planner")

    def test_complex_or_multi_requirement_routes_to_planner(self):
        complex_decision = RouterDecision(
            intent="knowledge_qa",
            complexity="complex",
            requires_rag=True,
            reason="复杂知识任务。",
        )
        multi_tool_decision = RouterDecision(
            intent="weather",
            complexity="medium",
            requires_weather=True,
            requires_calendar=True,
            reason="天气和日期都需要。",
        )

        self.assertEqual(resolve_route_target(complex_decision), "planner")
        self.assertEqual(resolve_route_target(multi_tool_decision), "planner")


if __name__ == "__main__":
    unittest.main()
