# HajiMind AI Agent 深度升级总控提示词

你现在是本项目的 Principal AI Engineer + Backend Architect。

你需要基于当前 HajiMind 项目进行一次系统性的 AI Agent 工程升级。

## 一、项目背景

HajiMind 是一个基于：

- Python 3.12+
- FastAPI
- LangChain / LangGraph Agent
- Milvus
- PostgreSQL
- Redis
- Vue 3
- SSE Streaming

构建的可观测、流式输出、Hybrid RAG 智能体。

当前已经具备：

1. Dense + BM25 Sparse Hybrid Search
2. Milvus RRF
3. Jina Rerank
4. 三级 Chunk
5. Parent Document / Auto-merging
6. Leaf-only Vector Storage
7. Query Rewrite
8. Step-back
9. HyDE
10. Document Grading
11. Streaming Agent
12. SSE
13. 实时 RAG Trace
14. Agent Thinking State
15. Abort / Cancel
16. Conversation Memory
17. Conversation Summary
18. PostgreSQL
19. Redis
20. JWT Authentication
21. Admin 权限
22. Document Management
23. Weather Tool

这些能力必须尽可能保留，不允许为了重构而破坏已有功能。

项目当前 Todo 中还包括：

- 文档结构解析
- 特殊代码/表格/图片处理
- BM25 参数扫描
- Hybrid Retrieval AB Test
- RAG Evaluation
- Rerank Evaluation
- 子问题分解
- Multi-document Refine
- 文档冲突检测
- Multi-modal Embedding
- SQL Assistant
- Network Search
- Multi-step Planning
- Router
- Memory Optimization
- Multi-Agent
- Loop Detection / Recovery

本次升级需要围绕这些方向进行系统实现。

---

# 二、最终目标

将 HajiMind 从：

「具备 RAG 能力的聊天 Agent」

升级为：

「具备 RAG + 外部工具调用 + 任务规划 + 多步骤执行 + 实时信息获取 + Memory + 可观测性 + Evaluation 的通用 AI Agent」

最终希望形成下面的能力：

User
 ↓
Agent Router
 ↓
Task Understanding
 ↓
Planner
 ↓
Tool Selection
 ↓
┌──────────────────────────────┐
│ Knowledge Tool               │
│ 百度搜索 Tool                │
│ 天气 / 生活服务 Tool         │
│ 日历 / 时间日期 Tool         │
└──────────────────────────────┘
 ↓
RAG / Web / API / Calendar
 ↓
Result Fusion
 ↓
Verification
 ↓
Final Answer
 ↓
Trace / Evaluation / Memory

---

# 三、三个核心外部 Tool

必须实现统一 Tool 抽象。

## Tool 1：百度搜索 API

能力：

- 搜索实时互联网信息
- 获取搜索结果标题
- URL
- 摘要
- 来源
- 时间
- 排名

Agent 可以根据问题判断是否需要调用。

例如：

用户：

“2026 年最新的 LangGraph 有哪些变化？”

Agent：

Router
→ 判断为实时信息
→ BaiduSearchTool
→ 获取搜索结果
→ 整理来源
→ 回答

必须支持：

- query
- top_k
- timeout
- retry
- error fallback

---

# 四、Tool 2：天气 / 生活服务 API

目前项目已经存在天气 Tool。

需要将其升级成统一 LifeServiceTool。

至少支持：

- 当前天气
- 天气预报
- 温度
- 湿度
- 风力
- 空气质量（如果 API 支持）
- 生活指数（如果 API 支持）

未来可以扩展：

- POI
- 路况
- 出行
- 餐饮
- 酒店

但是本阶段不要过度实现。

Tool 必须设计成可扩展接口。

---

# 五、Tool 3：日历 / 时间日期 API

新增 CalendarTimeTool。

至少支持：

1. 当前时间
2. 当前日期
3. 时间差计算
4. 日期加减
5. 星期计算
6. 时间区间判断
7. 时区转换
8. 相对时间解析

例如：

“距离国庆还有多久？”

→ CalendarTimeTool

“下周三是几号？”

→ CalendarTimeTool

“新加坡下午 3 点对应北京时间几点？”

→ CalendarTimeTool

---

# 六、不要把三个 API 简单写成三个函数

必须建立统一 Tool Architecture。

建议：

backend/
    tools/
        __init__.py
        base.py
        registry.py
        knowledge_tool.py
        baidu_search_tool.py
        life_service_tool.py
        calendar_tool.py

统一定义 Tool：

BaseTool

包含：

- name
- description
- input_schema
- invoke()
- async_invoke()
- timeout
- retry
- error handling
- metadata

所有工具都必须通过 Tool Registry 注册。

例如：

ToolRegistry

负责：

- 注册工具
- 查询工具
- Tool metadata
- Tool availability
- Tool health
- Tool execution trace

Agent 不应该直接 import 某个具体 API。

---

# 七、增加 Agent Router

需要新增 Router Node。

Router 不应该简单根据关键词判断。

需要让 LLM 输出结构化结果：

{
    "intent": "...",
    "requires_rag": true,
    "requires_web": false,
    "requires_weather": false,
    "requires_calendar": false,
    "requires_planning": false,
    "complexity": "simple|medium|complex"
}

根据 Router 决定后续流程。

例如：

“我上传的论文中 Transformer 的核心思想是什么？”

→ RAG

“今天上海天气怎么样？”

→ Weather

“帮我搜索一下最新的 LangGraph 教程”

→ Baidu Search

“距离毕业还有多少天？”

→ Calendar

“帮我制定一个学习 LangGraph 的计划，并结合我上传的资料和网上最新资料”

→ Planner
→ RAG
→ Baidu
→ Calendar
→ Final Synthesis

---

# 八、增加 Planner

这是本次升级最重要的功能之一。

不要让 Agent 所有任务都直接进入工具。

增加：

Planner Node

负责把复杂任务拆成：

Task List

例如：

用户：

“帮我制定未来 30 天的 LangGraph 学习计划，并结合我的资料和最新网络资料。”

Planner 输出：

1. 获取当前日期
2. 搜索最新 LangGraph 官方资料
3. 检索用户知识库
4. 分析学习内容
5. 生成 30 天计划
6. 检查计划时间合理性
7. 输出最终计划

每个任务需要包含：

- task_id
- description
- dependencies
- required_tool
- status
- result

---

# 九、支持多步骤任务执行

Planner 产生 DAG：

Task A
 ↓
Task B ──→ Task D
 ↓
Task C ──→ Task D

对于没有依赖关系的任务，可以并行执行。

例如：

Task A：百度搜索
Task B：RAG 检索
Task C：获取当前日期

三个任务可以并行。

然后：

Task D：结果融合

这样才能体现真正的 Agent Workflow，而不是简单 Chatbot。

---

# 十、增加 Tool Result Fusion

不同工具返回的结果必须统一。

建立：

ToolResult

例如：

{
    "tool_name": "baidu_search",
    "success": true,
    "data": ...,
    "sources": ...,
    "latency_ms": 1234,
    "error": null
}

所有 Tool 必须统一输出。

最终 Agent 可以：

RAGResult
+
WebResult
+
WeatherResult
+
CalendarResult

统一进入 Answer Synthesis。

---

# 十一、增加 Agent Trace

目前项目已经有 RAG Trace。

不要推翻。

将其升级成：

Agent Trace

例如：

{
    "request_id": "...",
    "session_id": "...",
    "intent": "...",
    "route": "...",
    "plan": [...],
    "tool_calls": [...],
    "rag_trace": {...},
    "latency": {...},
    "errors": [...],
    "final_answer": "..."
}

前端可以展示：

用户问题

↓
Intent Classification

↓
Planner

↓
Tool Selection

↓
🔎 百度搜索

↓
📚 知识库检索

↓
🌤 天气 API

↓
📅 日期计算

↓
🧠 Answer Synthesis

---

# 十二、增加复杂度路由

不是所有问题都需要复杂 Agent。

建立：

Simple / Medium / Complex

例如：

“1+1 等于多少？”

→ Simple

“解释一下 RAG”

→ Medium

“分析我上传的论文，并结合最新网络资料评价它的 RAG 架构，然后提出改进方案”

→ Complex

Simple：

直接 LLM

Medium：

Router → RAG / Tool

Complex：

Router → Planner → Multi-step Execution

这样可以降低：

- latency
- token cost
- tool call 数量

---

# 十三、增加死循环检测

实现：

_is_stuck

以及：

attempt_loop_recovery

例如：

Agent：

search
→ search
→ search
→ search

检测异常。

达到阈值：

停止循环

并进行：

Recovery Strategy

例如：

- 换 query
- 换工具
- 跳过该步骤
- 请求用户补充信息
- 使用已有结果

---

# 十四、增加 Tool Failure Recovery

任何外部 API 都可能：

- timeout
- 429
- 500
- connection error
- invalid response

必须：

retry

并支持：

exponential backoff

例如：

1s
2s
4s

超过最大次数：

fallback。

例如：

百度搜索失败：

→ 告知 Agent

Agent 可以：

→ 使用 RAG

或者：

→ 基于已有信息回答

绝不能让一个 Tool 挂掉整个 Agent。

---

# 十五、RAG 深度升级

按照当前 Todo 实现：

1. 文档结构解析
2. Recursive Chunk fallback
3. Semantic Chunk
4. Code Block 特殊处理
5. Table 特殊处理
6. Image metadata
7. BM25 k1/b 参数
8. Hybrid 权重 AB Test
9. Gold Chunk Dataset
10. RAG Evaluation
11. Rerank Evaluation

注意：

这些应该拆成独立阶段。

不要一次全部修改。

---

# 十六、RAG Evaluation

建立 eval/：

eval/
    datasets/
    retrieval/
    generation/
    metrics/
    runner.py

至少评估：

Retrieval：

- Recall@K
- Precision@K
- MRR
- Hit Rate
- NDCG

Generation：

- Faithfulness
- Answer Relevance
- Context Relevance

比较：

Dense

BM25

Hybrid

Hybrid + Rerank

最终输出实验报告。

---

# 十七、增加多文档冲突检测

例如：

Document A：

Python 3.12 支持 X

Document B：

Python 3.12 不支持 X

Agent 不应该选择一个直接回答。

应该：

检测冲突

然后：

“知识库中存在来源冲突。”

输出：

来源 A
来源 B
差异
可能原因

这会成为项目很好的 Agent/RAG 亮点。

---

# 十八、Memory 升级

将 Memory 分为：

1. Working Memory
2. Conversation Memory
3. Long-term Memory
4. User Preference Memory

但不要直接存储所有信息。

需要：

Memory Extraction

判断：

“这个信息是否值得长期记忆？”

再写入长期 Memory。

---

# 十九、Multi-Agent

不要一开始就把所有东西拆成大量 Agent。

最终可以设计：

Supervisor Agent

下面：

Research Agent
RAG Agent
Life Agent
Calendar Agent
SQL Agent

Supervisor：

负责：

- 任务理解
- Agent routing
- Plan
- Result synthesis

但是只有当 Tool 数量和任务复杂度确实增加后才引入。

---

# 二十、SQL Assistant

作为后续独立 Skill：

SQL Agent

支持：

自然语言：

“查询过去 7 天用户活跃情况”

→ SQL

→ SQL Validation

→ Read-only Execution

→ Result

→ Explanation

必须禁止：

DROP

DELETE

UPDATE

INSERT

ALTER

TRUNCATE

除非明确进入管理员模式。

---

# 二十一、前端升级

当前 Vue CDN 页面保留。

新增：

Agent Workflow UI

例如：

┌──────────────────────────────┐
│ 🧠 Agent Workflow            │
│                              │
│ ✓ Analyze Question           │
│ ✓ Create Plan                │
│ ✓ Search Web                 │
│ ✓ Search Knowledge Base      │
│ ✓ Query Weather              │
│ ✓ Calculate Date             │
│ ● Synthesizing Answer        │
│                              │
└──────────────────────────────┘

Tool Call：

🔎 Baidu Search
status: success
latency: 824ms
results: 8

RAG：

📚 Hybrid Search
Dense: 12
BM25: 12
Rerank: 8

---

# 二十二、Git 强制要求

这是非常重要的。

你不能一次性修改整个项目。

严格按照：

Phase 0
Phase 1
Phase 2
...

执行。

每完成一个 Phase：

1. 运行测试
2. 检查 git diff
3. 检查是否产生意外修改
4. 更新 README / CHANGELOG
5. 创建 commit

Commit 格式：

feat(agent): add intelligent router

feat(tools): add baidu search tool

feat(tools): add calendar tool

feat(planner): add multi-step planner

feat(agent): add tool recovery

test(rag): add retrieval evaluation

refactor(agent): introduce tool registry

---

# 二十三、强制 Git 安全规则

开始任何修改前：

执行：

git status
git branch --show-current
git log --oneline -10

确认当前分支。

禁止：

git reset --hard

禁止：

git clean -fd

禁止：

删除用户已有代码

禁止：

覆盖未提交修改

禁止：

强制 push

禁止：

修改 main/master

除非我明确允许。

建议：

feature/hajimind-agent-upgrade

每个阶段独立 commit。

---

# 二十四、编码原则

1. 不做无意义重构
2. 优先复用已有代码
3. 不重复实现已有 RAG
4. 不破坏 Streaming
5. 不破坏 SSE
6. 不破坏 Abort
7. 不破坏 JWT
8. 不破坏 PostgreSQL
9. 不破坏 Redis
10. 不破坏 Milvus
11. Tool 必须可测试
12. Tool 必须有 timeout
13. Tool 必须有 retry
14. Tool 必须有错误处理
15. Tool 必须有日志
16. Tool 必须有 trace
17. API Key 必须通过环境变量
18. 禁止把 API Key 写入代码
19. 不允许因为外部 API 失败导致 Agent 崩溃
20. 所有新增功能尽量保持 backward compatible

---

# 二十五、工作方式

你现在不要立即修改代码。

第一步：

扫描整个项目。

重点查看：

backend/
frontend/
tests/
pyproject.toml
docker-compose.yml
.env.example
README.md

然后分析：

1. 当前 Agent 架构
2. 当前 LangGraph / LangChain 使用方式
3. 当前 Tool 架构
4. 当前 RAG pipeline
5. 当前 API
6. 当前数据库模型
7. 当前 Redis 使用
8. 当前 Streaming
9. 当前 Trace
10. 当前 Todo 已完成情况

然后给我输出：

## A. 当前功能完成度

用：

DONE
PARTIAL
TODO

标记。

## B. 当前架构图

## C. 推荐升级架构

## D. Todo 优先级

P0 / P1 / P2

## E. 分阶段实施计划

每个 Phase 写：

- 目标
- 修改文件
- 新增文件
- 风险
- 测试方法
- Git Commit

注意：

此阶段只分析。

不要修改任何代码。

等我确认后，再进入 Phase 0。