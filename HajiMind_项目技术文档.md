# HajiMind 项目技术文档

> HajiMind（谐音「哈基米」）是一个基于 FastAPI、LangChain、LangGraph、Milvus、PostgreSQL 和 Redis 构建的流式、可观测、多用户 RAG Agent 系统。

## 一、项目定位

HajiMind 不是单一的聊天机器人，而是一个具备任务路由、多步骤规划、工具执行、知识库检索、持久化记忆、权限隔离和全链路观测能力的 AI Agent 服务。

当前版本的核心升级包括：

- Agent Router：把用户问题路由到 LLM、RAG、Web Search、天气、日历或 Planner。
- Multi-step Planner：对复杂任务生成结构化 Plan，并按依赖图串行或并行执行任务。
- 统一 Tool Infrastructure：所有工具继承 `BaseTool`，统一输入校验、超时、重试、错误返回、Trace 和 Circuit Breaker。
- Agent Trace：把 Router、Planner、Tool、Task、RAG、Memory、latency、error 和最终状态统一记录。
- Agent Reliability：增加重试、指数退避、最大步数、死循环检测、任务恢复和熔断。
- RAG Deep Engineering：结构化文档解析、语义分块、BM25 参数化、Hybrid 权重、离线评估。
- 多文档冲突检测：保留冲突来源、观点、证据和时间，不自动裁决。
- 结构化 Memory：支持 Working、Conversation、Long-term 和 Preference Memory，并按用户隔离。

## 二、技术栈

- 后端：FastAPI、Uvicorn、Pydantic、SQLAlchemy。
- Agent：LangChain Agent、LangGraph、结构化 LLM Output。
- 检索：Milvus Dense Vector、SPARSE_FLOAT_VECTOR、RRF、WeightedRanker、Rerank。
- 嵌入：OpenAI 兼容 Embedding API，自定义 BM25 稀疏向量。
- 数据：PostgreSQL、Redis、Milvus、MinIO、etcd。
- 前端：Vue 3 CDN、marked、highlight.js、ReadableStream、SSE。
- 评估：`eval/` 离线检索评估，支持 Recall@K、Precision@K、MRR、NDCG、Hit Rate。

## 三、整体架构

```text
用户 / 前端
  -> FastAPI API
  -> JWT Auth / RBAC
  -> Agent Router
       |-- llm          -> 直接模型回答
       |-- rag          -> 知识库工具 + LangGraph RAG
       |-- baidu        -> Web Search 工具
       |-- life_service -> 天气 / 生活服务工具
       |-- calendar     -> 日期 / 时区工具
       `-- planner      -> MultiStepPlanner
                              -> TaskExecutor
                              -> ToolRegistry
                              -> Synthesis
  -> Agent Trace / Memory Trace / RAG Trace
  -> PostgreSQL + Redis + Milvus
```

系统保留旧 Agent 的兼容行为：`get_agent_tools()` 仍只暴露 `get_current_weather` 和 `search_knowledge_base` 给原有 LangChain Agent；但 Router 和 Planner 可以通过统一 `ToolRegistry` 使用 `baidu_search`、`life_service`、`calendar_time` 等扩展工具。

## 四、核心模块

### 4.1 API 与认证

- `backend/app.py`：FastAPI 应用入口、CORS、静态前端挂载。
- `backend/api.py`：认证、聊天、会话、记忆、文档管理接口。
- `backend/auth.py`：JWT 生成校验、PBKDF2-SHA256 密码哈希、bcrypt 历史兼容、管理员依赖。
- `backend/models.py`：`users`、`chat_sessions`、`chat_messages`、`user_memories`、`parent_chunks`。
- `backend/cache.py`：Redis Cache-Aside 缓存，会话写入和删除时主动失效。

认证原则是后端只信任 `Authorization: Bearer <access_token>`，聊天和会话接口不再接收前端传入的 `user_id`。

### 4.2 Agent Router

- 文件：`backend/agent_router.py`
- 输出模型：`RouterDecision`
- 字段：`intent`、`complexity`、`requires_rag`、`requires_web`、`requires_weather`、`requires_calendar`、`requires_planning`、`reason`

路由目标：

| 条件 | route |
|---|---|
| 普通常识、闲聊 | `llm` |
| 知识库、上传文档、项目资料 | `rag` |
| 最新资料、网络搜索、新闻 | `baidu` |
| 天气、预报 | `life_service` |
| 日期、星期、时区、日期计算 | `calendar` |
| 多工具、复杂任务、需要计划 | `planner` |
| Router 异常 | `fallback` |

关键设计是 Router 不只选单一工具，而是保留多个 `requires_*` 标志。只要任务复杂、要求规划，或同时需要多个外部能力，就进入 Planner。

### 4.3 Multi-step Planner

- 文件：`backend/agent_planner.py`
- 核心模型：`Plan`、`Task`、`PlannerState`、`PlannerRunResult`
- 执行器：`TaskExecutor`

Planner 由结构化 LLM Output 生成 Plan。每个 Task 包含：

- `id`
- `description`
- `tool`
- `dependencies`
- `input`
- `timeout_seconds`
- `max_retries`
- `status`
- `result`
- `error`
- `attempts`

执行规则：

- 无依赖任务同批并行执行。
- 有依赖任务等待上游全部完成。
- 上游失败时下游标记为失败或跳过。
- `synthesis` 任务负责汇总前置任务结果。
- 用户终止时通过 `cancel_event` 取消仍在执行或等待的任务。

Planner 事件会实时发送到前端，包括 `planner_started`、`task_created`、`task_started`、`tool_start`、`tool_result`、`task_completed`、`task_failed`、`parallel_execution_started`、`synthesis_started`。为兼容旧 UI，这些事件也会映射成 `rag_step`。

### 4.4 统一工具基础设施

- `backend/tools/base.py`：`BaseTool`
- `backend/tools/registry.py`：`ToolRegistry`
- `backend/tools/models.py`：`ToolResult`、`ToolExecutionTrace`、`ToolStatus`
- 工具实现：`weather.py`、`knowledge.py`、`baidu_search.py`、`life_service.py`、`calendar_time.py`

所有工具统一返回 `ToolResult`：

- 成功时包含 `data`、`content`、`trace`。
- 失败时包含 `error` 和失败状态。
- Trace 记录 `tool_name`、`latency_ms`、`attempts`、`status`。

`ToolRegistry` 负责注册、执行、健康状态和 LangChain Tool 适配，并在工具连续失败时结合 Circuit Breaker 快速返回不可用结果，避免 Planner 或 Agent 在坏工具上反复重试。

### 4.5 RAG 工作流

- `backend/rag_pipeline.py`：LangGraph RAG 状态机。
- `backend/rag_utils.py`：检索、Auto-merging、Rerank、Step-Back、HyDE。
- `backend/milvus_client.py`：Milvus collection、dense/sparse/hybrid 检索。
- `backend/embedding.py`：Embedding API + BM25 稀疏向量。
- `backend/rag_conflicts.py`：多文档冲突检测。

RAG 状态机：

```text
retrieve_initial
  -> grade_documents
       |-- yes -> END
       `-- no  -> rewrite_question
                    -> retrieve_expanded
                    -> END
```

核心策略：

- 初次召回只过滤 `chunk_level == 3` 的叶子块。
- 默认 Hybrid：Dense + Sparse 两路检索，使用 RRF 或 WeightedRanker 融合。
- 支持 `retrieval_mode=dense`、`bm25`、`hybrid`，便于评估对比。
- Rerank 使用 `/v1/rerank` 兼容接口，未配置或失败时自动跳过。
- 检索后执行 Auto-merging，命中同一父块超过阈值时回取 L2/L1 父块。
- 相关性不足时通过 Step-Back、HyDE 或 complex 进行重写和二次召回。
- RAG Trace 记录两轮召回、评分、重写策略、Rerank 分数、Auto-merging 元数据和冲突检测结果。

### 4.6 RAG Deep Engineering

文档处理不再只是递归字符切分，而是优先结构化解析：

- `DocumentStructureParser` 识别 Heading、Paragraph、List、Code、Table、Image。
- `SemanticChunker` 先聚合 2000-3000 token 的结构块，再切成约 512-1024 token 的语义块。
- 代码块保持完整。
- 表格转换为结构化文本。
- 图片记录 caption、source、surrounding_text、`multimodal_embedding=false`，当前不伪装支持多模态 embedding。
- 结构化解析失败时回退 Recursive Character Splitter。

三级层级仍保留：

```text
L1：较大主题块
  -> L2：中等上下文块
      -> L3：叶子检索块
```

存储策略是 Leaf-only：

- L3 写入 Milvus。
- L1/L2 写入 PostgreSQL `parent_chunks`。
- 检索命中 L3 后按 `parent_chunk_id` 和 `root_chunk_id` 自动回取父块。

### 4.7 混合检索与 BM25

Dense 检索使用 OpenAI 兼容 Embedding API，向量维度由 `DENSE_DIM` 和实际模型配置决定。

BM25 是项目内自定义实现：

- 中文按单字切分。
- 英文按正则单词切分。
- 维护 `_vocab`、`_doc_freq`、`_total_docs`、`_avg_doc_len`。
- `BM25_K1` 和 `BM25_B` 可通过环境变量或评估参数调整。

需要注意：当前代码没有使用 jieba。面试和文档口径应说“手写 BM25 + 自定义中英文分词”，不要说“jieba BM25”。

Hybrid 支持：

- `HYBRID_DENSE_WEIGHT`
- `HYBRID_SPARSE_WEIGHT`
- `HYBRID_RRF_K`
- RRF 或 WeightedRanker

如果 Hybrid 调用失败，会降级到 Dense 或返回失败元数据，保证外部依赖不直接击穿主流程。

### 4.8 多文档冲突检测

`rag_conflicts.py` 对检索结果做保守冲突检测：只有相同主语或断言在不同来源出现不同取值时，才设置 `rag_trace.conflict.has_conflict=true`。

系统不会替用户自动裁决冲突，而是在知识库工具输出中明确提示“知识库中存在来源冲突”，并附带：

- 来源文件
- 观点或取值
- 证据文本
- 文档可用时间

这种设计避免 RAG 在证据冲突时强行生成单一答案。

### 4.9 Memory

- `backend/memory.py`
- `backend/models.py::UserMemory`
- `backend/api.py` 的 `/memories` CRUD 接口

Memory 分为：

- Working Memory：当前请求内的问题、路由、检索记忆，不落库。
- Conversation Memory：`ChatSession` 和 `ChatMessage` 保存会话历史。
- Long-term Memory：长期个人背景或目标。
- Preference Memory：稳定偏好，例如回答语言、格式偏好。

记忆策略：

- 按 `user_id` 隔离。
- 只保存稳定偏好、个人背景和长期目标。
- 拒绝一次性任务、临时提醒、API Key、密码、Token、Secret 等敏感内容。
- `memory_trace` 只记录工程摘要，例如检索到的 memory id、新建 memory id 和跳过原因。

### 4.10 Agent Trace 与观测

- 文件：`backend/agent_observability.py`
- 主模型：`AgentTrace`

最终 `trace` SSE 事件同时返回：

- `agent_trace`：新版本完整 Trace。
- `rag_trace`：兼容旧前端和旧历史数据。

Agent Trace 包含：

- `request_id`、`session_id`、`user_id`
- `intent`、`complexity`、`route`
- `plan`
- `tool_calls`
- `rag_trace`
- `memory`
- `task_results`、`task_traces`
- `recovery_attempts`、`recovery_events`
- `latency`
- `errors`
- `final_status`

敏感字段会被摘要和过滤，Trace 默认不保存 API Key、Token、password、secret 和完整敏感负载。

### 4.11 Reliability

- 文件：`backend/agent_reliability.py`
- 配置：`ReliabilityConfig`

可靠性能力：

- Tool timeout、retry、指数退避。
- Task retry、timeout、cancel。
- `MAX_AGENT_STEPS` 限制最大执行步数。
- `_is_stuck` 检测重复任务、重复工具调用、重复 query、连续工具失败、Planner 无进展。
- `attempt_loop_recovery` 按顺序尝试修改 query、降低 `top_k`、Web Search 回退到 RAG、跳过当前任务、给用户返回稳定失败说明。
- Circuit Breaker 在工具连续失败后短期熔断。

这部分的价值是把 Agent 从“能跑”推进到“失败可控、可解释、可恢复”。

## 五、核心请求链路

### 5.1 流式聊天链路

```text
POST /chat/stream
  -> JWT 认证
  -> 读取 PostgreSQL / Redis 会话
  -> 清理上一轮 RAG 上下文和工具调用保护
  -> Router 生成 RouterDecision
  -> 检索并注入 Memory Prompt
  -> 按 route 执行 LLM / RAG / 工具 / Planner
  -> 实时输出 agent_step / router_step / planner_step / tool_* / task_* / rag_step / content
  -> 完成后生成 Agent Trace
  -> 写入 PostgreSQL，失效 Redis 缓存
```

### 5.2 文档入库链路

```text
POST /documents/upload
  -> require_admin
  -> 保存 PDF / Word 原文件
  -> 删除同名旧 Milvus 叶子块
  -> 删除同名旧 PostgreSQL 父块
  -> 文档结构解析 + 语义分块 + 三级层级分块
  -> L1/L2 写入 parent_chunks
  -> L3 生成 Dense + Sparse 向量
  -> L3 写入 Milvus
```

### 5.3 非流式聊天链路

`POST /chat` 复用同一套 Router、Planner、RAG、Memory 和 Trace 逻辑，只是不通过 SSE 分块返回。

## 六、API 速览

认证：

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

聊天：

- `POST /chat`
- `POST /chat/stream`

会话：

- `GET /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`

记忆：

- `GET /memories`
- `POST /memories`
- `PATCH /memories/{memory_id}`
- `DELETE /memories/{memory_id}`

文档：

- `GET /documents`
- `POST /documents/upload`
- `DELETE /documents/{filename}`

除注册和登录外，接口都需要：

```http
Authorization: Bearer <access_token>
```

## 七、环境变量

模型：

- `ARK_API_KEY`
- `MODEL`
- `BASE_URL`
- `EMBEDDER`
- `EMBEDDING_BASE_URL`
- `DENSE_DIM`

检索：

- `MILVUS_HOST`
- `MILVUS_PORT`
- `MILVUS_COLLECTION`
- `BM25_K1`
- `BM25_B`
- `HYBRID_DENSE_WEIGHT`
- `HYBRID_SPARSE_WEIGHT`
- `HYBRID_RRF_K`
- `AUTO_MERGE_ENABLED`
- `AUTO_MERGE_THRESHOLD`
- `LEAF_RETRIEVE_LEVEL`

Rerank：

- `RERANK_MODEL`
- `RERANK_BINDING_HOST`
- `RERANK_API_KEY`

数据与认证：

- `DATABASE_URL`
- `REDIS_URL`
- `JWT_SECRET_KEY`
- `ACCESS_TOKEN_EXPIRE_MINUTES`

工具：

- `AMAP_WEATHER_API`
- `AMAP_API_KEY`
- `BAIDU_SEARCH_API_KEY`
- `BAIDU_SEARCH_API_URL`
- `DEFAULT_TIMEZONE`

可靠性：

- `MAX_TOOL_RETRIES`
- `MAX_TASK_RETRIES`
- `MAX_AGENT_STEPS`
- `MAX_RECOVERY_ATTEMPTS`
- `RETRY_BACKOFF_SECONDS`
- `RETRY_BACKOFF_MAX_SECONDS`
- `STUCK_REPEAT_THRESHOLD`
- `CIRCUIT_BREAKER_ENABLED`
- `CIRCUIT_BREAKER_FAILURE_THRESHOLD`
- `CIRCUIT_BREAKER_RESET_SECONDS`

Memory：

- `MEMORY_MAX_RETRIEVED`

## 八、部署与启动

```bash
uv sync
docker compose up -d postgres redis etcd minio standalone
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

访问：

- 前端：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

旧 JSON 数据迁移：

```bash
python backend/migrate_legacy.py --default-password "请替换为临时密码"
```

## 九、评估与测试

离线 RAG 评估：

```bash
python -m eval.runner --dataset eval/datasets
```

默认输出：

- `eval/reports/rag_eval_report.json`
- `eval/reports/rag_eval_report.csv`

当前测试覆盖方向：

- Router 十类问题路由。
- Planner 依赖、并行、失败、超时和取消。
- Reliability 重试、熔断、死循环检测和恢复。
- Tool Infrastructure 外部工具契约。
- RAG 冲突检测。
- Memory CRUD、隐私过滤和 Trace。
- 文档结构解析、语义分块和代码块保留。
- RAG 离线评估指标。

## 十、工程风险与改进方向

当前仍需主动说明的风险：

- 数据库迁移：生产环境应使用 Alembic 替代 `Base.metadata.create_all()`。
- 首个用户自动成为管理员：适合 Demo，生产应改为初始化管理员脚本或环境变量。
- BM25 统计：词表和文档统计主要在进程内存中，多实例部署需要持久化或引入成熟稀疏检索引擎。
- RAG 上下文：部分兼容链路仍使用模块级上下文传递，应继续推进请求级 `ContextVar` 或依赖注入。
- Web Search Provider：当前百度搜索工具依赖注入的 Provider 契约，真实供应商接入需要明确认证方式和响应 JSON。
- 多模态：文档图片只记录 metadata，不声称已经支持多模态 embedding。

下一步优先级建议：

1. 引入 Alembic。
2. 持久化 BM25 词表和统计，或接入成熟搜索引擎。
3. 用请求级上下文替换模块级最近一次 RAG Trace。
4. 扩充真实 gold 评测集，系统对比 Dense、BM25、Hybrid、Hybrid + Rerank、Rewrite。
5. 完善 Refresh Token、退出登录、密码修改、API 限流和审计日志。
