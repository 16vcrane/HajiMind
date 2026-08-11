# HajiMind 项目说明

> HajiMind（谐音「哈基米」）是一个基于 FastAPI、LangChain、LangGraph、Milvus、PostgreSQL 和 Redis 构建的流式、可观测、多用户 RAG Agent 系统。

HajiMind 已从早期的知识库 RAG 聊天 Demo 升级为完整的 AI Agent 服务：它能先判断用户问题类型，再选择 LLM、RAG、Web Search、天气、日历或多步骤 Planner；复杂任务会被拆解成结构化任务图并按依赖执行；整个过程通过 SSE 实时输出，并生成 Agent Trace、RAG Trace、Memory Trace 和任务执行 Trace。

相关文档：

- [CHANGELOG.md](CHANGELOG.md)：版本变更摘要。

## 核心能力

- **多用户服务化**：JWT 登录认证、RBAC、首个用户自动成为管理员、普通用户会话隔离。
- **Agent Router**：使用结构化 LLM Output 生成 `RouterDecision`，把问题路由到 `llm`、`rag`、`baidu`、`life_service`、`calendar` 或 `planner`。
- **Multi-step Planner**：复杂任务生成结构化 `Plan` 和 `Task`，由 `TaskExecutor` 按依赖图串行或并行执行。
- **统一 Tool Infrastructure**：所有工具继承 `BaseTool`，通过 `ToolRegistry` 统一注册、执行、校验、超时、重试、Trace 和 Circuit Breaker。
- **LangGraph RAG**：初次召回、相关性评分、查询重写、二次召回和完整 RAG Trace。
- **Hybrid Retrieval**：Milvus Dense + BM25 Sparse 混合检索，支持 RRF、WeightedRanker、Rerank 和降级。
- **RAG Deep Engineering**：结构化文档解析、语义分块、三级分块、Leaf-only 向量存储和 Auto-merging。
- **多文档冲突检测**：对不同来源中相同事实的冲突观点保留证据，不自动裁决。
- **结构化 Memory**：Working、Conversation、Long-term、Preference Memory，用户隔离并过滤敏感信息。
- **流式输出与工作流可视化**：SSE 推送 `content`、`router_step`、`planner_step`、`tool_start`、`tool_result`、`task_start`、`task_result`、`rag_step` 和 `trace`。
- **Agent Reliability**：工具/任务重试、指数退避、最大步数、死循环检测、恢复策略和熔断。
- **离线评估**：`eval/` 支持 Recall@K、Precision@K、MRR、NDCG 和 Hit Rate。

## 技术栈

- 后端：FastAPI、Uvicorn、Pydantic、SQLAlchemy。
- Agent：LangChain Agent、LangGraph、结构化 LLM Output。
- 检索：Milvus、Dense Vector、SPARSE_FLOAT_VECTOR、RRF、WeightedRanker、Rerank。
- 嵌入：OpenAI 兼容 Embedding API，自定义 BM25 稀疏向量。
- 数据：PostgreSQL、Redis、Milvus、MinIO、etcd。
- 前端：Vue 3 CDN、marked、highlight.js、ReadableStream、SSE。
- 评估：Python `eval/` 模块，JSONL 数据集，JSON/CSV 报告。

## 快速启动

### 1. 环境准备

- Python `3.12+`
- `uv` 或 `pip`
- Docker / Docker Compose
- 一个 OpenAI 兼容的 Chat Completions / Embeddings 服务

### 2. 安装依赖

推荐使用 `uv`：

```bash
uv sync
```

也可以使用 `pip`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Windows PowerShell 可使用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，并替换模型、数据库和外部工具配置。

```env
# ===== Model =====
ARK_API_KEY=your_ark_api_key
MODEL=your_model_name
BASE_URL=https://your-llm-endpoint/v1
EMBEDDER=your_embedding_model
DENSE_DIM=2560
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1

# ===== Rerank (optional) =====
RERANK_MODEL=your_rerank_model
RERANK_BINDING_HOST=https://your-rerank-host
RERANK_API_KEY=your_rerank_api_key

# ===== Database / Cache =====
DATABASE_URL=postgresql+psycopg://hajimind:hajimind@127.0.0.1:5432/hajimind
REDIS_URL=redis://127.0.0.1:6379/0
JWT_SECRET_KEY=replace-with-a-long-random-secret
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# ===== Milvus =====
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_COLLECTION=hajimind_docs

# ===== Tools =====
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=your_amap_api_key
BAIDU_SEARCH_API_KEY=your_baidu_search_api_key
BAIDU_SEARCH_API_URL=https://your-baidu-search-provider-endpoint
DEFAULT_TIMEZONE=UTC

# ===== Memory =====
MEMORY_MAX_RETRIEVED=6

# ===== Agent Reliability =====
MAX_TOOL_RETRIES=2
MAX_TASK_RETRIES=2
MAX_AGENT_STEPS=24
MAX_RECOVERY_ATTEMPTS=3
RETRY_BACKOFF_SECONDS=0.25
RETRY_BACKOFF_MAX_SECONDS=2.0
STUCK_REPEAT_THRESHOLD=3
CIRCUIT_BREAKER_ENABLED=true
CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
CIRCUIT_BREAKER_RESET_SECONDS=30

# ===== RAG Retrieval / Evaluation =====
BM25_K1=1.5
BM25_B=0.75
HYBRID_DENSE_WEIGHT=0.5
HYBRID_SPARSE_WEIGHT=0.5
HYBRID_RRF_K=60
LEAF_RETRIEVE_LEVEL=3
AUTO_MERGE_ENABLED=true
AUTO_MERGE_THRESHOLD=2
```

注意：

- `JWT_SECRET_KEY` 生产环境必须使用随机长密钥。
- `BAIDU_SEARCH_API_URL` 当前需要明确供应商 Provider 契约后才能接入真实在线搜索。
- 图片只保存 metadata，不代表当前已经启用多模态 embedding。

### 4. 启动依赖服务

`docker-compose.yml` 同时提供 PostgreSQL、Redis、Milvus、etcd、MinIO 和 Attu。

```bash
docker compose up -d postgres redis etcd minio standalone attu
docker compose ps
```

端口：

| 服务 | 端口 |
|---|---|
| PostgreSQL | `5432` |
| Redis | `6379` |
| Milvus | `19530` |
| Milvus health | `9091` |
| MinIO API | `9000` |
| MinIO Console | `9001` |
| Attu | `8080` |

### 5. 启动应用

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

或：

```bash
python backend/app.py
```

访问：

- 前端页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

首次注册的用户会自动成为管理员，后续注册用户默认为普通用户。

## 系统架构

```text
Vue 3 前端
  -> FastAPI API
  -> JWT Auth / RBAC
  -> Agent Router
       |-- llm          -> 直接模型回答
       |-- rag          -> KnowledgeBaseTool + LangGraph RAG
       |-- baidu        -> BaiduSearchTool
       |-- life_service -> LifeServiceTool
       |-- calendar     -> CalendarTimeTool
       `-- planner      -> MultiStepPlanner
                              -> TaskExecutor
                              -> ToolRegistry
                              -> Synthesis
  -> Agent Trace / RAG Trace / Memory Trace
  -> PostgreSQL / Redis / Milvus
```

职责边界：

- Router 负责入口分流，不回答用户问题。
- Planner 只处理复杂、多工具、多步骤任务。
- LangGraph RAG 负责确定性的知识库检索流程。
- ToolRegistry 负责工具注册、执行和观测。
- PostgreSQL 是事实来源，Redis 只做缓存。
- Milvus 只存可检索的 L3 叶子分块向量。

## 目录结构

```text
backend/
  app.py                    FastAPI 入口、CORS、静态资源挂载
  api.py                    认证、聊天、会话、记忆、文档接口
  agent.py                  Router/Planner/RAG/Memory 编排和流式输出
  agent_router.py           RouterDecision 与路由目标解析
  agent_planner.py          Plan、Task、TaskExecutor、多步骤执行
  agent_observability.py    AgentTrace、ToolTrace、TaskTrace、敏感字段摘要
  agent_reliability.py      重试、熔断、死循环检测和恢复策略
  auth.py                   JWT、密码哈希、管理员权限
  cache.py                  Redis Cache-Aside
  database.py               SQLAlchemy engine/session/init
  models.py                 User、ChatSession、ChatMessage、UserMemory、ParentChunk
  memory.py                 长期记忆、偏好记忆、隐私过滤和提示词注入
  document_loader.py        PDF/Word 加载与三级分块
  document_structure.py     结构化解析与语义分块
  embedding.py              Embedding API 与 BM25 Sparse Encoder
  milvus_client.py          Milvus collection、dense/sparse/hybrid 检索
  milvus_writer.py          L3 叶子块向量写入
  parent_chunk_store.py     L1/L2 父级分块 PostgreSQL 存取
  rag_pipeline.py           LangGraph RAG 状态机
  rag_utils.py              检索、Rerank、Auto-merging、Step-Back、HyDE
  rag_conflicts.py          多文档冲突检测
  tools/                    统一工具基础设施与工具实现

frontend/
  index.html
  script.js
  style.css

eval/
  dataset.py
  metrics.py
  runner.py
  reporter.py
  datasets/

tests/
```

## 核心流程

### 1. 流式聊天

```text
POST /chat/stream
  -> JWT 认证
  -> 加载 PostgreSQL / Redis 会话
  -> Router 输出 RouterDecision
  -> 检索并注入 Memory Prompt
  -> 按 route 执行 LLM / RAG / Tool / Planner
  -> SSE 推送 content、router_step、planner_step、tool_*、task_*、rag_step
  -> 生成 Agent Trace
  -> 写入 PostgreSQL，失效 Redis 缓存
```

### 2. Router 路由

`RouterDecision` 字段：

- `intent`
- `complexity`
- `requires_rag`
- `requires_web`
- `requires_weather`
- `requires_calendar`
- `requires_planning`
- `reason`

路由规则：

| 问题类型 | 路由 |
|---|---|
| 闲聊、常识、无需工具 | `llm` |
| 用户上传文档、知识库、项目资料 | `rag` |
| 最新资料、网页、新闻、外部实时信息 | `baidu` |
| 天气、天气预报 | `life_service` |
| 当前时间、日期差、星期、时区 | `calendar` |
| 多工具、复杂任务、需要计划 | `planner` |
| Router 异常 | `fallback` |

### 3. Planner 执行

```text
用户复杂问题
  -> MultiStepPlanner.create_plan()
  -> Plan(tasks=[...])
  -> TaskExecutor
       -> 找出依赖已完成的 ready tasks
       -> 无依赖任务并行执行
       -> 有依赖任务等待上游
       -> 工具结果写入 task_results
       -> synthesis 汇总最终回答
  -> PlannerState 写入 Agent Trace
```

Planner 支持：

- 任务状态：`pending`、`running`、`completed`、`failed`、`skipped`、`cancelled`
- 任务依赖校验和环检测
- timeout、retry、cancel
- 并行批次执行
- 工具执行 Trace
- 恢复策略和失败说明

### 4. RAG 工作流

```text
retrieve_initial
  -> grade_documents
       |-- yes -> END
       `-- no  -> rewrite_question
                    -> retrieve_expanded
                    -> END
```

RAG Trace 记录：

- 原始问题
- 初次召回结果
- 相关性评分
- 是否触发重写
- Step-Back / HyDE / complex 策略和中间内容
- 二次召回结果
- Dense / BM25 / Hybrid 模式
- RRF / WeightedRanker / Rerank 元数据
- Auto-merging 元数据
- 多文档冲突检测结果

### 5. 文档入库

```text
管理员上传 PDF / Word
  -> 保存原文件
  -> 删除同名旧 Milvus 叶子块
  -> 删除同名旧 PostgreSQL 父块
  -> 结构化解析 Heading / Paragraph / List / Code / Table / Image
  -> 语义分块
  -> L1/L2/L3 三级分块
  -> L1/L2 写 PostgreSQL parent_chunks
  -> L3 生成 Dense + Sparse 向量
  -> L3 写 Milvus
```

文档处理策略：

- 优先结构化解析，失败时回退 Recursive Character Splitter。
- 代码块保持完整。
- 表格转换为结构化文本。
- 图片记录 caption、source、surrounding_text、`multimodal_embedding=false`。
- 只向量化 L3 叶子块，父块通过 Auto-merging 回取。

### 6. Memory

Memory 分层：

- Working Memory：当前请求内的问题、路由和检索上下文，不落库。
- Conversation Memory：`ChatSession` 和 `ChatMessage` 保存会话历史。
- Long-term Memory：长期个人背景或目标。
- Preference Memory：稳定偏好，例如回答语言、输出格式。

策略：

- 按用户隔离。
- 自动提取只接受稳定偏好、个人背景和长期目标。
- 拒绝临时任务、提醒、API Key、密码、Token、Secret 等敏感信息。
- `memory_trace` 只保存工程摘要，不保存敏感原文。

## 检索设计

### Dense + Sparse Hybrid

Dense 负责语义相似，Sparse 负责关键词、编号和专有名词。Milvus collection 同时存储：

- `dense_embedding`：`FLOAT_VECTOR`
- `sparse_embedding`：`SPARSE_FLOAT_VECTOR`

检索模式：

- `dense`
- `bm25`
- `hybrid`
- `hybrid_weighted`

### BM25 口径校准

当前代码没有使用 `jieba`。BM25 是项目内自定义实现：

- 中文按单字切分。
- 英文按正则单词切分。
- 维护 `_vocab`、`_doc_freq`、`_total_docs`、`_avg_doc_len`。
- `BM25_K1` 和 `BM25_B` 可配置。

Embedding 也是 OpenAI 兼容端点，维度由模型和 `DENSE_DIM` 配置决定，不应固定描述为 `text-embedding-3-small / 1536 维`。

### Rerank 与降级

Rerank 使用 `/v1/rerank` 兼容接口。未配置或调用失败时保留原始召回结果，不阻断问答。

Hybrid 检索失败时记录失败元数据，并尽量回退到 Dense 检索或返回可解释错误。

## 流式输出与实时观测

SSE 事件类型：

- `agent_step`
- `router_step`
- `planner_step`
- `tool_start`
- `tool_result`
- `task_start`
- `task_result`
- `rag_step`
- `trace`
- `content`
- `error`
- `[DONE]`

旧事件兼容：

- `planner_started`
- `task_created`
- `task_started`
- `task_completed`
- `task_failed`
- `parallel_execution_started`
- `synthesis_started`

### 跨线程事件调度

同步 RAG 工具可能在线程池中执行，不能直接操作主线程的 `asyncio.Queue`。系统在主线程捕获 event loop，并在工具线程中使用：

```python
loop.call_soon_threadsafe(queue.put_nowait, step)
```

这样工具执行期间也能实时推送检索、评分、重写、Planner 和 Tool 状态，而不是等工具返回后一次性展示。

### 终止与资源回收

前端通过 `AbortController.abort()` 断开 fetch。后端生成器收到 `GeneratorExit` 后显式 `agent_task.cancel()`，让上游流式 LLM 请求释放连接，减少无效推理和 token 消耗。

## API 速览

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

## 数据存储

PostgreSQL：

- `users`：用户、角色、启用状态、密码哈希。
- `chat_sessions`：按用户隔离的会话。
- `chat_messages`：消息内容、时间戳、Agent/RAG Trace。
- `user_memories`：长期记忆和偏好记忆。
- `parent_chunks`：L1/L2 父级分块。

Redis：

- `hajimind:sessions:list:{user_id}`
- `hajimind:sessions:{user_id}:{session_id}`

Milvus：

- 只保存 L3 叶子块向量和检索元数据。

旧数据迁移：

```bash
python backend/migrate_legacy.py --default-password "请替换为临时密码"
```

迁移后应要求旧用户修改临时密码。

## 评估

运行离线 RAG 评估：

```bash
python -m eval.runner --dataset eval/datasets
```

默认输出：

- `eval/reports/rag_eval_report.json`
- `eval/reports/rag_eval_report.csv`

支持指标：

- Recall@K
- Precision@K
- MRR
- NDCG
- Hit Rate

建议对比：

- Dense only
- BM25 only
- Hybrid
- Hybrid + Rerank
- Hybrid + Query Rewrite

## 测试覆盖

当前测试覆盖：

- Router 十类问题路由。
- Planner 依赖执行、并行执行、失败、超时和取消。
- Reliability 重试、熔断、死循环检测和恢复。
- Tool Infrastructure 外部工具契约。
- RAG 冲突检测。
- Memory CRUD、隐私过滤和 Agent Trace。
- 文档结构解析、语义分块和代码块保留。
- RAG 离线评估指标。

常用检查：

```bash
python -m compileall -q backend eval tests
python -m unittest discover -s tests
```

## 当前边界与生产化风险

- 数据库迁移：当前适合本地迭代，生产环境应使用 Alembic。
- 首个用户自动成为管理员：适合 Demo，生产应改为初始化管理员命令、邀请码或环境变量。
- BM25 统计：词表和文档频率主要在进程内存中，多实例部署需要持久化或引入成熟搜索引擎。
- 请求级上下文：部分兼容链路仍使用模块级上下文传递，后续应迁移到 `contextvars.ContextVar` 或请求级依赖注入。
- Web Search Provider：真实百度搜索需要明确接口 URL、认证方式、请求格式和成功响应 JSON。
- 多模态能力：当前仅记录图片 metadata，不声称已支持多模态 embedding。
- JWT：生产环境需要短期 access token、refresh token、退出登录和 token 撤销策略。

## 更新日志摘要

### Unreleased

- 增加 Agent Router，支持结构化路由决策。
- 增加 Multi-step Planner、依赖感知 TaskExecutor 和并行任务执行。
- 增加 Planner/Tool/Task 流式事件与 Agent Workflow 可视化。
- 增加统一 Agent Trace，覆盖 Router、Planner、Tool、Task、RAG、Memory、latency、error 和 final status。
- 增加 Reliability：bounded retry、指数退避、Circuit Breaker、最大步数、死循环检测和恢复策略。
- 增加结构化文档解析、语义分块、代码块/表格/图片 metadata 支持。
- 增加可配置 BM25、Hybrid dense/sparse 权重、sparse 检索和离线 RAG 评估。
- 增加多文档冲突检测。
- 增加结构化 Memory、隐私过滤、用户隔离 CRUD 和 Trace 集成。

### 2026-03-21 后端服务建设升级

- 新增注册、登录、JWT Token 和管理员权限控制。
- 聊天历史迁移到 PostgreSQL，按认证用户隔离。
- 父级分块迁移到 PostgreSQL。
- 引入 Redis 缓存会话列表和消息读取结果。
- 聊天和会话接口不再接收前端传入的 `user_id`。
- 文档列表、上传、删除接口仅允许管理员访问。
- 密码使用 PBKDF2-SHA256，并兼容历史 bcrypt 哈希校验。

### 2026-03-13 三级分块与 Auto-merging 升级

- 新增 L1/L2/L3 三级滑动窗口分块。
- 存储策略调整为 Leaf-only：仅 L3 叶子块写入 Milvus。
- Auto-merging 从父块存储中回取 L2/L1。
- `rag_trace` 新增 `leaf_retrieve_level` 和 `auto_merge_*` 字段。

### 2026-02-19 RAG 实时思考链路修复

- 修复同步工具在线程池中无法向主 asyncio loop 推送 RAG step 的问题。
- 通过 `set_rag_step_queue` 捕获主线程 loop。
- `emit_rag_step` 使用 `call_soon_threadsafe` 跨线程调度事件。
- 前端初始化 `ragSteps`，确保 Vue 响应式更新。
