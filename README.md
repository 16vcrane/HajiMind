# HajiMind 项目说明

> HajiMind（谐音「哈基米」）——一个可观测、流式输出的 LangGraph RAG 智能体。

Agent的项目记录，方便后续持续更新与展示。


## 本地部署

### 1) 环境准备
- Python `3.12+`
- 包管理建议：`uv`（也支持 `pip`）
- Docker / Docker Compose（用于启动 Milvus 依赖）

### 2) 使用 pyproject 安装依赖
在项目根目录执行：

```bash
# 方式 A：推荐（uv）
uv sync

# 运行服务
uv run python backend/app.py
# 或
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

```bash
# 方式 B：pip
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .

# 运行服务
python backend/app.py
# 或
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

### 3) 创建 `.env` 文件
在项目根目录新建 `.env`，可直接使用下面模板：

```env
# ===== Model =====
ARK_API_KEY=your_ark_api_key
MODEL=your_model_name
BASE_URL=https://your-llm-endpoint/v1
EMBEDDER=your_embedding_model
DENSE_DIM=2560
# 可选：嵌入服务与聊天服务不同时单独配置
EMBEDDING_BASE_URL=https://your-embedding-endpoint/v1

# ===== Rerank (可选，不配则自动降级) =====
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

# ===== Tools （可选）=====
AMAP_WEATHER_API=https://restapi.amap.com/v3/weather/weatherInfo
AMAP_API_KEY=your_amap_api_key
BAIDU_SEARCH_API_KEY=your_baidu_search_api_key
BAIDU_SEARCH_API_URL=https://your-baidu-search-provider-endpoint
DEFAULT_TIMEZONE=UTC

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

```

### 4) Docker 部署（Milvus 向量库）
当前仓库的 `docker-compose.yml` 主要用于启动 Milvus 相关组件（`etcd` / `minio` / `standalone` / `attu`）：

```bash
# 启动向量库依赖
docker compose up -d

# 查看服务状态
docker compose ps

# 查看日志（可选）
docker compose logs -f standalone
```

端口说明：
- Milvus：`19530`
- Milvus 健康检查：`9091`
- MinIO API：`9000`
- MinIO Console：`9001`
- Attu：`8080`

### 5) 启动应用并访问
在 Milvus 启动后，运行后端应用：

```bash
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

浏览器访问：
- 前端页面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`

## 项目概览
- **核心能力**：
  - LangChain Agent + 自定义工具。
  - 文档上传后执行三级滑动窗口分块，叶子分块向量化写入 Milvus，父级分块写入本地 DocStore。
  - 会话记忆与摘要，保持长对话上下文。
- **运行形态**：FastAPI 后端 + 纯前端（Vue 3 CDN 单页）+ Milvus 向量库。

## 关键创新点
- **混合检索落地**：稠密向量 + BM25 稀疏向量，Milvus Hybrid Search + RRF 排序，兼顾语义与词匹配。
- **Jina Rerank 接入**：Hybrid/Dense 召回后进行 API 级精排，支持返回 `rerank_score` 并在前端可视化。
- **双向降级**：稀疏生成或 Hybrid 调用失败时自动降级为纯稠密检索，提升稳定性。
- **流式输出（Streaming）**：后端基于 `agent.astream(stream_mode="messages")` 逐 token 推送，前端 SSE + ReadableStream 实现打字机效果。
- **实时 RAG 过程可视化**：检索过程在模型"思考中"阶段就开始展示，通过 `asyncio.Queue` + 后台任务架构实现工具执行期间的实时推送。
- **回答终止功能**：前端 `AbortController` + 后端 `StreamingResponse` 支持用户随时中断正在生成的回答。
- **会话摘要记忆**：自动摘要旧消息并注入系统提示，维持上下文且控制 token。
- **文档处理链路**：上传 → 切分 → 稠密/稀疏向量同步生成 → Milvus 入库，支持重复上传自动清理旧 chunk。
- **三级分块 + Auto-merging**：L1/L2/L3 三层滑窗切分；检索时优先召回 L3，满足阈值后自动合并到父块（L3->L2->L1）。
- **Leaf-only 向量化存储**：仅叶子分块写入 Milvus，父块写入 DocStore，减少向量冗余并保留上下文聚合能力。
- **工具可扩展**：天气查询示例 + 知识库检索，便于按需增添第三方 API 或企业数据源。
- **统一 Tool Infrastructure**：所有工具通过 `ToolRegistry` 注册，继承 `BaseTool` 的输入校验、超时、重试、错误处理与执行 Trace；LangChain Agent 仍使用原有工具名称和文本输出契约。
- **RAG 过程可观测**：记录检索、评分、重写与来源信息，前端可展开查看每一步细节。
- **多文档冲突提示**：对不同来源中相同事实的相反结论保留全部证据，不强行选择单一答案；`rag_trace.conflict` 返回来源、观点、证据和可用时间信息。
- **查询重写体系**：Step-Back 与 HyDE 两种扩展方式 + 路由选择，必要时触发重写检索。
- **相关性评分门控**：基于结构化输出的 `grade_documents` 判断是否需要重写检索。
- **实时思考链路展示**：通过 `asyncio` 事件循环穿透技术，实现 Agent 在执行 RAG、评分、重写等同步工具时，实时向前端推送思考步骤（Searching -> Grading -> Rewriting），彻底解决"静默思考"问题。

## 未来迭代（Todo Lists）

### RAG部分

#### 数据层、Chunk分块

1. 先做文档结构解析，按文档结构做粗拆分，再用递归字符分块兜底，保证打的主题单元不被拆分（2000-3000token）；再用语义分块做精细化拆分，控制单块大小（512-1024token）
2. 代码块、表格、图片特殊处理
3. 实现 ParentDocument/Auto-merging Retriever 策略 --done

#### 召回层

1. BM25的k1和b新增参数扫描
2. RRF额外做BM25和dense的权重，可以通过AB test确定
3. 做一个小型标注集比较dense only、sparse only、hybrid、hybrid + rerank的gold chunk

#### 生成层

1. 子问题分解（CoT、专门的分解小模型、判断分几个子问题）
2. 多文档Refine（一次拼接、串行Refine）
3. 多文档冲突处理（A文档说X，B文档说非X），回答中显式输出“来源存在冲突”

#### 其他

1. 向量嵌入：新增多模态 embedding 能力
2. 搭建 RAG 评估体系
3. Rerank 策略评估（top_k、candidate_k、召回/精排比例）

### 其他能力拓展

1. 开发 SQL assistant Skill
2. 实现暂停功能与人工介入机制 --done
3. 新增问题类型判断，简单问题跳过复杂处理流程
4. 扩展网络搜索能力
5. 支持多步骤规划与任务并行执行
6. 搭建路由器节点，由 LLM 自主判断下一步动作
7. 优化 memory 管理：集成 MemO、LangMem 等方案
8. multi-agent：工具过多，把工具拆分给职责明确的专业化agent，提升工具选择的准确性和整体稳定性
9. 历史记录会话名称可修改
10. 死循环检测与恢复：_is_stuck + attempt_loop_recovery

### 后端服务建设

1. 实现用户注册登录、密码加密、权限管理，基于 sqlalchemy 搭建 ORM 数据库
2. 聊天记录落地数据库，引入 redis 做缓存优化

## 目录与架构
- 后端：`backend/`
  - [app.py](backend/app.py)：FastAPI 入口、CORS、静态资源挂载。
  - [api.py](backend/api.py)：聊天、会话、记忆与文档管理接口。
  - [agent.py](backend/agent.py)：LangChain Agent、会话存储、摘要逻辑。
  - [memory.py](backend/memory.py)：工作记忆、会话记忆、长期记忆和偏好记忆的提取、过滤、CRUD 与提示词注入。
  - `tools/`：统一工具基础设施与工具实现。
    - `base.py`：`BaseTool`，统一输入校验、超时、重试、错误结果和 LangChain 适配。
    - `registry.py`：`ToolRegistry`，负责注册、查询、健康状态和 LangChain Tool 列表。
    - `models.py`：`ToolResult`、`ToolExecutionTrace`、`ToolStatus`。
    - `weather.py` / `knowledge.py`：现有天气与知识库工具的兼容迁移。
  - [embedding.py](backend/embedding.py)：稠密向量 API 调用 + BM25 稀疏向量生成。
  - [document_loader.py](backend/document_loader.py)：PDF/Word 加载与分片。
  - [parent_chunk_store.py](backend/parent_chunk_store.py)：父级分块 DocStore（用于 Auto-merging 回取父块）。
  - [milvus_writer.py](backend/milvus_writer.py)：向量写入（稠密+稀疏）。
  - [milvus_client.py](backend/milvus_client.py)：Milvus 集合定义、混合检索。
  - [schemas.py](backend/schemas.py)：Pydantic 请求/响应模型。
- 前端：`frontend/`
  - [index.html](frontend/index.html) + [script.js](frontend/script.js) + [style.css](frontend/style.css)：Vue 3 + marked + highlight.js，提供聊天、历史会话、文档上传/删除界面。
- 数据：`data/`
  - `documents/`：上传文档原文件。
- PostgreSQL：用户、会话、消息和父级分块。
- Redis：会话列表和会话消息缓存。
- 向量库：Milvus（可由 `docker-compose` 或自建服务提供）。

## 核心流程

### 1) 项目全链路（端到端）
1. 用户在前端输入问题，调用 `POST /chat/stream`（流式）。
2. FastAPI `api.py` 返回 `StreamingResponse(media_type="text/event-stream")`。
3. LangChain Agent 根据问题类型决定是否调用工具：
  - 天气问题 → `get_current_weather`
  - 知识问答 → `search_knowledge_base`
4. 若命中知识库工具，进入 `rag_pipeline.py` 执行检索工作流，各阶段通过 `emit_rag_step()` 实时推送到前端。
5. 检索结果与 RAG Trace 一起返回，Agent 流式生成最终回答（逐 token 推送）。
6. 前端 ReadableStream 逐块解析 SSE，打字机效果实时渲染。
7. 同时消息写入 PostgreSQL，并通过 Redis 缓存会话读取结果，支持历史会话回放。

### 2) RAG 全链路（重点）
1. **初次召回**：`retrieve_initial`
  - 调用 `retrieve_documents`。
  - 先按 `chunk_level == 3` 执行 Milvus Hybrid 检索（Dense + Sparse + RRF）。
  - 取更大候选集后走 Jina Rerank 精排。
  - 对召回叶子块执行 Auto-merging（L3->L2->L1），父块从 DocStore 读取。
2. **相关性打分门控**：`grade_documents`
  - 使用结构化输出打分 `yes/no`。
  - `yes` 直接进入生成回答；`no` 进入重写阶段。
3. **查询重写路由**：`rewrite_question`
  - 在 `step_back / hyde / complex` 中选择策略。
  - 生成 `rewrite_query`、`step_back_question`、`hypothetical_doc` 等中间结果。
4. **二次召回**：`retrieve_expanded`
  - 对重写后的查询（或 HyDE 文档）再次检索。
  - 同样执行 L3 召回 + Auto-merging，结果去重后返回上下文。
5. **答案生成**：Agent 结合上下文生成最终回答。
6. **可观测追踪**：返回 `rag_trace`，包括
  - 评分结果与路由决策
  - 重写策略与重写内容
  - 初次/二次检索结果
  - 三级检索与合并信息（`leaf_retrieve_level`、`auto_merge_*`）
  - 检索分数 `score` 与精排分数 `rerank_score`

### 3) 文档入库链路
1. 前端上传 PDF/Word 到 `POST /documents/upload`。
2. `document_loader.py` 执行三级滑动窗口分块并写入层级元数据（chunk_id / parent_chunk_id / root_chunk_id / chunk_level）。
3. L1/L2 父级分块写入 `parent_chunk_store.py`（DocStore）。
4. L3 叶子分块进入 `embedding.py` 生成 Dense 向量与 BM25 Sparse 向量。
5. `milvus_writer.py` 仅将叶子块向量 + 元数据写入 Milvus。
5. 后续检索可直接利用新文档参与召回。

### 4) 会话记忆链路
1. 每轮问答按 JWT 中的用户身份和 `session_id` 写入 PostgreSQL。
2. 当消息过长时触发摘要压缩，保留长期上下文。
3. 前端可通过会话接口读取、删除历史对话。

## 技术栈
- 后端：FastAPI、LangChain Agents、Pydantic、Uvicorn。
- 向量与检索：Milvus（HNSW 稠密索引 + SPARSE_INVERTED_INDEX 稀疏索引）、RRF 融合、Jina Rerank 精排。
- 嵌入与稀疏：自定义 API 调用获取稠密向量；BM25 手写稀疏向量；同时输出双塔特征。
- 前端：Vue 3 (CDN)、marked、highlight.js、纯静态部署。
- 工具链：dotenv 配置、requests、langchain_text_splitters、langchain_community.loaders。

## Tool Architecture

当前 Agent 通过 `ToolRegistry` 获取 `get_current_weather` 和
`search_knowledge_base` 两个 LangChain Tool。每次工具执行都会生成统一的
`ToolResult` 与 `ToolExecutionTrace`，包含 `tool_name`、`status`、
`latency_ms`、`attempts` 和 `error`。Agent 仍接收原有的文本工具结果，
因此聊天 API、Streaming API 与 SSE 事件协议保持不变。

后续 Baidu Search、Calendar / Time、Life Service、SQL Assistant 等工具应继承
`BaseTool` 并通过同一个 Registry 注册，不应直接向 Agent 工具列表追加临时函数。

Phase 2 已注册 `baidu_search`、`life_service` 和 `calendar_time`，它们都会生成
统一 Trace，但尚未加入 `get_agent_tools()`。当前 Agent 仍仅暴露
`get_current_weather` 和 `search_knowledge_base`，以保持现有 Agent 选择行为、
聊天 API、Streaming API 和 SSE 协议不变。

- `life_service`：支持 `current_weather` 与 `forecast`，复用已有
  `AMAP_WEATHER_API` / `AMAP_API_KEY` 兼容逻辑；后续可扩展 POI、交通和餐饮。
- `calendar_time`：支持当前时间、日期差、加减天数、星期计算和时区转换；
  可使用 `Asia/Shanghai`、`Asia/Singapore` 和 `UTC`，默认时区由
  `DEFAULT_TIMEZONE` 配置，未配置时回退至 `UTC`。
- `baidu_search`：支持 `query` 和 `top_k`，结果标准化为 `title`、`url`、
  `snippet`、`source`、`published_at` 和 `rank`。项目当前没有百度搜索供应商的
  API 契约，因此工具仅接受已注入的 Provider Adapter，不会猜测认证头、请求参数
  或响应字段并发送线上请求。

要启用真实百度搜索 Provider Adapter，需要确认并提供：接口 URL、认证方式、
请求参数或请求体格式，以及一份成功响应 JSON 示例（含搜索结果字段）。

Phase 3 引入 Agent Router。Router 使用结构化 LLM Output 生成
`RouterDecision`，字段包括 `intent`、`complexity`、`requires_rag`、
`requires_web`、`requires_weather`、`requires_calendar`、`requires_planning`
和 `reason`。路由目标为：
- Simple → LLM
- Knowledge → RAG
- Web → Baidu
- Weather → LifeService
- Calendar → Calendar
- Complex / Mixed → Planner

Router 失败时自动回退到旧 Agent。Router 分析过程通过现有 `rag_step` SSE
事件展示，最终 `RouterDecision` 会写入 Agent Trace；未新增 SSE 事件类型。

Phase 4 引入真正的 Multi-step Planner。Router 判定为 `planner` 后会进入
`MultiStepPlanner`，由结构化 LLM Output 生成 `Plan`，再由 `TaskExecutor`
按依赖图执行任务。Planner State 会写入 Agent Trace：
- `plan`
- `current_task`
- `task_results`
- `completed_tasks`
- `failed_tasks`
- `execution_trace`

执行规则：
- 无依赖任务会以同一批次并行执行，例如 Baidu Search、RAG、Calendar。
- 有依赖任务会等待依赖全部完成后串行进入下一批次。
- 每个任务记录 `status`、`result`、`error` 和 `attempts`。
- 任务支持 retry、timeout；依赖失败时下游任务会标记失败/跳过。
- 用户 Abort 时会设置 Planner cancellation，并取消仍在等待中的任务。

Streaming 会实时发送 `planner_started`、`task_created`、`task_started`、
`task_completed`、`task_failed`、`parallel_execution_started` 和
`synthesis_started`。前端同时把这些事件映射到现有思考气泡中的步骤展示。

Phase 5 将原有 RAG Trace 扩展为完整 Agent Trace，并保留旧字段兼容。最终
`trace` SSE 事件同时返回 `agent_trace` 和兼容字段 `rag_trace`。Agent Trace
包含：
- `request_id`、`session_id`、`user_id`
- `intent`、`complexity`、`route`
- `plan`
- `tool_calls`
- `rag_trace`
- `task_results`、`task_traces`
- `latency`
- `errors`
- `final_status`

每个 Tool 和 Planner Task 只记录工程摘要：`start_time`、`end_time`、
`latency_ms`、`status`、`input_summary`、`output_summary` 和 `error`。摘要会过滤
API Key、token、password、secret 等敏感字段，不默认保存完整输入输出。

Streaming 现在同时支持新的观测事件：`agent_step`、`router_step`、
`planner_step`、`tool_start`、`tool_result`、`task_start`、`task_result`、
`trace`、`content`、`error` 和 `[DONE]`；旧的 `rag_step`、`planner_started`、
`task_created`、`task_started`、`task_completed`、`task_failed`、
`parallel_execution_started`、`synthesis_started` 仍保留兼容。

前端新增 Agent Workflow Panel，展示 User Question → Router → Planner →
Tool/RAG → Synthesis → Final Answer 的工程状态。Tool 节点展示名称、状态、
latency 和 result count；RAG 节点展示 Dense、BM25、RRF、Rerank 状态，不展示
模型内部 chain-of-thought。

Phase 6 增加 Agent Reliability。`BaseTool` 的 timeout、retry 与指数退避受
`MAX_TOOL_RETRIES` 约束；`ToolRegistry` 对连续失败工具启用 Circuit Breaker。
Planner 对任务 retry、执行步数和 recovery 次数分别受 `MAX_TASK_RETRIES`、
`MAX_AGENT_STEPS`、`MAX_RECOVERY_ATTEMPTS` 约束。执行 Trace 会检测重复 Tool
Call、重复 Query、重复 Planner Task、Planner 无进展和连续工具失败；恢复顺序为：
修改 Query、降低 `top_k`、Web Search 回退至 RAG、跳过当前任务，最后返回无法获得
稳定实时结果的用户可见说明。恢复状态写入 `recovery_attempts` 和
`recovery_events`，不包含 API Key 或完整敏感负载。

Phase 7 增加 RAG Deep Engineering：
- 文档优先按 Heading、Paragraph、List、Code、Table、Image 解析，失败时回退 Recursive Character Splitter。
- 结构块先按约 2000-3000 token 聚合，再按语义边界切成约 512-1024 token；代码块保持完整。
- Table 转换为结构化文本；Image 保存 caption、source、周边文本和 `multimodal_embedding=false`，当前不伪装支持多模态 embedding。
- BM25 支持 `k1` / `b`，Hybrid 支持 dense/sparse 权重和 RRF/WeightedRanker。
- `eval/datasets/` 支持真实 gold 标注与 LLM synthetic 数据，输出 Recall@K、Precision@K、MRR、NDCG、Hit Rate。

运行离线评估：
```bash
python -m eval.runner --dataset eval/datasets
```
默认输出 `eval/reports/rag_eval_report.json` 和
`eval/reports/rag_eval_report.csv`。评估器只基于 `gold_chunk_ids` 计算检索指标，
不会凭空生成准确率。

Phase 8 增加多文档冲突检测：RAG 在初次和扩展检索完成后，以可解释的事实主张比对不同来源。仅当相同主语/断言出现不同取值时，`rag_trace.conflict.has_conflict` 才为 `true`。系统不会自动裁决冲突；知识库工具会明确提示“知识库中存在来源冲突。”并提供来源、观点、证据和文档可用时间。

Phase 8 同时增加结构化 Memory：
- Working Memory：仅在当前请求内保存问题、路由与已检索记忆，不落库。
- Conversation Memory：继续使用现有 `ChatSession` / `ChatMessage` 保存会话历史。
- Long-term Memory / Preference Memory：写入 PostgreSQL `user_memories`，按用户隔离，并支持创建、读取、更新、删除。
- 自动提取只接受稳定偏好、个人背景和长期目标；一次性问题、临时任务、提醒、API Key、密码、Token、Secret 等敏感信息会被拒绝。
- `AgentTrace.memory` 和兼容字段 `memory_trace` 只记录工程摘要，例如已检索/新建记忆 ID、会话消息数和跳过原因，不保存 API Key 或敏感原文。

## 环境变量
需在仓库根目录或运行环境配置：
- 模型相关：`ARK_API_KEY`、`MODEL`、`BASE_URL`、`EMBEDDER`
- Rerank 相关：`RERANK_MODEL`、`RERANK_BINDING_HOST`、`RERANK_API_KEY`
- Milvus：`MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION`
- Auto-merging：`AUTO_MERGE_ENABLED`、`AUTO_MERGE_THRESHOLD`、`LEAF_RETRIEVE_LEVEL`
- 工具：`AMAP_WEATHER_API`、`AMAP_API_KEY`、`BAIDU_SEARCH_API_KEY`、
  `BAIDU_SEARCH_API_URL`、`DEFAULT_TIMEZONE`
- Memory：`MEMORY_MAX_RETRIEVED`（单次 Agent 请求最多注入的长期记忆数量，默认 `6`）
- Agent Reliability：`MAX_TOOL_RETRIES`、`MAX_TASK_RETRIES`、
  `MAX_AGENT_STEPS`、`MAX_RECOVERY_ATTEMPTS`、`RETRY_BACKOFF_SECONDS`、
  `RETRY_BACKOFF_MAX_SECONDS`、`STUCK_REPEAT_THRESHOLD`、
  `CIRCUIT_BREAKER_ENABLED`、`CIRCUIT_BREAKER_FAILURE_THRESHOLD`、
  `CIRCUIT_BREAKER_RESET_SECONDS`
- RAG Retrieval：`BM25_K1`、`BM25_B`、`HYBRID_DENSE_WEIGHT`、
  `HYBRID_SPARSE_WEIGHT`、`HYBRID_RRF_K`

## API 速览
- `POST /auth/register`：注册并返回 JWT；首个注册用户自动成为管理员。
- `POST /auth/login`：登录并返回 JWT。
- `GET /auth/me`：获取当前用户信息。
- `POST /chat`：聊天（非流式），入参 `message`、`session_id`。
- `POST /chat/stream`：聊天（流式 SSE），入参同上，返回 `text/event-stream`。
- `GET /sessions`：列出当前用户的会话。
- `GET /sessions/{session_id}`：拉取当前用户的会话消息。
- `DELETE /sessions/{session_id}`：删除当前用户的会话。
- `GET /memories`：列出当前用户的长期记忆或偏好记忆，可按 `query`、`memory_type` 过滤。
- `POST /memories`：创建一条 `long_term` 或 `preference` 记忆。
- `PATCH /memories/{memory_id}`：更新当前用户的一条持久化记忆。
- `DELETE /memories/{memory_id}`：删除当前用户的一条持久化记忆。
- `GET /documents`：管理员列出已入库文档及 chunk 数。
- `POST /documents/upload`：管理员上传并向量化 PDF/Word。
- `DELETE /documents/{filename}`：管理员删除指定文档的向量数据。

除注册和登录接口外，所有接口都需要：

```http
Authorization: Bearer <access_token>
```

## 数据库与缓存部署

`docker-compose.yml` 现在同时提供 PostgreSQL、Redis 和 Milvus 依赖：

```bash
docker compose up -d postgres redis etcd minio standalone
uv run uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

服务首次启动时会自动创建数据库表。旧版本的 `data/customer_service_history.json` 和 `data/parent_chunks.json` 不再作为运行时存储；如需迁移历史数据，可执行：

```bash
python backend/migrate_legacy.py --default-password "请替换为临时密码"
```

迁移完成后应要求旧用户立即修改临时密码。

## 流式输出与实时检索过程 — 技术细节

#### 1. 跨线程事件调度（Cross-Thread Event Scheduling）
这是一个解决 **"同步工具阻塞异步事件循环"** 问题的关键架构设计，常用于 Python 异步 Web 服务与 CPU 密集型/IO 密集型任务的混合场景。

**痛点**：
FastAPI 运行在单线程的 asyncio Event Loop 上。为了不阻塞主线程，LangChain 通常将同步工具（如 `search_knowledge_base`）放到 `ThreadPoolExecutor` 中运行。但在子线程中，无法直接访问主线程的 `asyncio.Queue`，且 `asyncio.get_event_loop()` 通常会失败。

**解决方案**：
我们采用了 **"Global Loop Capture + Threadsafe Callback"** 模式：

1.  **Loop 捕获 (Main Thread)**:
    在 Agent 开始生成前，主线程调用 `set_rag_step_queue()`。此时我们捕获当前的运行循环：`_RAG_STEP_LOOP = asyncio.get_running_loop()` 并保存为全局变量。
2.  **跨线程发射 (Worker Thread)**:
    当 RAG 工具在子线程运行时，调用 `emit_rag_step()`。
    函数内部使用 `_RAG_STEP_LOOP.call_soon_threadsafe(queue.put_nowait, step_data)`。
3.  **原理**:
    `call_soon_threadsafe` 是 asyncio 唯一允许从其他线程向 Loop 注入回调的方法。它相当于向主 Loop 的"待办事项箱"投递了一个任务（即 `queue.put_nowait`），主 Loop 会在下一次 tick 立即执行它，从而实现数据的平滑流转。

```python
# 核心代码摘要 (tools/context.py)
def set_rag_step_queue(queue):
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    # 关键：在主线程捕获 Loop
    _RAG_STEP_LOOP = asyncio.get_running_loop()

def emit_rag_step(icon, label):
    # 关键：从子线程安全调度回主 Loop
    if _RAG_STEP_LOOP and not _RAG_STEP_LOOP.is_closed():
        _RAG_STEP_LOOP.call_soon_threadsafe(
            _RAG_STEP_QUEUE.put_nowait, 
            {"icon": icon, "label": label}
        )
```

### 2. 混合检索（Hybrid Search）深度实现
项目并非简单调用 Milvus 接口，而是手动构建了工业级的稀疏-稠密双塔检索：

- **Dense Pathway**: 使用 OpenAI `text-embedding-3-small` 生成 1536 维稠密向量，捕捉语义匹配。
- **Sparse Pathway**:
    - 在 `embedding.py` 中实现了基于 `jieba` 分词的自定义 BM25 算法。
    - 生成 `{word_id: tf_idf_score}` 格式的稀疏向量，模拟 ElasticSearch 的关键词匹配能力。
- **Milvus 融合**:
    - 使用 Milvus 的 `AnnSearchRequest` 同时发起两个请求。
    - **RRFRanker (Reciprocal Rank Fusion)**: 采用 `k=60` 的倒数排名融合算法，将两路召回结果无参数化地合并，避免了加权求和中调节 `alpha` 参数的困难。

### 3. 前端 "Thinking State Machine"
前端 `script.js` 维护了一个微型状态机来处理通过 SSE 传回的复杂混合流：

1.  **Idle**: 等待用户输入。
2.  **Thinking (Initial)**: 收到请求，创建消息气泡，`isThinking=true`，显示默认动画。
3.  **Thinking (Active RAG)**: 收到 `type: rag_step` 事件。
    - 状态机保持 `isThinking=true`。
    - 动态更新 Header 文字（如 "正在重写查询..."）。
    - 向 `ragSteps` 数组追加步骤，触发 Vue 列表渲染。
4.  **Streaming**: 收到第一个 `type: content` 事件。
    - **立即切换**: 设置 `isThinking=false`。
    - 并不销毁气泡，而是隐藏思考 header，开始在同一气泡内追加 Markdown 文本。
    - 这样实现了从"思考"到"回答"的无缝视觉过渡，没有突兀的 UI 抖动。

## 整体架构

```
用户发送消息
    │
    ▼
POST /chat/stream → StreamingResponse(text/event-stream)
    │
    ▼
chat_with_agent_stream()
    │
    ├── 创建统一输出队列 (asyncio.Queue)
    ├── 设置 _RagStepProxy → emit_rag_step() 的输出直接入队
    ├── 启动 _agent_worker 后台任务 (asyncio.create_task)
    │     └── agent.astream(stream_mode="messages") 逐 token 产出
    │           ├── AIMessageChunk (文本) → {"type": "content"} 入队
    │           └── tool_call_chunks (工具调用) → 跳过
    │
    └── 主循环：await output_queue.get() → yield SSE
          ▲
          │ (并发) RAG 工具在线程池中执行
          │ emit_rag_step() → loop.call_soon_threadsafe → 入队
          │ {"type": "rag_step"} 立即从队列取出并推送到前端
```

### 后端实现

#### 1) 流式生成 (`agent.py`)
- 使用 LangGraph `agent.astream(stream_mode="messages")` 获取逐 token 的 `AIMessageChunk`。
- 过滤 `tool_call_chunks`，只转发文本内容给前端。
- **关键设计**：Agent 流式循环运行在 `asyncio.create_task` 后台任务中，主生成器只负责从统一 `output_queue` 取事件并 yield。这样 RAG 步骤在工具执行期间（agent 阻塞等待工具返回时）仍然可以实时推送到前端。

#### 2) 实时 RAG 步骤推送 (`tools/context.py` + `rag_pipeline.py`)
- `emit_rag_step(icon, label, detail)` 通过 `asyncio.get_event_loop().call_soon_threadsafe()` 将步骤从同步线程安全地推送到异步队列。
- `_RagStepProxy` 代理对象将原始 step dict 包装为 `{"type": "rag_step", "step": {...}}` 后放入统一输出队列，**无需额外 relay 任务**。
- `rag_pipeline.py` 在每个关键节点发射步骤：
  - `retrieve_initial` → "正在检索知识库..."
  - `grade_documents` → "正在评估文档相关性..."
  - `rewrite_question` → "正在重写查询..."（含策略选择）
  - `retrieve_expanded` → "使用扩展查询重新检索..."

#### 3) SSE 协议格式
每个事件格式：`data: {JSON}\n\n`，类型字段：
- `content`：文本 token（打字机效果）
- `rag_step`：实时检索步骤（`{icon, label, detail}`）
- `trace`：完整 RAG 追踪信息（回答完成后发送）
- `error`：错误信息
- `[DONE]`：流结束标记

#### 4) StreamingResponse 配置 (`api.py`)
```python
StreamingResponse(
    event_generator(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
    },
)
```

### 前端实现

#### 1) ReadableStream 解析 (`script.js`)
- 使用 `response.body.getReader()` + `TextDecoder` 逐块读取。
- 手动按 `\n\n` 分割 SSE 事件，解析 `data: ` 前缀后的 JSON。
- `content` 事件追加到消息文本；`rag_step` 事件追加到检索步骤数组并同步更新思考状态文字。

#### 2) 思考气泡二合一
- 发送消息后立即创建带 `isThinking: true` 的气泡，显示跳动圆点 + 动态文字。
- 收到 `rag_step` 时，`thinkingLabel` 更新为当前步骤（如"正在检索知识库..."）。
- 收到第一个 `content` token 时，`isThinking = false`，同一气泡无缝切换为正常文本流。
- **不存在两个分离的气泡**，从思考 → 检索 → 回答全程在同一个气泡内完成。

#### 3) Vue 3 响应式注意事项
- 通过 `this.messages[botMsgIdx]` 索引访问（而非缓存对象引用），确保拿到 Vue 的 reactive proxy。
- `ragSteps` 数组通过 `push()` 触发响应式更新。

### 终止功能

#### 前端
- 发送按钮在 `isLoading` 期间切换为红色终止按钮（`v-if/v-else`）。
- 点击调用 `AbortController.abort()`，取消正在进行的 `fetch` 请求。
- 捕获 `AbortError`，在气泡中显示"(已终止回答)"。

#### 后端
- FastAPI 的 `StreamingResponse` 在客户端断开连接（如浏览器触发 `abort()` 或关闭标签页）时，会检测到 socket 断开。
- Python 的生成器协议会向响应生成器抛出 `GeneratorExit` 异常。
- **实现细节**：采用**主动防御式编程**，显式捕获 `GeneratorExit` 并执行 `agent_task.cancel()`。
- **为什么不依赖框架自动取消？**：虽然 Starlette/FastAPI 拥有基于 `BaseHTTPMiddleware` 的级联取消机制（Cascading Cancellation），但在复杂的后台任务结构或特定中间件配置下，取消信号可能延迟或在传递链中丢失。显式调用 `.cancel()` 提供了**确定性的资源回收**保证。
- **即时止损原理**：`agent_task.cancel()` 会立即在任务挂起点注入 `asyncio.CancelledError`。对于流式 LLM 请求，这会触发 `httpx` 关闭 TCP 连接。服务端（OpenAI 等）检测到 client 掉线后会立即停止推理，从而实现**真正的 Token 节省**。

## 更新日志

## 2026-03-21 后端服务建设升级

- 新增注册、登录、JWT Token 和管理员权限控制。
- 聊天历史从本地 JSON 迁移到 PostgreSQL，按认证用户隔离会话。
- 父级分块从 `parent_chunks.json` 迁移到 PostgreSQL。
- 引入 Redis 缓存会话列表和消息读取结果，缓存不可用时自动回源数据库。
- 聊天和会话接口不再接收前端传入的 `user_id`，用户身份从 `Authorization: Bearer <token>` 获取。
- 文档列表、上传、删除接口仅允许管理员访问。
- 密码使用 PBKDF2-SHA256，同时兼容历史 bcrypt 哈希校验。
- 
### 2026-03-13 三级分块与 Auto-merging 升级
- 新增三级滑动窗口分块（L1/L2/L3），并为分块写入层级元数据。
- 存储策略调整为 Leaf-only：仅 L3 叶子块写入 Milvus，L1/L2 写入本地 DocStore。
- Auto-merging 改为从 DocStore 拉取父块，减少向量冗余存储。
- 思考链路新增三级检索与自动合并步骤事件。
- `rag_trace` 新增 `leaf_retrieve_level` 与 `auto_merge_*` 字段，且历史会话读取同样保留这些字段。

### 2026-02-19 RAG 实时思考链路修复
- **问题**：Agent 在执行同步工具（如 `search_knowledge_base`）时，由于运行在线程池中，无法正确获取主线程的 asyncio 事件循环，导致 `emit_rag_step` 事件丢失，前端"思考中"气泡一直静止。
- **修复**：
  1. **Backend (`tools/context.py`)**：在 `set_rag_step_queue` 中显式捕获主线程的 `loop`。
  2. **Backend (`tools/context.py`)**：更新 `emit_rag_step` 使用捕获的 `_RAG_STEP_LOOP.call_soon_threadsafe` 跨线程调度事件。
  3. **Frontend (`script.js`)**：在发送消息时初始化空的 `ragSteps: []` 数组，确保 Vue 响应式系统能立即追踪后续的 push 操作。
- **效果**：用户提问后，思考气泡内实时跳动显示检索步骤（如"🔍 正在检索知识库..." -> "📊 正在评估文档相关性..."），不再只有静态的"正在思考中..."。
