# HajiMind 项目技术文档

> HajiMind（谐音「哈基米」）——基于 FastAPI + LangGraph 的**流式、可观测 RAG 智能体**。

---

## 一句话定位

HajiMind：基于 FastAPI + LangGraph 的流式、可观测 RAG 智能体，实现了混合检索、三级分块 Auto-merging、查询重写路由，以及在同步工具执行期间把检索过程实时推送到前端的跨线程事件调度。

> 当前版本新增后端服务建设：JWT 认证、管理员权限、PostgreSQL 持久化、Redis 缓存，以及 Token 驱动的 API。

---

## 技术栈

- **后端**：FastAPI、LangChain Agents、LangGraph、Pydantic、Uvicorn、SQLAlchemy
- **认证与数据服务**：JWT、PBKDF2-SHA256、PostgreSQL、Redis
- **向量与检索**：Milvus（HNSW 稠密索引 + SPARSE_INVERTED_INDEX 稀疏索引）、RRF 融合、Rerank 精排
- **嵌入与稀疏**：OpenAI 兼容嵌入 API 生成稠密向量；手写 BM25 生成稀疏向量
- **前端**：Vue 3 (CDN)、marked、highlight.js、纯静态部署
- **运行形态**：FastAPI 后端 + 纯前端单页 + PostgreSQL + Redis + Milvus（Docker Compose 部署依赖）

---

## 核心亮点（按含金量排序）

### 1. 流式输出 + 实时 RAG 过程可视化（最硬核）

**痛点**：FastAPI 跑在单线程 asyncio loop 上；LangChain 的同步工具 `search_knowledge_base` 被丢到线程池执行，子线程里 `asyncio.get_running_loop()` 会失败，拿不到主线程的 `asyncio.Queue`，导致检索步骤推不出去，前端「思考中」气泡一直静止。

**解法**：`set_rag_step_queue()` 在**主线程捕获** loop 存为全局 `_RAG_STEP_LOOP`；子线程里 `emit_rag_step()` 用 `loop.call_soon_threadsafe(queue.put_nowait, step)` 把事件**安全注入主 loop**（这是 asyncio 唯一允许跨线程投递回调的 API）。

**架构**：`chat_with_agent_stream` 建一个统一 `asyncio.Queue`，`agent.astream(stream_mode="messages")` 跑在 `asyncio.create_task` 后台任务里逐 token 产文本入队；主循环只 `await queue.get()` → `yield` SSE。

**关键收益**：工具执行期间（agent 阻塞等工具返回时）检索步骤仍能实时推送，而不是等工具跑完才一次性显示。

> 相关代码：`backend/tools.py`（set_rag_step_queue / emit_rag_step）、`backend/agent.py`（chat_with_agent_stream）

### 2. 回答终止 —— 确定性资源回收

前端 `AbortController.abort()` → fetch 断开 → 服务端 socket 断 → Python 生成器收到 `GeneratorExit` → **显式** `agent_task.cancel()` → `httpx` 关闭 TCP → 上游 LLM 检测到掉线停止推理，**真正节省 token**。
1
**亮点话术**：不依赖框架的级联取消（可能延迟或丢信号），主动捕获 `GeneratorExit` 保证确定性资源回收。

> 相关代码：`backend/agent.py`（GeneratorExit 捕获与 agent_task.cancel）

### 3. 混合检索 Hybrid Search + RRF

- Milvus 单 collection 双向量字段：`dense_embedding`（FLOAT_VECTOR / HNSW / 内积 IP）+ `sparse_embedding`（SPARSE_FLOAT_VECTOR / SPARSE_INVERTED_INDEX）。
- `hybrid_search` 发两个 `AnnSearchRequest`（各取 `top_k*2` 候选），用 `RRFRanker(k=60)` **无参融合**——避免了加权求和调 alpha 参数的痛点。
- 稀疏向量是**手写 BM25**（k1=1.5, b=0.75），自建 vocab + 文档频率 + IDF，输出 `{word_id: score}` 格式的稀疏向量。
- **降级机制**：hybrid 调用失败自动降级为纯稠密 `dense_retrieve`，提升稳定性。

> 相关代码：`backend/milvus_client.py`（hybrid_retrieve / dense_retrieve）、`backend/embedding.py`（BM25）

### 4. 三级分块 + Auto-merging（Leaf-only 存储）

- 三层 `RecursiveCharacterTextSplitter`：L1(≥1200) → L2(≥600) → L3(≥300) 滑动窗口，中文标点优先切分，写入层级元数据 `chunk_id / parent_chunk_id / root_chunk_id / chunk_level`。
- **存储策略**：只有 L3 叶子块进 Milvus（带稠密+稀疏向量），L1/L2 父块进本地 DocStore → 减少向量冗余。
- **检索**：`filter chunk_level==3` 只召回叶子块；`_auto_merge_documents` 两段合并 L3→L2→L1，当同一父块下命中叶子数 ≥ 阈值(2) 时，从 DocStore 拉取父块替换子块，取 max score → 兼顾精准召回与上下文完整。

> 相关代码：`backend/document_loader.py`、`backend/rag_utils.py`、`backend/parent_chunk_store.py`

### 5. 查询重写路由（LangGraph 状态机）

- `StateGraph`：`retrieve_initial → grade_documents →(条件边)→ [END | rewrite_question → retrieve_expanded → END]`。
- `grade_documents`：LLM `with_structured_output(GradeDocuments)` 输出二值 yes/no，门控是否需要重写。
- `rewrite_question`：router LLM 结构化输出选择策略 `step_back / hyde / complex`：
  - **Step-Back**：抽象出退步问题 + 生成答案，拼进 expanded_query。
  - **HyDE**：生成假设性文档再检索。
  - **complex**：两者都做，多路召回后去重。

> 相关代码：`backend/rag_pipeline.py`、`backend/rag_utils.py`（step_back_expand / generate_hypothetical_document）

### 6. Rerank 精排

- rerank 候选 `candidate_k = top_k * 3`；调用外部 Rerank API（`/v1/rerank`，Jina 兼容协议），返回 `relevance_score` → `rerank_score`；未配置时自动跳过精排。

> 相关代码：`backend/rag_utils.py`（_rerank_documents）

### 7. 会话摘要记忆 + 全链路可观测 trace

- 消息数 >50 时把前 40 条用 LLM 摘要成 `SystemMessage` 注入，控制 token 同时保留长期上下文。
- `rag_trace`（评分/路由决策、重写策略与内容、两轮召回结果、三级检索与合并信息、检索 score 与 rerank_score）随消息落盘，历史回放完整保留检索过程。

> 相关代码：`backend/agent.py`（ConversationStorage / summarize_old_messages）

---

## 核心流程（端到端）

1. 用户在前端输入问题，调用 `POST /chat/stream`（流式 SSE）。
2. FastAPI 返回 `StreamingResponse(media_type="text/event-stream")`。
3. LangChain Agent 根据问题类型决定是否调工具：天气 → `get_current_weather`；知识问答 → `search_knowledge_base`。
4. 命中知识库工具则进入 LangGraph RAG 工作流，各阶段通过 `emit_rag_step()` 实时推送到前端。
5. 检索结果与 RAG Trace 一起返回，Agent 逐 token 流式生成最终回答。
6. 前端 ReadableStream 逐块解析 SSE，打字机效果实时渲染。
7. 消息写入 PostgreSQL，Redis 缓存会话读取结果，支持历史会话回放。

---

## 简历可用 Bullet（中文）

- 基于 **FastAPI + LangGraph** 构建流式 RAG 智能体，用 `asyncio.Queue` + 后台任务 + `call_soon_threadsafe` **跨线程事件调度**，解决同步工具阻塞异步事件循环的问题，实现工具执行期间检索过程（检索 → 评分 → 重写）的**实时 SSE 推送**。
- 设计 **Milvus 稠密 + 稀疏双塔混合检索**，手写 BM25 稀疏向量，`RRFRanker(k=60)` 无参融合，并实现 hybrid → dense 自动降级。
- 实现 **三级滑窗分块 + Auto-merging**，Leaf-only 向量存储（仅叶子块入库，父块存 DocStore），命中密度达阈值时回取父块，降低向量冗余同时保留上下文。
- 用 **LangGraph 状态机**实现相关性门控 + 查询重写路由（Step-Back / HyDE），LLM 结构化输出驱动条件分支。
- 支持 `AbortController` → `GeneratorExit` → `task.cancel()` 的**确定性中断**，客户端断连即停止上游推理，节省 token。

---

## 简历可用 Bullet（English）

- Built a streaming RAG agent with **FastAPI + LangGraph**; solved the "sync tool blocking the async event loop" problem via **cross-thread event scheduling** (`asyncio.Queue` + background task + `call_soon_threadsafe`), enabling **real-time SSE push** of the retrieval process (retrieve → grade → rewrite) while tools are still executing.
- Designed **dense + sparse hybrid retrieval on Milvus** with a hand-written BM25 sparse encoder, fused via parameter-free `RRFRanker(k=60)`, with automatic hybrid → dense fallback.
- Implemented **three-level sliding-window chunking + Auto-merging** with leaf-only vector storage (only leaf chunks indexed in Milvus, parent chunks in a local DocStore); parents are merged back when hit density crosses a threshold, cutting vector redundancy while preserving context.
- Built a **LangGraph state machine** for relevance gating and query-rewrite routing (Step-Back / HyDE), driven by LLM structured output on conditional edges.
- Added **deterministic cancellation** (`AbortController` → `GeneratorExit` → `task.cancel()`): a client disconnect stops upstream LLM inference and saves tokens.

---

## 注意事项（口径校准，务必核对）

原 README 有 3 处描述与实际代码不符，写进简历/面试前请按下表校准：

| 原 README 说 | 实际代码 | 建议口径 |
|---|---|---|
| BM25 用 jieba 分词 | 自定义中英混合分词（中文按单字、英文正则单词），无 jieba | 「手写 BM25 + 自研中英文分词」 |
| text-embedding-3-small / 1536 维 | 可配置 OpenAI 兼容嵌入端点，Milvus 默认 dense_dim=2560 | 「OpenAI 兼容嵌入 API（2560 维）」或写实际使用的模型 |
| Jina Rerank | 通用 `/v1/rerank` 接口（Jina 兼容协议），未配置自动跳过 | 「Rerank 精排（Jina 兼容协议），支持缺省降级」 |

另：BM25 的 IDF 是**运行时增量累积**（`vocab` / `doc_freq` 随文本填充），非离线全量 `fit_corpus`。若被问「IDF 语料统计怎么保证」，需能解释增量累积逻辑（建议核对入库路径是否调用 `fit_corpus`）。

---

## 目录与架构

**后端 `backend/`**

- `app.py`：FastAPI 入口、CORS、静态资源挂载
- `api.py`：聊天、会话管理、文档管理接口
- `agent.py`：LangChain Agent、会话存储、摘要逻辑、流式生成
- `tools.py`：天气查询、知识库检索工具、跨线程事件调度
- `embedding.py`：稠密向量 API 调用 + BM25 稀疏向量生成
- `document_loader.py`：PDF/Word 加载与三级分块
- `parent_chunk_store.py`：父级分块 DocStore（Auto-merging 回取父块）
- `milvus_writer.py`：向量写入（稠密 + 稀疏）
- `milvus_client.py`：Milvus 集合定义、混合检索
- `rag_pipeline.py`：LangGraph RAG 状态机
- `rag_utils.py`：检索、Auto-merging、Rerank、Step-Back、HyDE
- `schemas.py`：Pydantic 请求/响应模型

**前端 `frontend/`**

- `index.html` + `script.js` + `style.css`：Vue 3 + marked + highlight.js，聊天 / 历史会话 / 文档上传删除界面

**数据与服务存储**

- `data/documents/`：上传文档原文件
- PostgreSQL：用户、会话、消息和 L1/L2 父级分块
- Redis：会话列表和消息读取缓存

**新增后端模块**

- `database.py`：PostgreSQL 连接、Session 和表初始化
- `models.py`：用户、会话、消息、父级分块模型
- `auth.py`：PBKDF2/bcrypt 密码校验、JWT 和权限依赖
- `cache.py`：Redis 缓存，Redis 不可用时自动降级
- `migrate_legacy.py`：一次性导入旧 JSON 会话和父级分块数据

## 2026-03-21 后端服务建设升级

### 认证与权限

- 新增 `POST /auth/register`、`POST /auth/login` 和 `GET /auth/me`。
- 注册后返回 JWT，后续请求使用 `Authorization: Bearer <access_token>`。
- 第一个注册用户自动成为管理员，后续注册用户默认为普通用户。
- 新密码使用 PBKDF2-SHA256；校验逻辑兼容历史 bcrypt 哈希。
- `/documents`、`/documents/upload` 和 `/documents/{filename}` 仅允许管理员访问。

### PostgreSQL 持久化

- `users`：用户账号、密码哈希、角色和启用状态。
- `chat_sessions`：按用户隔离的会话信息。
- `chat_messages`：消息内容、时间戳和 `rag_trace` JSON。
- `parent_chunks`：L1/L2 父级分块和层级元数据。
- 聊天和父块不再以本地 JSON 作为运行时主存储。

### Redis 缓存

- 缓存当前用户的会话列表。
- 缓存单个会话的消息读取结果。
- 会话写入和删除时主动删除相关缓存。
- Redis 不可用时自动回源 PostgreSQL，不影响主要功能。

### API 变化

- `ChatRequest` 不再包含 `user_id`，只保留 `message` 和 `session_id`。
- 会话接口从 `/sessions/{user_id}` 改为 `/sessions`，用户身份从 JWT 获取。
- 会话详情从 `/sessions/{user_id}/{session_id}` 改为 `/sessions/{session_id}`。
- 前端登录后统一通过 `authFetch` 自动添加 Token。

### 生产化注意事项

- `JWT_SECRET_KEY` 必须使用随机长密钥，不能使用默认值。
- 生产环境应使用 Alembic 管理数据库迁移，而不是只依赖 `create_all`。
- 旧的 `customer_service_history.json` 和 `parent_chunks.json` 可使用 `python backend/migrate_legacy.py --default-password "<temporary-password>"` 一次性导入数据库。
- 当前“首个用户自动成为管理员”适合本地开发，生产环境应改为初始化管理员命令或环境变量。
- 当前 Redis 为可选缓存，若要严格保证缓存一致性，应增加版本号、失效策略和监控。
