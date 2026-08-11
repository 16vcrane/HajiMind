# HajiMind AI Agent 工程师面试准备指南

## 一、项目定位

HajiMind 是一个面向知识库问答和多工具任务处理的流式 AI Agent 系统。

它的价值不只是“接了 LangChain 和向量库”，而是把一个 Demo 型 RAG 项目升级成了具备以下能力的工程化服务：

- 多用户认证、权限控制和数据隔离。
- Agent Router：自动判断问题应该走 LLM、RAG、Web、天气、日历还是 Planner。
- Multi-step Planner：复杂任务拆解成结构化 Plan，并按依赖关系串行或并行执行。
- 统一工具基础设施：工具输入校验、超时、重试、Trace、错误返回和熔断。
- LangGraph RAG：初次召回、相关性评分、查询重写、二次召回。
- Hybrid Retrieval：Milvus Dense + BM25 Sparse + RRF / WeightedRanker + Rerank。
- RAG Deep Engineering：结构化文档解析、语义分块、三级分块、Auto-merging。
- 多文档冲突检测：不强行裁决，保留来源、观点、证据和时间。
- 结构化 Memory：会话记忆、长期记忆、偏好记忆，按用户隔离并过滤敏感信息。
- SSE 流式输出和实时 Agent Workflow 可视化。
- Agent Reliability：重试、指数退避、最大步数、死循环检测、恢复策略和 Circuit Breaker。
- Agent Trace：记录 Router、Planner、Tool、Task、RAG、Memory、latency、error 和最终状态。

一句话概括：

> 我把 HajiMind 从单纯的 RAG 聊天项目升级成了一个流式、可观测、可恢复、多用户隔离的 AI Agent 服务。它能先路由任务，再对复杂问题做多步骤规划，通过统一工具层执行 RAG、搜索、天气、日历等能力，并把全过程 Trace、Memory 和失败恢复信息落盘。

## 二、30 秒项目介绍

HajiMind 是我做的一个工程化 AI Agent 项目。后端基于 FastAPI、LangChain 和 LangGraph，前端用 Vue 3 和 SSE 实现流式交互。系统支持 JWT 登录、管理员文档管理、PostgreSQL 持久化和 Redis 缓存。

新版本重点加入了 Agent Router 和 Multi-step Planner。Router 会把问题路由到 LLM、RAG、Web Search、天气、日历或 Planner；复杂任务由 Planner 生成结构化任务图，按依赖并行或串行执行。RAG 部分使用 Milvus Dense + BM25 Sparse 混合检索、RRF 融合、Rerank、三级分块 Auto-merging，并支持多文档冲突提示。整个链路会生成 Agent Trace，记录路由、工具、任务、RAG、Memory、延迟和错误，便于调试和展示。

## 三、1 分钟项目介绍

我开发了 HajiMind，一个面向知识库问答和复杂任务处理的多用户流式 AI Agent。

后端使用 FastAPI，用户通过注册和登录获取 JWT，服务端从 Token 中解析用户身份，不再信任前端传入的 `user_id`。普通用户只能管理自己的会话和记忆，管理员可以上传、查看和删除知识库文档。会话、消息、父级分块和用户记忆存储在 PostgreSQL，Redis 作为 Cache-Aside 缓存。

Agent 层面我新增了 Router 和 Planner。Router 使用结构化 LLM Output 判断问题类型，决定走普通 LLM、知识库 RAG、Web Search、天气、日历还是 Planner。复杂任务进入 Multi-step Planner，Planner 会生成包含依赖关系的 Plan，再由 TaskExecutor 按依赖图并行或串行执行工具，最后用 synthesis 任务综合结果。

RAG 内部用 LangGraph 显式编排，流程包括初次召回、相关性评分、查询重写和二次召回。检索上使用 Milvus 的 Dense + Sparse Hybrid Search，自定义 BM25 稀疏向量，支持 RRF 或 WeightedRanker 融合，再接可选 Rerank。文档入库时先做结构化解析，识别标题、段落、列表、代码、表格和图片元数据，再做语义分块和 L1/L2/L3 三级分块；只有 L3 叶子块入 Milvus，L1/L2 父块存 PostgreSQL，检索后通过 Auto-merging 回取父块。

工程上我重点解决了两个问题：第一，流式输出时同步工具运行在线程池中，不能直接向 FastAPI 的 asyncio 事件循环推送状态，所以我通过 `asyncio.Queue` 和 `call_soon_threadsafe` 实现工具执行期间的实时 SSE 进度推送。第二，Agent 执行容易遇到工具失败、重复调用和死循环，所以我加了 bounded retry、指数退避、Circuit Breaker、最大执行步数、stuck detection 和恢复策略。最终所有 Router、Planner、Tool、Task、RAG 和 Memory 信息都会进入 Agent Trace，前端可以展示完整工作流。

## 四、简历 Bullet

中文版本：

- 基于 FastAPI、LangChain、LangGraph 构建多用户流式 AI Agent，支持 JWT/RBAC、PostgreSQL 持久化、Redis 缓存和 SSE 实时输出。
- 设计 Agent Router，使用结构化 LLM Output 将问题路由到 LLM、RAG、Web Search、天气、日历或 Multi-step Planner，并在失败时回退旧 Agent。
- 实现 Multi-step Planner 和 TaskExecutor，将复杂任务拆解为带依赖的结构化 Plan，支持并行任务、串行依赖、synthesis 汇总、任务取消、重试和执行 Trace。
- 封装统一 Tool Infrastructure，工具统一继承 `BaseTool`，支持输入校验、timeout、retry、错误结果、Trace、Registry 和 Circuit Breaker。
- 搭建 Milvus Dense + BM25 Sparse 混合检索，支持 RRF/WeightedRanker 融合、Rerank 精排、Hybrid 降级和离线 RAG 指标评估。
- 实现结构化文档解析和语义分块，识别 Heading、List、Code、Table、Image 元数据，并结合三级分块和 Auto-merging 降低向量冗余、保留上下文。
- 增加 Agent Trace 和 Agent Workflow Panel，统一记录 Router、Planner、Tool、Task、RAG、Memory、latency、error 和 final status。
- 实现结构化长期记忆和偏好记忆，按用户隔离，过滤临时任务和 API Key、Token、password 等敏感信息。

English version:

- Built a multi-user streaming AI Agent with FastAPI, LangChain and LangGraph, including JWT/RBAC, PostgreSQL persistence, Redis cache and SSE streaming.
- Designed an Agent Router with structured LLM output to route requests to LLM, RAG, Web Search, weather, calendar or a multi-step Planner, with fallback to the legacy Agent.
- Implemented a dependency-aware Multi-step Planner and TaskExecutor supporting parallel tasks, sequential dependencies, synthesis, cancellation, retries and execution traces.
- Built a unified Tool Infrastructure with input validation, timeout, retry, error normalization, trace logging, registry-based dispatch and circuit breaking.
- Implemented Milvus dense + BM25 sparse hybrid retrieval with RRF/WeightedRanker fusion, optional reranking, fallback paths and offline RAG retrieval evaluation.
- Added structure-aware parsing and semantic chunking for headings, lists, code, tables and image metadata, combined with three-level chunking and Auto-merging.
- Added Agent Trace and workflow visualization covering Router, Planner, Tool, Task, RAG, Memory, latency, errors and final status.

## 五、系统架构怎么讲

建议按“从外到内”的顺序讲：

```text
Vue 前端
  -> FastAPI
  -> JWT Auth / RBAC
  -> Agent Router
  -> LLM / RAG / Tool Agent / Multi-step Planner
  -> ToolRegistry
  -> Milvus / PostgreSQL / Redis / 外部 API
  -> Agent Trace + SSE Workflow
```

讲解重点：

- 前端不是静态等待结果，而是实时消费 `content`、`rag_step`、`router_step`、`planner_step`、`tool_start`、`tool_result`、`task_start`、`task_result`、`trace` 等 SSE 事件。
- Router 是任务入口，减少所有问题都交给一个 Agent 的不可控性。
- Planner 只处理复杂或多工具任务，简单问题直接走 LLM 或单工具路径，控制成本。
- RAG 内部用 LangGraph 做确定性流程，不让 Agent 自由反复调用检索工具。
- Trace 是工程化关键，不只是调试日志，而是前端可视化和历史回放的数据基础。

## 六、核心模块回答口径

### 6.1 Router 怎么设计

回答：

> Router 使用结构化 LLM Output 输出 `RouterDecision`，包含 intent、complexity 和多个 requires 标志。它不是简单分类器，因为一个问题可能同时需要知识库、网络和日历。只要任务复杂、需要 planning，或多个 requires 同时为 true，就进入 Planner；否则走对应单一路由。Router 失败时会回退到旧 Agent，保证可用性。

可举例：

- “解释一下 RAG” -> `llm`
- “查我的知识库里项目架构” -> `rag`
- “搜索最新 LangGraph 资料” -> `baidu`
- “上海今天的天气” -> `life_service`
- “2026-08-11 是星期几” -> `calendar`
- “结合我的文档和最新资料制定学习计划” -> `planner`

### 6.2 Planner 怎么执行

回答：

> Planner 先生成结构化 `Plan`，每个 `Task` 有 id、description、tool、dependencies、input、timeout 和 max_retries。TaskExecutor 每轮找出依赖已完成的任务，无依赖任务并行执行，有依赖任务串行等待。工具结果统一进入 `task_results`，最后由 synthesis 任务汇总。每个任务都有 status、attempts、error 和 trace，用户中断时通过 cancel_event 取消仍在执行的任务。

强调点：

- 不是让模型直接长篇规划，而是让模型输出可验证的数据结构。
- Plan 会校验任务 ID 唯一、依赖存在、无环。
- Planner State 能还原整个执行过程。

### 6.3 为什么保留 LangChain Agent

回答：

> 我没有把所有东西都塞进 Planner。旧 Agent 负责兼容原有天气和知识库工具选择；Router 负责入口分流；Planner 只接管复杂、多工具、多步骤任务；RAG 内部仍由 LangGraph 控制。这样职责边界清楚，简单问题不会被过度编排，复杂问题也不会让普通 Agent 失控循环。

### 6.4 Tool Infrastructure 解决什么

回答：

> 以前工具如果只是函数，超时、重试、错误格式、Trace 和健康状态都很难统一。现在所有工具继承 `BaseTool`，通过 `ToolRegistry` 注册和执行，返回统一 `ToolResult`。这样 Planner 不需要关心每个工具的异常细节，只看 status、data、error 和 trace。Registry 还集成 Circuit Breaker，工具连续失败时短期熔断，避免 Agent 反复调用坏工具。

### 6.5 RAG 为什么用 LangGraph

回答：

> RAG 是确定性很强的流程：召回、评分、重写、二次召回、生成。用 LangGraph 可以明确状态、条件边和结束条件，比让 Agent 自由决定更可控，也更容易记录 Trace 和测试。Agent 只负责判断是否进入知识库工具，RAG 内部不完全交给 Agent 自由发挥。

## 七、RAG 技术细节怎么讲

### 7.1 文档入库

```text
上传 PDF / Word
  -> 结构化解析 Heading / Paragraph / List / Code / Table / Image
  -> 语义分块
  -> L1/L2/L3 三级分块
  -> L1/L2 写 PostgreSQL parent_chunks
  -> L3 生成 Dense + Sparse 向量
  -> L3 写 Milvus
```

回答重点：

- 代码块保持完整。
- 表格转换为结构化文本。
- 图片只记录 caption、source、surrounding_text 和 `multimodal_embedding=false`，不夸大成已支持多模态。
- 结构化解析失败时回退递归字符分块。

### 7.2 Hybrid Search

回答：

> Dense 负责语义相似，Sparse 负责关键词、编号、专有名词。Milvus 中同一个 collection 存 `dense_embedding` 和 `sparse_embedding` 两个字段，Hybrid Search 同时发起两路 AnnSearchRequest，再用 RRF 或 WeightedRanker 融合。RRF 不依赖两路分数同尺度，早期更稳；WeightedRanker 适合后续用评测集调 dense/sparse 权重。

注意口径：

- 不要说用了 jieba，当前是自定义中英文分词。
- 不要固定说 `text-embedding-3-small / 1536 维`，实际是 OpenAI 兼容嵌入端点，维度由配置和模型决定，README 模板里 `DENSE_DIM=2560`。

### 7.3 Auto-merging

回答：

> Milvus 只存 L3 叶子块，保证召回粒度细。检索后如果同一父块下多个 L3 被命中，说明这个父块整体相关，就从 PostgreSQL 回取 L2 或 L1 父块替换子块。这样既避免所有父块都向量化造成冗余，也能在回答时拿到完整上下文。

### 7.4 查询重写

回答：

> 初次召回后用结构化 Grader 判断相关性，输出 yes/no。如果不足，就进入 rewrite_question。重写支持 Step-Back、HyDE 和 complex。Step-Back 把具体问题抽象成上位问题，HyDE 生成假设性答案文档再检索，complex 则两者结合并去重。

### 7.5 多文档冲突

回答：

> 冲突检测只在不同来源对相同主张出现不同取值时触发。系统不会自动选择谁对，而是在工具结果和 Trace 中保留来源、观点、证据和时间，并提示“知识库中存在来源冲突”。这样比强行生成单一答案更符合可解释 RAG。

## 八、流式输出和异步工程

### 8.1 为什么需要 `asyncio.Queue`

回答：

> 流式文本和工具进度来自不同路径。LLM token 来自 Agent 后台任务，RAG 和 Planner 工具进度可能来自线程池或异步任务。如果主生成器只等待 Agent 返回，工具执行期间前端看不到任何状态。所以我用统一 `asyncio.Queue` 汇聚 content、rag_step、planner_step、tool_result 等事件，主生成器只负责从队列取事件并 yield SSE。

### 8.2 为什么用 `call_soon_threadsafe`

回答：

> LangChain 的同步工具可能在线程池里运行，子线程不能直接操作主线程的 asyncio Queue，也拿不到 running loop。所以在主线程捕获 event loop，工具线程发进度时用 `loop.call_soon_threadsafe(queue.put_nowait, step)` 把事件安全投递回主 loop。这解决了同步工具阻塞异步事件循环时，前端进度无法实时显示的问题。

### 8.3 中断如何真正省 token

回答：

> 前端用 `AbortController.abort()` 断开 fetch，后端 StreamingResponse 的生成器收到 `GeneratorExit`，显式执行 `agent_task.cancel()`。这会在异步任务挂起点注入取消异常，并关闭上游 HTTP 流式连接。目标不是只改变按钮状态，而是让上游模型推理也停止，减少无效 token 消耗。

## 九、Memory 怎么讲

回答：

> Memory 分四层。Working Memory 只在当前请求内记录问题、路由和检索上下文；Conversation Memory 用 PostgreSQL 保存会话消息；Long-term Memory 和 Preference Memory 写入 `user_memories`，按用户隔离。提取策略比较保守，只接受稳定偏好、个人背景和长期目标，拒绝临时任务、提醒、API Key、Token、password 等敏感信息。Trace 里只记录 memory id 和跳过原因，不保存敏感原文。

可以举例：

- “以后请用中文回答” -> Preference Memory。
- “我正在准备 AI Agent 工程师面试” -> Long-term Memory。
- “提醒我明天交作业” -> 跳过，因为是临时任务。
- “我的 API key 是...” -> 跳过，因为是敏感信息。

## 十、Reliability 怎么讲

回答：

> Agent 最大的问题是失败不可控和循环不可控。我加了三层保护：工具层有 timeout、retry、指数退避和 Circuit Breaker；Planner 层有 task retry、task timeout、依赖失败跳过和 cancel；执行 Trace 层有 stuck detection，检测重复工具调用、重复 query、重复任务、连续工具失败和 Planner 无进展。恢复策略按顺序尝试修改 query、降低 top_k、Web Search 回退 RAG、跳过当前任务，最后给用户可见失败说明。

关键配置：

- `MAX_TOOL_RETRIES`
- `MAX_TASK_RETRIES`
- `MAX_AGENT_STEPS`
- `MAX_RECOVERY_ATTEMPTS`
- `STUCK_REPEAT_THRESHOLD`
- `CIRCUIT_BREAKER_*`

## 十一、Agent Trace 怎么讲

回答：

> Agent Trace 是新版本的统一观测模型。它不是简单日志，而是每次请求的结构化执行报告。里面包括 request_id、session_id、user_id、intent、complexity、route、plan、tool_calls、rag_trace、memory、task_results、task_traces、recovery_events、latency、errors 和 final_status。Trace 会过滤 API Key、Token、password、secret 等敏感字段。前端的 Agent Workflow Panel 就是基于这些事件和最终 Trace 展示执行过程。

常见 final status：

- `success`
- `partial_success`
- `error`
- `cancelled`

## 十二、认证和数据隔离

### 12.1 为什么移除前端 `user_id`

回答：

> 旧方案如果请求体能传 `user_id`，用户可以伪造别人的 ID 读取或删除会话。现在用户身份只来自服务端验证过的 JWT，所有会话查询都带上 `ChatSession.user_id == current_user.id`，文档管理接口还要 `require_admin`。这是多用户服务最基本的安全边界。

### 12.2 为什么 Redis 不能替代 PostgreSQL

回答：

> Redis 在这里是缓存，不是事实来源。聊天历史、RAG Trace、用户记忆和父级分块需要可靠持久化，所以存在 PostgreSQL。Redis 采用 Cache-Aside，只缓存会话列表和会话详情；Redis 挂了会回源数据库，只影响性能，不影响数据正确性。

## 十三、面试高频问题

### Q1：你这个项目和普通 RAG Demo 的区别是什么？

回答：

> 普通 RAG Demo 通常只有上传文档、向量检索和问答。HajiMind 增加了多用户认证、权限隔离、Router、Planner、统一工具层、可靠性控制、结构化 Memory、Agent Trace、离线评估和前端工作流可视化。它更接近一个可维护的 AI Agent 服务，而不是一次性 Demo。

### Q2：为什么需要 Router，不直接让 Agent 自己选工具？

回答：

> 直接让 Agent 面对所有工具，工具多了以后选择不稳定，也更容易重复调用。Router 把入口决策结构化，先判断任务类型和复杂度，简单问题直接走便宜路径，复杂或多工具任务再进入 Planner。这样能降低成本、提高可控性，也方便 Trace。

### Q3：Planner 和 Agent 的区别是什么？

回答：

> Agent 更适合开放式判断和工具选择；Planner 更适合复杂任务拆解和依赖执行。Planner 输出结构化 Plan，并由程序控制并行、串行、重试、超时和取消。它不是让模型自由长推理，而是把模型的规划能力约束在可校验的数据结构里。

### Q4：为什么 RAG 内部还要 LangGraph？

回答：

> RAG 内部流程确定性强，用 LangGraph 能明确节点和条件边：初次召回、评分、必要时重写、二次召回。这样可控、可测、可观测，也避免 Agent 反复调用检索工具。

### Q5：Hybrid Search 为什么比纯 Dense 更好？

回答：

> Dense 对语义相似更好，但对编号、专有名词、精确关键词不稳定；BM25 Sparse 对关键词更敏感。Hybrid 结合两者，RRF 通过排名融合解决两路分数尺度不同的问题，初期更稳。后续可以用离线评测集决定是否改用 WeightedRanker 和调整 dense/sparse 权重。

### Q6：Rerank 挂了怎么办？

回答：

> Rerank 是可选增强，不是主链路强依赖。未配置或调用失败时保留原始召回结果，并在 Trace 记录 rerank 状态和错误。这样精排服务故障不会导致整个问答失败。

### Q7：Web Search 挂了怎么办？

回答：

> Planner 的恢复策略会先修改 query、降低 top_k；如果 Web Search 仍不可用，会尝试回退到 knowledge_base/RAG。超过恢复次数后返回用户可见的稳定失败说明，而不是无限重试。

### Q8：如何评估 RAG 效果？

回答：

> 项目里有 `eval/` 离线评估，数据集支持 gold 和 synthetic。检索指标包括 Recall@K、Precision@K、MRR、NDCG、Hit Rate。应至少对比 Dense only、BM25 only、Hybrid、Hybrid + Rerank、Hybrid + Rewrite。生成侧再看答案正确性、Faithfulness、延迟和 token 消耗。

### Q9：如何保证 Trace 不泄露敏感信息？

回答：

> Trace 只保存工程摘要。`agent_observability.py` 会对 `api_key`、`authorization`、`access_token`、`token`、`password`、`secret`、`key` 等字段做 redaction，并限制摘要长度。Memory 也拒绝保存 API Key、Token、password 等敏感内容。

### Q10：为什么说当前还没完全生产化？

回答：

> 我会主动说明几个风险：数据库迁移还应引入 Alembic；首个用户自动成为管理员只适合 Demo；BM25 词表和统计需要持久化或接入成熟搜索引擎；部分兼容链路仍有模块级上下文，需要继续推进请求级上下文；真实百度搜索 Provider 还需要明确供应商 API 契约。

## 十四、必须主动说明的风险

### 14.1 数据库迁移

当前服务启动时会自动建表，生产应使用 Alembic 管理 schema 变更。

面试口径：

> Demo 和本地迭代可以用 `create_all` 降低成本，但生产环境需要 migration 文件、审核和回滚策略。

### 14.2 首个用户自动成为管理员

适合本地演示，不适合公网环境。

生产替代：

- 初始化管理员命令。
- 环境变量指定管理员。
- 管理员邀请码。
- 关闭公开注册管理员能力。

### 14.3 BM25 统计持久化

当前 BM25 词表、文档频率和平均文档长度主要在进程内存中。多实例或重启后需要持久化，或者接入 Elasticsearch / OpenSearch / Milvus 内置稀疏能力的成熟方案。

### 14.4 请求级上下文

为兼容旧链路，部分 RAG 上下文仍通过模块级变量传递。并发量上来后，应使用 `contextvars.ContextVar` 或请求级依赖注入。

### 14.5 多模态边界

当前只记录图片 metadata，没有做多模态 embedding。面试时必须说清楚，避免过度包装。

## 十五、Demo 顺序

1. 注册第一个用户，展示自动成为管理员。
2. 登录，展示请求携带 JWT。
3. 上传 PDF/Word，说明管理员权限和文档入库链路。
4. 提问知识库问题，展示 RAG 实时步骤。
5. 展开 Trace，说明初次召回、Rerank、Auto-merging 和来源。
6. 提问需要最新信息或日期的问题，展示 Router 分流。
7. 提问“结合知识库和最新资料制定计划”，展示 Planner 任务创建、并行执行和 synthesis。
8. 点击停止回答，说明 AbortController 到 `task.cancel()` 的取消链路。
9. 展示 `/memories`，说明长期记忆和偏好记忆隔离。
10. 注册普通用户，验证看不到管理员文档管理和其他用户会话。

## 十六、准备清单

必须熟练：

- 画出 Router、Planner、ToolRegistry、RAG、Memory、Trace 的完整链路。
- 解释为什么 Router、Planner、Agent、LangGraph 不是重复设计。
- 解释 Hybrid Search、RRF、WeightedRanker、Rerank 的边界。
- 解释三级分块、Leaf-only 存储和 Auto-merging。
- 解释 `asyncio.Queue`、`call_soon_threadsafe` 和 SSE。
- 解释 Agent Reliability 的 retry、circuit breaker、stuck detection 和 recovery。
- 解释 JWT 用户隔离和 RBAC。
- 解释 PostgreSQL 与 Redis 的职责边界。
- 解释 Memory 为什么拒绝临时任务和敏感信息。
- 承认 Alembic、BM25 持久化、请求级上下文等生产化缺口。

建议补充：

- 准备 10 条不同类型 Router 问题。
- 准备 3 个 Planner Demo：并行工具、依赖工具、工具失败恢复。
- 准备 20 条以上 RAG gold 评测问题。
- 记录 Dense、BM25、Hybrid、Hybrid + Rerank 的 Recall@K 和 MRR。
- 准备一份冲突文档 Demo，展示系统如何提示来源冲突。

## 十七、最终面试表达

你需要传达的核心不是“我会用 LangChain”，而是：

> 我能把 LLM 的不确定能力约束在 Router、Planner 和结构化输出里，把确定性的 RAG 流程交给 LangGraph，把工具接入统一到可观测、可重试、可熔断的基础设施里；同时能处理多用户认证、持久化、缓存、Memory、SSE 流式、任务取消、失败恢复和离线评估这些工程问题。
