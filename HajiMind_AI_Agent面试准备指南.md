# HajiMind AI Agent 工程师面试准备指南

## 一、项目定位

HajiMind 是一个面向知识库问答场景的多用户 RAG Agent 系统。

它不是简单的“聊天机器人”，而是把以下能力组合成了一个完整服务：

- JWT 登录认证和用户隔离。
- 基于 LangChain Agent 的工具路由。
- 基于 LangGraph 的 RAG 状态机。
- Milvus Dense + BM25 Sparse 混合检索。
- RRF 融合、Rerank 精排和三级分块 Auto-merging。
- FastAPI SSE 流式回答。
- 同步工具线程到异步事件循环的实时 RAG 进度推送。
- PostgreSQL 会话、消息和父级分块持久化。
- Redis 会话缓存。
- 管理员文档管理和普通用户权限隔离。

面试时建议将项目概括为：

> 我构建了一个具备认证、权限、持久化、缓存和可观测 RAG 能力的流式 AI Agent 服务，重点解决了检索质量、长上下文、异步流式和多用户数据隔离问题。

## 二、推荐的一分钟项目介绍

我开发了 HajiMind，一个面向企业知识库问答的多用户流式 RAG Agent。

后端使用 FastAPI，用户通过注册和登录获取 JWT，后续请求由服务端从 Token 中解析用户身份，不再信任前端传递的 `user_id`。普通用户可以进行聊天并管理自己的会话，管理员负责知识库文档的上传、查看和删除。

Agent 使用 LangChain，根据用户问题决定是否调用天气工具或知识库工具。知识库工具内部使用 LangGraph 编排 RAG 流程：先进行 Dense 和 BM25 Sparse 混合检索，再通过 RRF 融合和可选 Rerank 精排；如果相关性评分不足，则使用 Step-back、HyDE 或 complex 策略重写查询并进行二次检索。

文档入库时采用 L1/L2/L3 三级分块，只将 L3 叶子块写入 Milvus，L1/L2 父块存入 PostgreSQL。检索命中多个子块时，从数据库取回父块进行 Auto-merging，兼顾召回精度和上下文完整性。

项目中我还解决了一个异步工程问题：同步 RAG 工具运行在线程池中，无法直接向 FastAPI 的 asyncio 事件循环推送进度。我通过 `asyncio.Queue`、后台 Agent 任务和 `call_soon_threadsafe` 实现工具执行期间的实时 SSE 事件推送。同时使用 AbortController、GeneratorExit 和 `task.cancel()` 实现客户端断开后的资源回收。

## 三、整体架构

```text
                         ┌─────────────────────┐
                         │ Vue 3 前端           │
                         │ 登录 / 聊天 / 历史     │
                         │ 管理员文档页面        │
                         └──────────┬──────────┘
                                    │ JWT + SSE
                                    v
                         ┌─────────────────────┐
                         │ FastAPI API          │
                         │ Auth / RBAC / Chat   │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  v                                   v
       ┌────────────────────┐              ┌────────────────────┐
       │ LangChain Agent     │              │ PostgreSQL          │
       │ 工具选择与回答生成   │              │ 用户/会话/消息/父块   │
       └─────────┬──────────┘              └────────────────────┘
                 │
                 v
       ┌────────────────────┐              ┌────────────────────┐
       │ LangGraph RAG       │              │ Redis               │
       │ 召回/评分/重写/二召回 │              │ 会话缓存             │
       └─────────┬──────────┘              └────────────────────┘
                 │
                 v
       ┌────────────────────────────────────────────────────────┐
       │ Dense Embedding + BM25 Sparse -> Milvus -> RRF -> Rerank │
       └────────────────────────────────────────────────────────┘
```

## 四、请求链路

### 4.1 登录和认证链路

```text
注册/登录
  -> 服务端校验密码
  -> 生成 JWT
  -> 前端保存 access_token
  -> 后续请求添加 Authorization: Bearer <token>
  -> FastAPI 解析 Token
  -> 查询当前用户
  -> 进入业务接口
```

当前实现：

- 第一个注册用户自动成为管理员。
- 后续注册用户默认为普通用户。
- JWT 中包含用户 ID、用户名、角色和过期时间。
- 用户停用后，即使 Token 未过期也无法继续访问。

这里的“首个用户自动成为管理员”适合本地演示，生产环境应改为初始化管理员命令或环境变量。

### 4.2 聊天链路

```text
POST /chat/stream
  -> Bearer Token 认证
  -> 获取当前 user.id
  -> 读取 PostgreSQL / Redis 会话
  -> LangChain Agent
  -> 工具调用或直接回答
  -> 流式返回 content / rag_step / trace
  -> PostgreSQL 保存消息
  -> 删除相关 Redis 缓存
```

聊天请求只包含：

```json
{
  "message": "用户问题",
  "session_id": "session_xxx"
}
```

不再包含 `user_id`。用户身份由服务器从 JWT 获取。

### 4.3 文档入库链路

```text
管理员上传 PDF/Word
  -> JWT + admin 权限检查
  -> 保存原文件
  -> 文档解析
  -> L1/L2/L3 三级分块
  -> L1/L2 写入 PostgreSQL parent_chunks
  -> L3 生成 Dense + Sparse 向量
  -> L3 写入 Milvus
```

重复上传同名文件时，会先删除 Milvus 中旧叶子块和 PostgreSQL 中旧父级块，再重新入库。

## 五、认证和权限设计

### 5.1 为什么必须移除前端 user_id

旧方案中，前端可以请求：

```json
{
  "user_id": "other_user",
  "session_id": "some_session"
}
```

如果后端直接信任这个字段，就存在越权读取和删除其他用户会话的风险。

新方案的核心原则是：

> 用户身份必须来自服务端验证过的凭证，而不是来自请求体。

接口依赖关系：

```text
get_current_user
        |
        +-- 聊天接口
        +-- 会话列表
        +-- 会话详情
        +-- 删除会话

require_admin
        |
        +-- 文档列表
        +-- 文档上传
        +-- 文档删除
```

### 5.2 JWT 适合当前项目的原因

JWT 适合当前前后端分离的单体服务：

- 服务端不需要保存每个 Token 的 Session 状态。
- 前端请求携带方式简单。
- FastAPI 可以通过依赖统一实现认证。
- Token 中可以携带角色信息。

但 JWT 也有明显限制：

- Token 签发后，单独撤销比较麻烦。
- 密钥泄露会影响全部 Token。
- 生产环境需要短过期时间和 Refresh Token。
- 角色变化后应考虑 Token 失效策略。

## 六、数据库设计

### 6.1 `users`

保存：

- 用户名。
- PBKDF2-SHA256 密码哈希。
- 角色：`admin` 或 `user`。
- 是否启用。
- 创建时间。

用户名建立唯一索引。

### 6.2 `chat_sessions`

保存：

- 数据库主键。
- 用户 ID。
- 业务会话 ID。
- 会话元数据。
- 创建时间和更新时间。

`user_id + session_id` 建立联合唯一约束，确保同一用户不会出现重复会话。

### 6.3 `chat_messages`

保存：

- 所属会话。
- 所属用户。
- 消息类型。
- 消息内容。
- 时间戳。
- `rag_trace`。

`rag_trace` 使用 PostgreSQL JSONB 保存，适合检索字段经常变化的 Agent 调试数据。

### 6.4 `parent_chunks`

保存 L1/L2 父级块：

- `chunk_id`
- `parent_chunk_id`
- `root_chunk_id`
- `chunk_level`
- 文件名
- 页码
- 文本内容

`chunk_id` 唯一，方便 Auto-merging 按 ID 批量查询。

## 七、Redis 缓存设计

当前采用 Cache-Aside 模式。

### 7.1 读取流程

```text
读取会话
  -> Redis 命中：直接返回
  -> Redis 未命中
  -> 查询 PostgreSQL
  -> 写入 Redis
  -> 返回结果
```

缓存 Key：

```text
hajimind:sessions:list:{user_id}
hajimind:sessions:{user_id}:{session_id}
```

### 7.2 写入流程

```text
写入 PostgreSQL
  -> 删除会话详情缓存
  -> 删除用户会话列表缓存
```

### 7.3 为什么 Redis 只作为缓存

聊天历史是核心数据，Redis 不应成为唯一存储。当前设计让 PostgreSQL 负责可靠持久化，Redis 只负责提高读取性能。

Redis 不可用时，系统会自动回源 PostgreSQL，保证缓存故障不会直接导致聊天功能不可用。

## 八、Agent 设计

当前 Agent 有两个工具：

- `get_current_weather`
- `search_knowledge_base`

Agent 的职责是：

1. 理解用户问题。
2. 判断是否需要调用工具。
3. 选择合适工具。
4. 结合工具结果生成最终回答。

RAG 内部流程不完全交给 Agent 自由决策，而是使用 LangGraph 显式编排。这样可以限制工具调用次数，减少重复检索和死循环。

当前提示词还限制：

- 每轮最多调用一次知识库工具。
- 检索工具返回结果后直接回答。
- 上下文不足时诚实回答，不编造事实。
- 不向用户暴露 Chain of Thought。

## 九、LangGraph RAG 状态机

```text
retrieve_initial
        |
        v
grade_documents
    |          |
  yes          no
    |          |
  END    rewrite_question
                   |
                   v
          retrieve_expanded
                   |
                   v
                  END
```

### 9.1 初次召回

1. 接收用户原始问题。
2. 只过滤召回 `chunk_level == 3` 的叶子块。
3. 生成 Dense 和 Sparse 查询向量。
4. Milvus 混合检索。
5. 对候选集进行 Rerank。
6. Auto-merging。
7. 组织上下文和 Trace。

### 9.2 相关性评分

Grader 使用结构化输出：

```python
class GradeDocuments(BaseModel):
    binary_score: str
```

输出 `yes/no`，控制是否进入二次检索。

这是一个成本控制点：只有初次召回质量不足时，才触发额外 LLM 调用。

### 9.3 查询重写

支持三种策略：

- `step_back`：把具体问题抽象为通用原理。
- `hyde`：生成一段假设性文档，用于改善检索语义。
- `complex`：同时使用 Step-back 和 HyDE，多路召回后去重。

### 9.4 RAG Trace

每次知识库检索记录：

- 初始问题。
- 初次召回结果。
- 相关性评分。
- 是否重写。
- 重写策略和内容。
- 二次召回结果。
- 检索模式。
- RRF 排名。
- Rerank 分数。
- Auto-merging 信息。

这些信息会随 AI 消息保存在 PostgreSQL，前端可以展开查看。

## 十、检索系统设计

### 10.1 Dense 检索

Dense Embedding 通过 OpenAI 兼容 API 生成，Milvus 使用 HNSW 索引和内积距离。

优点：

- 能捕捉语义相似。
- 对同义表达更友好。

缺点：

- 对编号、专有名词和精确关键词不一定稳定。

### 10.2 BM25 Sparse 检索

当前使用自定义中英文分词：

- 中文按单字切分。
- 英文按正则单词切分。
- 自己维护词表、文档频率、IDF。
- 使用 `k1=1.5`、`b=0.75`。

不要说项目使用 jieba，实际代码并没有使用 jieba。

### 10.3 RRF 融合

Dense 和 Sparse 分数分布不一致，直接加权求和需要调节权重。当前使用：

```text
Dense Top-K
Sparse Top-K
       |
       v
RRFRanker(k=60)
       |
       v
融合排名
```

RRF 的优势是参数少、实现稳定。后续可以通过标注集比较 RRF 和加权融合的效果。

### 10.4 Rerank

先取 `top_k * 3` 候选，再调用 `/v1/rerank` 兼容接口进行精排。

Rerank 未配置或调用失败时，系统保留原有检索结果，不让精排服务故障阻断问答。

## 十一、三级分块和 Auto-merging

### 11.1 为什么使用三级分块

单一 chunk size 很难同时满足：

- 精准召回。
- 完整上下文。
- 控制上下文长度。

因此项目采用：

```text
L1：大主题块
  -> L2：段落块
      -> L3：叶子块
```

### 11.2 Leaf-only 存储

- 只有 L3 进入 Milvus。
- L1/L2 写入 PostgreSQL。
- Milvus 只承担精准召回职责。
- PostgreSQL 负责父块恢复。

### 11.3 Auto-merging

```text
命中多个 L3
  -> 按 parent_chunk_id 分组
  -> 命中数量达到阈值
  -> 查询 PostgreSQL 父块
  -> L3 合并为 L2
  -> 再尝试合并到 L1
```

这样可以减少重复上下文，同时避免只返回很短的碎片。

## 十二、流式输出和异步工程

### 12.1 为什么需要统一 Queue

Agent 生成文本和 RAG 工具进度来自不同执行路径：

```text
Agent 后台任务 -> content
同步工具线程 -> rag_step
主生成器       -> Queue -> SSE
```

如果主生成器直接等待 Agent 工具返回，工具执行期间前端无法看到检索步骤。

### 12.2 跨线程事件调度

在主线程保存当前 Event Loop：

```python
loop = asyncio.get_running_loop()
```

工具线程中不能直接操作主线程 Queue，因此使用：

```python
loop.call_soon_threadsafe(
    queue.put_nowait,
    step
)
```

这保证了同步线程向异步事件循环投递事件时的线程安全。

### 12.3 SSE 事件类型

- `content`：答案 Token。
- `rag_step`：检索过程。
- `trace`：完整 RAG Trace。
- `error`：错误。
- `[DONE]`：完成标记。

### 12.4 中断和资源回收

```text
AbortController.abort()
  -> fetch 断开
  -> 服务端生成器关闭
  -> GeneratorExit
  -> agent_task.cancel()
  -> 上游 HTTP 请求释放
```

面试时要强调：

> 这个设计的目标不是简单改变前端按钮状态，而是确保客户端断开后，上游模型任务也被取消，减少无效推理和 Token 消耗。

## 十三、会话记忆

会话消息存储在 PostgreSQL。

当消息数量超过 50 条时：

1. 取前 40 条旧消息。
2. 调用模型生成摘要。
3. 用 `SystemMessage` 注入摘要。
4. 保留较新的消息继续对话。

这是“短期原始消息 + 长期摘要”的记忆策略。

需要承认的限制：

- 摘要本身可能丢失细节。
- 当前没有独立的用户画像存储。
- 还没有接入 Mem0、LangMem 等长期记忆方案。

## 十四、接口和权限速览

### 认证接口

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`

### 聊天接口

- `POST /chat`
- `POST /chat/stream`

### 会话接口

- `GET /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`

### 管理员文档接口

- `GET /documents`
- `POST /documents/upload`
- `DELETE /documents/{filename}`

除注册和登录外，接口统一要求：

```http
Authorization: Bearer <access_token>
```

## 十五、面试高频问题

### Q1：为什么 Agent 和 LangGraph 要同时使用？

回答：

> Agent 负责开放式的任务判断和工具选择，LangGraph 负责 RAG 内部确定性的状态流转。两者职责不同，结合起来既保留 Agent 的灵活性，又保证检索流程可控、可观测和容易测试。

### Q2：为什么不把所有流程交给 Agent？

回答：

> 如果把召回、评分、重写全部交给 Agent，工具调用次数、成本和结束条件更难控制。RAG 是有明确业务流程的部分，更适合使用图状态机；Agent 只负责决定是否进入这个工具。

### Q3：为什么使用 PostgreSQL 而不是继续使用 JSON？

回答：

> JSON 整体读写不适合多用户并发和复杂查询，也无法很好支持索引、事务和级联删除。PostgreSQL 可以按用户和会话查询，并通过外键保证数据关系。

### Q4：为什么 Redis 不能替代 PostgreSQL？

回答：

> Redis 在这里是缓存，不是核心持久化。聊天记录和 RAG Trace 必须可靠保存，Redis 数据可能过期或丢失，因此 PostgreSQL 是事实来源，Redis 只用于加速读取。

### Q5：如何保证用户不能查看别人的会话？

回答：

> 用户 ID 从 JWT 中解析，所有会话查询都同时带上 `ChatSession.user_id == current_user.id` 条件，URL 中不再接收 user_id。文档接口还增加角色依赖，只有 admin 才能访问。

### Q6：PBKDF2 和 bcrypt 如何兼容？

回答：

> 新用户使用 PBKDF2-SHA256 存储，密码字符串包含算法标记、迭代次数、盐和摘要。校验时先判断前缀；如果是 PBKDF2 走 hashlib，如果是 `$2a$`、`$2b$` 或 `$2y$` 则走 bcrypt。

### Q7：RRF 为什么比加权求和简单？

回答：

> Dense 和 Sparse 的分数尺度不一致，加权求和需要调 alpha。RRF 只看排序名次，对分数尺度不敏感，初期更容易稳定落地。后续仍然可以用评测集验证是否需要加权融合。

### Q8：如果 Redis 挂了怎么办？

回答：

> Redis 操作全部是可选的。缓存读取失败时回源 PostgreSQL，缓存写入失败不影响主流程。这样缓存故障只影响性能，不影响数据正确性。

### Q9：Rerank 服务挂了怎么办？

回答：

> Rerank 失败时保留混合召回结果，并在 Trace 中记录错误。系统具备 hybrid 到 dense 的降级路径，避免单个外部服务故障导致整个问答失败。

### Q10：如何评估 RAG 效果？

回答：

> 目前需要继续建设标注集，至少对比 Dense only、Sparse only、Hybrid、Hybrid + Rerank 和 Query Rewrite。检索指标使用 Recall@K、MRR、nDCG，生成指标关注答案正确性、Faithfulness、响应延迟和 Token 消耗。

## 十六、必须主动说明的工程风险

### 16.1 全局 RAG 上下文存在并发风险

当前 RAG 事件队列和最近一次 Trace 仍通过模块级变量传递。多个请求并发时，可能发生事件或 Trace 串扰。

改进方向：

- 使用 `contextvars.ContextVar`。
- 将请求上下文封装成对象。
- 使用请求级依赖注入工具。
- 不再使用“最近一次全局结果”。

### 16.2 当前数据库初始化还不是正式迁移系统

现在服务启动时使用 `Base.metadata.create_all()`。生产环境应使用 Alembic：

```text
模型变更
  -> 生成 migration
  -> 审核 migration
  -> 部署执行
```

### 16.3 首个用户自动成为管理员只适合 Demo

生产环境应使用：

- 初始化管理员脚本。
- 管理员邀请码。
- 环境变量指定管理员。
- 禁止公开注册管理员。

### 16.4 BM25 词表需要持久化

当前 BM25 词表和统计信息主要在进程内存中。服务重启或多实例部署时，需要持久化词表和文档统计，或者使用成熟搜索引擎的稀疏检索能力。

### 16.5 旧 JSON 迁移需要账号映射

旧会话只有历史 user_id，没有完整账号和密码体系。仓库提供：

```bash
python backend/migrate_legacy.py --default-password "<temporary-password>"
```

迁移后应要求用户立即修改临时密码。

## 十七、面试 Demo 顺序

1. 注册第一个用户，展示其自动成为管理员。
2. 登录并展示 Token 请求。
3. 上传 PDF，说明只有管理员可以操作。
4. 提问知识库问题。
5. 展示实时 RAG 步骤。
6. 展开 RAG Trace，展示来源、RRF 名次和 Rerank 分数。
7. 注册第二个普通用户，验证不能访问文档管理。
8. 展示普通用户只能看到自己的会话。
9. 点击停止回答，展示中断链路。
10. 查看 PostgreSQL 数据和 Redis 缓存。

## 十八、准备清单

### 必须熟练

- 能画出认证、Agent、RAG、数据库、Redis 的完整架构。
- 能解释 JWT 为什么比前端 user_id 更安全。
- 能解释数据库四张核心表的关系。
- 能解释 Cache-Aside 和缓存失效。
- 能解释 Agent 与 LangGraph 的职责边界。
- 能解释 Dense、BM25、RRF、Rerank。
- 能解释三级分块和 Auto-merging。
- 能解释 `asyncio.Queue` 和 `call_soon_threadsafe`。
- 能解释 AbortController 到 `task.cancel()`。

### 建议补充

- 建立 20 条以上 RAG 评测问题。
- 记录 Recall@K、MRR、答案正确性和延迟。
- 使用 Alembic 替代 `create_all`。
- 使用请求级上下文解决并发串扰。
- 增加 Refresh Token、退出登录和密码修改接口。
- 增加 API 限流、审计日志和管理员操作日志。

## 十九、最终面试表达

你需要让面试官感受到的不是“会使用 LangChain”，而是：

> 我能把 LLM 能力封装成 Agent，把确定性 RAG 流程交给状态机；能从召回、融合、精排和上下文合并优化检索质量；能用 PostgreSQL、Redis、JWT 和 RBAC 把 Demo 改造成多用户服务；同时能处理 SSE 流式、跨线程事件调度、任务取消、缓存降级和数据迁移等工程问题。
