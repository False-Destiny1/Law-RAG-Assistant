# Agent 开发面试高频问题 — 基于项目代码的完整QA

> 本文档基于「智能法律助手」项目代码，模拟面试场景，覆盖 RAG 系统设计、Agent 架构、向量检索、LLM 应用、安全、性能优化等高频考点。每个问题包含：完整回答、代码引用、改进建议。

---

## 一、RAG 系统设计

### Q1: 请介绍你的 RAG 系统的整体架构

**回答：**

我的项目是一个基于 RAG 的中文法律智能问答系统。整体架构分为五层：

1. **安全层**（`app.py`）：CSRF 防护、基于 Redis 的固定窗口限流、HMAC 签名的 Cookie Session 认证、正则表达式提示注入检测
2. **查询分析层**（`rag.py:analyze_query()`）：单次 LLM 调用同时完成多轮对话融合、口语→法律术语改写、查询分解（最多 3 个子查询）、HyDE 假设文档生成
3. **三路融合检索层**（`rag.py:hybrid_retrieve_documents()`）：FAISS 向量检索（权重 0.4）+ BM25 关键词检索（权重 0.3）+ Neo4j 知识图谱检索（权重 0.3），通过 ThreadPoolExecutor 并行执行子查询
4. **重排与过滤层**：本地 CrossEncoder 重排器（`BAAI/bge-reranker-v2-m3`）+ 相关性阈值过滤（默认 0.15）
5. **生成层**（`rag.py:generate_response_stream()`）：构建带引用编号 `[来源N]` 的上下文，SSE 流式输出

整个流程每轮对话只需 2 次 LLM 调用（1 次查询分析 + 1 次回答生成），相比传统 RAG 的 3-4 次调用节省了约 60% 的查询准备阶段延迟。

**改进建议：**
- `app.py` 是一个 1200 行的单体文件，包含了 ORM 模型、认证逻辑、路由处理器、后台任务。应拆分为 `models.py`、`auth.py`、`routes/`、`services/` 等模块
- `Base.metadata.create_all(engine)` 在模块导入时执行，生产环境应使用 Alembic 做数据库迁移

---

### Q2: 你的查询分析为什么要用单次 LLM 调用完成多任务？相比多次调用有什么优势？

**回答：**

传统做法是分三次 LLM 调用：第一次做多轮对话融合，第二次做查询改写，第三次做查询分解。我选择用一个精心设计的 prompt（`prompts.yaml` 中的 `query_analysis_prompt`），让 LLM 一次性输出结构化 JSON，包含 `rewritten_query`、`sub_queries`、`hypothetical_doc` 三个字段。

优势：
- **延迟降低**：省去两次 LLM 调用的网络往返，按每次调用 500ms 算，节省约 1 秒
- **成本降低**：LLM API 调用次数减半
- **上下文一致性**：改写和分解共享同一个上下文窗口，结果更连贯

**改进建议：**
- 如果未来模型输出不稳定，可以考虑用 JSON Schema 约束输出格式，或者加一层输出校验/重试机制
- 当前 prompt 里包含了口语→法律术语的映射规则（如「老板→用人单位」），这个映射表是硬编码的，应该外置到配置文件中方便维护

---

### Q3: HyDE（假设文档嵌入）是什么？在你的系统中怎么用的？

**回答：**

HyDE 的核心思想是：先让 LLM 根据用户问题生成一篇「假设的答案文档」，然后用这篇假设文档去检索，而不是用原始问题检索。因为假设文档和目标文档在语义空间中更接近。

在我的系统中，`analyze_query()` 会让 LLM 生成一段 400-600 字的中文法律条文片段作为假设文档。然后这个假设文档和原始查询、子查询一起并行送入三路检索。

实现细节（`rag.py:retrieve_documents()`）：
```python
# 子查询 + HyDE 文档并行检索
futures = []
for sq in sub_queries:
    futures.append(executor.submit(self.hybrid_retrieve_documents, sq, top_k))
if hypothetical_doc:
    futures.append(executor.submit(self.hybrid_retrieve_documents, hypothetical_doc, top_k))
```

假设文档的检索结果和主查询结果通过加权融合，命中多路的文档会获得额外加分。

**改进建议：**
- HyDE 生成质量完全依赖 LLM，如果 LLM 生成的假设文档偏题，会引入噪声。可以加一个假设文档质量评估环节，或者用多个假设文档取平均
- 假设文档检索可以单独用更宽松的阈值，因为它是辅助召回

---

### Q4: 为什么选择三路融合检索而不是单一检索方式？权重怎么确定的？

**回答：**

三种检索方式各有优劣：
- **FAISS 向量检索**：语义理解强，能找到意思相近但用词不同的内容；但对精确关键词匹配弱
- **BM25 关键词检索**：精确匹配能力强，适合法律条文中特定术语（如「善意取得」「违约责任」）；但无法理解同义词
- **Neo4j 知识图谱检索**：能发现法律条文之间的引用关系和概念关联，提供结构化的关联信息；但覆盖率依赖图谱构建质量

三路融合通过加权求和，实现了互补：
```
final_score = 0.4 * vector_score + 0.3 * bm25_score + 0.3 * graph_score
```

权重初始值是经验值，系统提供了管理员 API（`/api/retrieval-weights`）可以动态调整。当 Neo4j 不可用时，权重自动归一化为 `vector:0.57, bm25:0.43`。

**改进建议：**
- 权重可以改为自适应学习：基于用户反馈（thumbs up/down）自动优化各路权重
- 融合策略可以尝试 RRF（Reciprocal Rank Fusion），它不依赖分数归一化，对不同检索器更鲁棒
- 相关性阈值 0.15 是硬编码的，应该也是可配置的

---

## 二、向量数据库与检索

### Q5: FAISS 在你的系统中是怎么用的？有什么优缺点？

**回答：**

FAISS 用于存储法律文档的向量索引。向量模型使用 `BAAI/bge-small-zh-v1.5`（本地 CUDA），也支持切换到 DashScope 云端 API。

初始化时从磁盘加载已有的 FAISS 索引（`rag.py:load_vector_db()`），并用 SHA-256 哈希校验完整性防篡改。检索时用 `similarity_search_with_score` 获取文档和相似度分数。

关键设计：
1. **脏标记延迟重建**：文档删除时不立即重建 FAISS，而是标记 `_faiss_dirty = True`，下次查询时才重建。避免了频繁删除时的性能抖动
2. **写锁保护**：`_faiss_write_lock` 确保并发写入安全
3. **完整性校验**：通过计算索引文件的 SHA-256 哈希检测损坏

```python
# 脏标记机制
self._faiss_dirty = False  # 文档删除时设为 True

def rebuild_faiss_if_dirty(self):
    if self._faiss_dirty:
        # 从 BM25 文档列表重建完整 FAISS 索引
        ...
        self._faiss_dirty = False
```

**FAISS 的优点**：内存占用低、查询速度快（百万级毫秒响应）、支持 GPU 加速
**FAISS 的缺点**：不支持增量更新（需要整体重建）、没有元数据过滤能力（需要后处理）、不支持分布式

**改进建议：**
- 当前 `rebuild_faiss_if_dirty` 从 BM25 文档列表重建，如果 BM25 和 FAISS 的文档不一致会导致数据丢失。应该从统一的文档源重建
- `allow_dangerous_deserialization=True` 是安全隐患，应确保索引文件来源可信
- Token 估算用 `len(text) * 2 // 3`，中英混合文本误差可达 30-50%，应引入 tiktoken 或字符级别的更精确估算

---

### Q6: BM25 检索的实现细节是什么？为什么需要自定义分词？

**回答：**

BM25 检索基于 `rank_bm25.BM25Okapi`，分词使用 jieba。关键点在于 jieba 默认的分词会把法律术语切成更小的词，导致检索效果下降。

所以我手动往 jieba 词典中添加了 129 个法律领域高频术语，设置高频词权重为 9999，确保它们不会被错误切分：

```python
LEGAL_TERMS = [
    "善意取得", "不当得利", "无因管理", "违约责任",
    "缔约过失", "格式条款", "物权变动", ...
]
for term in LEGAL_TERMS:
    jieba.add_word(term, freq=9999)
```

BM25 索引采用批量重建策略：文档先缓冲到 `pending_documents` 列表中，达到阈值（50 篇）或单文件上传时才触发全量重建。索引使用 JSON 持久化（不用 pickle），消除反序列化攻击面。

**改进建议：**
- `remove_documents` 方法在锁内做分词，CPU 密集操作阻塞其他线程。应该像 `build_index` 一样在锁外预分词，锁内只做原子替换
- BM25 不支持增量更新，大规模文档集每次重建很慢。可以考虑用 Elasticsearch 替代，或者引入 Anserini 等支持增量的工具
- JSON 序列化分词结果比较浪费空间，可以考虑 msgpack 或二进制格式

---

### Q7: 知识图谱在你的系统中扮演什么角色？检索策略是怎样的？

**回答：**

知识图谱基于 Neo4j，存储四种实体：法律（Law）、章节（Chapter）、条款（Article）、法律概念（Concept），以及它们之间的关系（包含、引用、定义）。

图谱的价值在于**发现关联信息**：当用户问「善意取得的构成要件」时，图谱不仅能找到定义「善意取得」的条文，还能通过 2-hop 遍历找到引用该概念的其他条文，提供更全面的上下文。

检索策略是四层优先级：
1. **精确条款匹配**（权重 1.0）：查询指定了法律名称+条款号
2. **法律范围内检索**（权重 0.8）：只指定了法律名称
3. **概念 2-hop 遍历**（权重 0.9/0.7）：找到定义概念的条款，再沿引用边扩展
4. **全文模糊匹配**（权重 0.5/0.3）：fallback 策略

**改进建议：**
- 图谱构建有严重的 N+1 问题：`build_from_text` 对每个条款执行单独的 `session.run()`，一部法律可能有上百条，会执行上百次事务。应该用 UNWIND 批量操作
- 引用提取的 `clause` 类型定义了模式但从未处理，存在死代码
- `get_stats` 方法用 f-string 拼接 Cypher，虽然有白名单校验但模式本身有风险
- Neo4j 连接没有连接池，图谱检索从线程池调用，高并发下可能成为瓶颈

---

## 三、对话记忆管理

### Q8: 你的对话记忆系统是怎么设计的？为什么用三级缓存？

**回答：**

对话记忆采用 L1→L2→L3 三级缓存架构：

| 层级 | 存储 | 容量 | TTL | 用途 |
|------|------|------|-----|------|
| L1 | 内存 OrderedDict | 500 个对话 | 无 | 热数据快速访问 |
| L2 | Redis | 不限 | 24 小时 | 跨进程共享 |
| L3 | PostgreSQL | 不限 | 永久 | 持久化兜底 |

查询时从 L1 开始查找，未命中则依次查 L2、L3，命中后回填到 L1（cache-aside 模式）。

最关键的设计是**两阶段锁**（`memory.py:add_message()`）：

```python
# 阶段1：在锁内做内存操作（快速）
with self._lock:
    history.append({"role": role, "content": content, "timestamp": ...})
    if token_budget_exceeded:
        split_point = len(history) // 2
        old_messages = history[:split_point]

# 阶段2：在锁外调用 LLM 做摘要（慢操作，不持锁）
summary = self.summarizer(old_messages)

# 阶段3：重新获取锁写回摘要
with self._lock:
    history = history[split_point:]
    memory["summary"] = summary
```

这避免了在 LLM 调用期间持锁，防止了线程死锁。

Token 预算管理：总预算 2500 tokens，摘要部分 500 tokens，最近消息部分 2000 tokens。摘要超过 600 字符时会被截断。

**改进建议：**
- 摘要截断用 `combined[-600:]` 取后 600 字符，会丢失摘要开头内容。应该改为保留前 600 字符，或者在截断前让 LLM 重新总结
- L1 缓存没有 TTL，长时间不活跃的对话会一直占用内存直到 LRU 淘汰。可以加一个 idle timeout
- `_write_to_redis` 在锁外调用，依赖 CPython GIL 保证字典读一致性，这不符合 Python 语言规范
- Token 估算用字符数/2，不够精确。可以引入 tiktoken 计算

---

### Q9: 为什么要在对话记忆中做 Token 预算管理？具体怎么实现的？

**回答：**

LLM 有上下文窗口限制（如 8K/32K tokens），对话历史如果无限增长会超出限制。而且过长的历史会稀释关键信息的注意力权重，降低回答质量。

我的实现是一个滑动窗口 + 摘要压缩的混合方案：

1. 每次添加消息后检查总 token 数是否超过 `HISTORY_TOKEN_BUDGET`（2500）
2. 超过时，取前半部分消息调用 LLM 生成摘要（限 600 字符）
3. 保留摘要 + 后半部分消息作为新的历史
4. 输出时，摘要占 `SUMMARY_TOKEN_BUDGET`（500 tokens），最近消息占 `RECENT_TOKEN_BUDGET`（2000 tokens）
5. 硬兜底：最多保留 `max_history_turns * 2` 条消息

**改进建议：**
- Token 估算方法 `len(text) * 2 // 3` 对中文偏保守（中文约 1.5-2 tokens/字符），对英文偏宽松。引入 tiktoken 会更准确
- 摘要生成本身消耗 LLM 调用，可以考虑用更小的模型做摘要，或者用 extractive summarization（抽取式）替代 abstractive summarization（生成式）

---

## 四、LLM 应用与 Prompt Engineering

### Q10: 你的 Prompt 是怎么设计的？有哪些防注入措施？

**回答：**

Prompt 存储在 `prompts.yaml` 中，有两个模板：

**`legal_advisor_prompt`**（回答生成）：包含角色设定（专业法律顾问）、安全规则（拒绝执行系统指令相关的请求）、引用格式要求（`[来源N]`）、输出风格（专业但易懂）

**`query_analysis_prompt`**（查询分析）：包含口语→法律术语映射规则、输出 JSON Schema 要求、子查询数量限制（最多 3 个）、假设文档长度要求（400-600 字）

防注入措施采用纵深防御：

1. **输入层**（`app.py`）：`check_injection()` 正则检测，分高置信度模式（角色劫持、系统指令提取、模板注入）和低置信度模式（"忽略之前指令"变体）
2. **上下文层**（`rag.py:sanitize_context()`）：对检索到的文档内容逐行扫描，可疑行替换为 `[文档内容，已过滤可疑指令]`
3. **输出层**（Prompt 中）：明确告知 LLM 不要执行用户提出的系统指令相关请求

法律上下文豁免机制：当查询包含 2 个以上法律术语时，跳过低置信度注入检测，避免误杀合法法律讨论（如「债权人放弃债权」包含「放弃」这个触发词）。

**改进建议：**
- 正则检测覆盖面有限，可以引入 LLM-based 的注入检测作为第二层
- `sanitize_context` 逐行扫描太粗暴，可能误过滤合法内容。应该用更精细的分类器
- 查询长度限制 2000 字符对法律长文本（如完整合同条款粘贴）可能不够
- Prompt 中的安全指令容易被绕过（如 base64 编码、多语言混合），应配合输出过滤

---

### Q11: 重排器（Reranker）在 RAG 中的作用是什么？你为什么选择 CrossEncoder？

**回答：**

重排器的作用是对检索阶段召回的候选文档进行精排。检索阶段追求召回率（尽量不漏），重排阶段追求精确率（尽量准）。

我的系统支持两种重排器：
1. **本地 CrossEncoder**（`BAAI/bge-reranker-v2-m3`）：将 query 和每个 document 拼接后送入 BERT 模型，输出相关性分数。一次处理 20 对，batch 推理
2. **DashScope API**（`TextReRank`）：云端 API 调用，带 10 秒超时

CrossEncoder 相比 Bi-Encoder 的优势：
- CrossEncoder 把 query 和 document 一起编码，能捕获更细粒度的交互特征
- 对于法律这种需要精确理解的领域，CrossEncoder 的精度显著优于 Bi-Encoder
- 本地推理避免了网络延迟和 API 限制

重排后会过滤掉分数低于 `RELEVANCE_THRESHOLD`（0.15）的结果，防止低质量文档污染 LLM 上下文。

**改进建议：**
- CrossEncoder 处理上限 20 对，如果召回文档多需要分 batch。可以引入预过滤（如先用 Bi-Encoder 粗筛到 50 篇）减少 CrossEncoder 负载
- 重排超时 10 秒可能不够（GPU 冷启动时）。可以加预热机制
- 可以引入 ColBERT 等 late-interaction 模型作为折中方案，兼顾速度和精度

---

## 五、系统设计与架构

### Q12: 你的系统是怎么处理并发的？有哪些线程安全措施？

**回答：**

并发处理涉及三个层面：

**1. 检索层并行化**
使用 `ThreadPoolExecutor(max_workers=8)` 并行执行子查询检索。`hybrid_retrieve_documents` 内部的三路检索是串行执行的（避免嵌套线程池死锁），但外层的多个子查询是并行的。

**2. 线程安全保护**
- `_faiss_write_lock`：保护 FAISS 索引写操作
- `_cache_lock`：保护 KB 文本缓存的 L1 写入
- `BM25Retriever._lock`：保护 BM25 索引和 pending 文档列表
- `ConversationMemory._lock`：保护对话历史的内存读写

**3. 锁策略优化**
`ConversationMemory.add_message()` 采用两阶段锁：内存操作在锁内（快速），LLM 摘要调用在锁外（慢操作），写回在锁内。避免了持锁调用 LLM 导致的死锁。

```python
# 关键设计：锁外做 LLM 调用
with self._lock:
    # 快速内存操作
    ...
old_messages_for_summary = [...]  # 在锁外

summary = self.summarizer(old_messages_for_summary)  # 慢操作，不持锁

with self._lock:
    # 写回摘要
    memory["summary"] = summary
```

**改进建议：**
- FAISS 脏标记 `_faiss_dirty` 依赖 GIL 保证原子性，在非 CPython 实现中不安全。应该用 `threading.Event` 或 `threading.Condition`
- BM25 的 `remove_documents` 在锁内做分词，应该像 `build_index` 一样在锁外预处理
- `_SHARED_EXECUTOR` 是模块级共享的，但 `hybrid_retrieve_documents` 内部串行执行三路检索，没有充分利用线程池。可以考虑让三路检索也并行

---

### Q13: 你的系统是如何做服务降级的？如果某个组件挂了会怎样？

**回答：**

系统在每个外部依赖点都实现了优雅降级：

| 组件 | 故障时行为 | 降级方案 |
|------|-----------|---------|
| Redis | 连接失败 | 本地内存缓存兜底（`redis_fallback` 装饰器） |
| Neo4j | 连接失败 | 权重自动归一化，退化为向量+BM25 双路检索 |
| CrossEncoder | 推理超时 | 回退到原始融合分数排序 |
| DashScope API | 调用失败 | 回退到本地 CrossEncoder |
| Embedding 模型 | 推理失败 | 主模型失败后尝试备选模型，3 次重试+指数退避 |
| FAISS 索引 | 不存在/损坏 | 从 BM25 文档列表重建 |

Redis 降级的实现方式：
```python
def redis_fallback(default=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                return default
        return wrapper
    return decorator
```

**改进建议：**
- 降级都是静默的，没有监控告警。应该在降级时记录 metrics 和日志，便于运维发现
- Redis 故障后的本地内存限流是单进程的，多 worker 时形同虚设（每个 worker 独立计数）
- 重连冷却 30 秒太长，Redis 可能早就恢复了。应该用指数退避+最大冷却时间
- 没有 circuit breaker 模式，如果 Neo4j 反复连接失败会持续重试

---

### Q14: 你的 SSE 流式输出是怎么实现的？有什么设计考量？

**回答：**

SSE 流式输出是整个系统用户体验的核心。实现分为后端和前端两部分。

**后端**（`app.py` `/ask_stream` 端点）：
1. 用户消息先同步写入数据库（确保不丢）
2. 调用 `generate_response_stream()` 获取生成器
3. 返回 `StreamingResponse(media_type="text/event-stream")` 并设置 `X-Accel-Buffering: no`
4. 生成器逐 chunk 迭代 LLM 输出，yield `data: {"content": "..."}`
5. 流结束后保存完整回复到数据库和对话记忆
6. 最终发送 `data: {"done": true, "message_id": N}`

**前端**（`index.js:handleStream()`）：
1. 使用 `ReadableStream` + `TextDecoder` 读取 SSE 数据
2. 实现了缓冲区系统处理跨 chunk 的不完整行
3. 流式阶段用 `escapeHtml + <br>` 快速渲染纯文本
4. 流结束后用 `marked.parse()` + `DOMPurify.sanitize()` 渲染完整 Markdown
5. `AbortController` 支持用户中途停止生成

```javascript
// 前端流式渲染的性能优化
if (data.content) {
    accumulatedText += data.content;
    // 流式阶段：纯文本+换行（快）
    messageDiv.innerHTML = escapeHtml(accumulatedText).replace(/\n/g, '<br>') + streamingCursor;
}
// 完成时：完整 Markdown 渲染（慢但美观）
if (data.done) {
    messageDiv.innerHTML = DOMPurify.sanitize(marked.parse(accumulatedText));
}
```

**改进建议：**
- 流式阶段不做 Markdown 渲染是个好优化，但 `<br>` 换行和最终 Markdown 渲染会有视觉跳跃。可以考虑增量 Markdown 渲染（如 `marked` 的 streaming mode）
- 没有心跳机制，长时间无输出时连接可能被代理服务器断开。应该定期发送 `:keepalive\n\n`
- `generate_response_stream` 内部创建了新的 DB session，但外层请求的 session 可能已关闭，存在 stale read 风险

---

## 六、安全设计

### Q15: 你的系统有哪些安全措施？怎么防止提示注入？

**回答：**

安全措施覆盖多个层面：

**1. 认证与会话管理**
- 密码用 bcrypt 哈希加盐存储
- Session token 格式：`user_id:nonce:timestamp:hmac_signature`
- HMAC-SHA256 签名防伪造，服务端 24 小时过期检查
- 登出时 token 加入 Redis 黑名单

**2. CSRF 防护**
- 双提交 Cookie 模式：`csrf_token = SHA256(session_token + SECRET)[:32]`
- 通过 `X-CSRF-Token` 头或表单隐藏字段提交
- 常量时间比较（`hmac.compare_digest`）防时序攻击

**3. 限流**
- 基于 Redis 的固定窗口限流，不同端点不同限制
- `/ask_stream` 30次/分钟，`/login` 10次/分钟

**4. 安全头**
- `X-Frame-Options: DENY`（防点击劫持）
- `Content-Security-Policy`（限制脚本来源）
- `X-Content-Type-Options: nosniff`

**5. 提示注入防御（纵深防御）**
- 输入层：正则模式匹配（高置信度+低置信度两级）
- 上下文层：检索文档逐行扫描过滤
- 输出层：Prompt 中的安全指令
- 法律上下文豁免：包含 2+ 法律术语时跳过低置信度检测

**改进建议：**
- `/ask_stream` 端点跳过了 CSRF 校验，虽然有 SameSite=Lax 保护，但如果未来改为跨域调用会有风险
- CSP 中的 `unsafe-inline` 允许内联脚本，降低了 XSS 防护效果。应该改用 nonce 或 hash
- 正则注入检测无法防御高级攻击（如 Unicode 混淆、多语言注入），应引入 LLM-based 检测
- SESSION_SECRET 自动生成写入 `.env`，多 worker 并发时有竞态条件

---

### Q16: 你的系统怎么处理用户上传的文档？有什么安全考量？

**回答：**

文档上传流程：
1. 角色检查：只有 `expert` 和 `admin` 可以上传
2. 文件大小检查：分块读取（8KB）避免全量加载，上限 20MB
3. 文件保存到磁盘
4. 后台异步处理（`BackgroundTasks`）：自动检测法律/通用文档 → 分块 → 写入 FAISS + BM25 + Neo4j

安全考量：
- FAISS 索引加载使用 `allow_dangerous_deserialization=True`，这是 NumPy pickle 的必要设置。通过 SHA-256 哈希校验索引完整性
- BM25 索引用 JSON 持久化而非 pickle，消除了反序列化攻击面
- 上传的文件直接存储在本地磁盘，没有做文件类型二次校验（仅靠前端传的 file_type）
- OCR 处理时有内存管理：每 10 页做一次 `gc.collect()`，图片用后即删

**改进建议：**
- 文件上传应该在后端二次校验 MIME type（通过 `python-magic`），不信任前端传的类型
- 没有防病毒扫描，上传的文档可能包含恶意内容
- 文件名直接用用户上传的原始文件名，可能存在路径穿越风险（如 `../../etc/passwd`）
- 没有对上传文件内容做注入检测，恶意文档中的 prompt injection 可能被索引后污染后续检索

---

## 七、性能优化

### Q17: 你的系统有哪些性能优化措施？

**回答：**

**1. LLM 调用优化**
- 查询分析单次调用完成多任务（多轮融合+改写+分解+HyDE），省去 2 次 LLM 调用
- 总计每轮对话只需 2 次 LLM 调用

**2. 检索优化**
- 子查询并行执行（ThreadPoolExecutor）
- 三路检索结果缓存（Redis，TTL 10 分钟）
- 相关性阈值过滤（0.15）减少无效文档进入 LLM 上下文

**3. 缓存体系**
- 对话记忆：L1 内存 + L2 Redis + L3 DB
- KB 文本：L1 内存 LRU + L2 Redis + L3 重建
- FAISS 索引：磁盘持久化 + 内存加载
- BM25 索引：磁盘持久化 + 内存加载

**4. 文档处理优化**
- FAISS 脏标记延迟重建
- BM25 批量重建（阈值 50 篇）
- 文档上传异步处理（BackgroundTasks）
- PaddleOCR 懒加载（首次使用才初始化）

**5. 前端优化**
- 流式阶段纯文本渲染（不做 Markdown 解析）
- DOMPurify 防 XSS
- 自动调整 textarea 高度

**改进建议：**
- 没有 A/B 测试框架，无法量化各优化的效果
- Redis 缓存的 key 设计没有命名空间规范，可能和其他应用冲突
- 没有连接池复用（Neo4j、FAISS），高并发下可能成为瓶颈
- Token 估算精度不够，可能导致上下文截断不当

---

### Q18: 如果文档量增长到百万级，你的系统会遇到什么瓶颈？怎么解决？

**回答：**

当前系统在百万级文档下会遇到以下瓶颈：

**1. FAISS 索引重建耗时**
当前方案是全量重建。百万级文档的向量化可能需要数小时。
解决：引入 FAISS 增量索引（`IndexIDMap` + `add_with_ids`），只向量化新增文档。

**2. BM25 内存爆炸**
`BM25Okapi` 将所有分词结果存在内存中，百万级文档可能超出内存限制。
解决：分片 BM25（按法律分类），查询时并行搜索各分片再融合。

**3. Neo4j 图谱构建 N+1**
每条条款单独执行 Cypher，百万级条款构建时间不可接受。
解决：用 `UNWIND` 批量写入，或者分法律并行构建。

**4. 重排器瓶颈**
CrossEncoder 一次只能处理 20 对，百万召回文档需要 5 万次推理。
解决：先用 Bi-Encoder 粗筛到 100 篇，再用 CrossEncoder 精排。

**5. 启动时间**
`initialize_vector_database()` 在启动时索引所有文档。
解决：分离索引构建和应用启动，索引增量更新。

**改进建议：**
- 引入 Elasticsearch 替代 BM25，天然支持分布式和增量更新
- FAISS 可以用 `IndexIVFFlat` 或 `IndexHNSW` 提升大规模下的查询速度
- 知识图谱可以考虑用 NebulaGraph 等分布式图数据库
- 引入向量数据库如 Milvus/Qdrant，天然支持增量更新和过滤

---

## 八、Agent 相关

### Q19: 你的系统和传统的 Agent 有什么区别？为什么说它是 RAG 系统而不是 Agent？

**回答：**

严格来说，我的系统是 **RAG 系统**而非 **Agent 系统**。两者的核心区别：

| 特征 | RAG 系统 | Agent 系统 |
|------|---------|-----------|
| 执行模式 | 固定流水线 | 动态决策循环 |
| 工具使用 | 无（只有检索+生成） | 有（可调用外部工具/API） |
| 推理方式 | 单次推理 | 多步推理（ReAct/CoT） |
| 自主性 | 无（用户驱动） | 有（Agent 自主规划） |

我的系统是**检索增强生成**：用户提问 → 检索相关文档 → 基于文档生成回答。流程是固定的，没有自主决策能力。

如果要升级为 Agent 系统，可以：
1. 加入工具调用：允许 LLM 决定是否需要搜索法律数据库、调用计算器、查询案例库
2. 加入推理循环：实现 ReAct 模式，LLM 自主决定下一步行动
3. 加入规划能力：对复杂法律问题进行任务分解，分步执行

**改进建议：**
- 可以引入 LangGraph 或 CrewAI 构建多 Agent 协作系统
- 复杂法律问题可以拆分为：事实分析 Agent + 法律检索 Agent + 案例对比 Agent + 结论生成 Agent
- 加入自我反思机制：生成回答后自动评估质量，不满意则重新检索

---

### Q20: 如果要把你的系统升级为 Agent 系统，你会怎么设计？

**回答：**

升级为 Agent 系统需要以下改造：

**1. 工具层**
将现有能力封装为 Agent 可调用的工具：
```python
tools = [
    Tool("legal_search", rag.hybrid_retrieve_documents, "搜索法律文档"),
    Tool("knowledge_graph_query", graph.graph_search, "查询法律知识图谱"),
    Tool("web_search", web_search, "搜索互联网法律资源"),
    Tool("calculator", calculate, "法律金额计算"),
    Tool("case_database", query_cases, "查询相似案例"),
]
```

**2. 推理循环（ReAct）**
```
Thought: 用户问的是劳动合同解除的经济补偿，我需要先检索相关法条
Action: legal_search("劳动合同解除 经济补偿金")
Observation: 找到《劳动合同法》第46条、第47条...
Thought: 还需要确认计算标准，查询第47条的具体内容
Action: legal_search("经济补偿金 计算标准 工作年限")
Observation: 每满一年支付一个月工资...
Thought: 信息足够了，生成回答
Final Answer: 根据《劳动合同法》第46条...
```

**3. 规划层**
对复杂问题先做任务分解：
- 事实分析 → 法律检索 → 案例对比 → 结论生成 → 质量评估

**4. 记忆层**
保持现有的对话记忆，但增加长期记忆（跨会话的法律知识积累）

**改进建议：**
- Agent 系统的调试难度远高于 RAG，需要完善的 tracing（如 LangSmith）
- 工具调用需要严格的错误处理和重试机制
- 多步推理会显著增加延迟和成本，需要设置最大步数限制
- 需要引入自我反思机制，避免 Agent 陷入循环

---

## 九、部署与运维

### Q21: 你的系统是怎么部署的？有什么运维考量？

**回答：**

当前部署方案：
- 单机部署：`start.bat` 启动脚本自动启动 Redis、Neo4j、uvicorn
- uvicorn 单 worker 运行在 `127.0.0.1:8080`
- PostgreSQL 作为持久化存储
- Redis 用于缓存和限流
- Neo4j 用于知识图谱

运维考量：
- 启动时自动初始化：创建 FAISS 索引、BM25 索引、知识图谱（首次启动耗时 5-10 分钟）
- `SESSION_SECRET` 自动生成并写入 `.env`（多 worker 有竞态风险）
- 健康检查端点 `/health` 返回 Redis、DB、RAG 状态
- CI：GitHub Actions 做 ruff lint + pytest

**改进建议：**
- 单 worker 部署无法利用多核，应该用 gunicorn + uvicorn workers 或 Kubernetes 多副本
- 没有容器化（Docker），部署依赖本地环境
- 没有日志收集和监控（如 Prometheus + Grafana）
- 没有蓝绿部署或滚动更新能力
- 数据库 schema 在代码中 `create_all()`，应该用 Alembic 做版本化迁移
- 索引重建阻塞启动，应该异步初始化

---

## 十、设计模式与工程实践

### Q22: 你的项目中用了哪些设计模式？

**回答：**

| 模式 | 应用位置 | 说明 |
|------|---------|------|
| 策略模式 | Embedding/Reranker/LLM 提供商 | 通过环境变量切换，接口统一 |
| 装饰器模式 | `@redis_fallback` | 统一的 Redis 故障降级 |
| 三级缓存 | 对话记忆、KB 文本、检索结果 | Memory → Redis → DB |
| 两阶段锁 | `ConversationMemory.add_message()` | 内存操作在锁内，LLM 调用在锁外 |
| 懒加载 | PaddleOCR、FAISS、Neo4j | 首次使用才初始化 |
| 脏标记 | FAISS 索引 | 删除时不重建，查询时才重建 |
| 后台任务 | 文档上传处理 | FastAPI BackgroundTasks |
| 依赖注入 | ORM 模型、DB session | 避免循环导入 |
| 优雅降级 | Redis/Neo4j/Reranker | 故障时自动 fallback |

**改进建议：**
- 工厂模式缺失：Embedding/Reranker 的创建逻辑散落在 `__init__` 中，应该用工厂模式封装
- 中间件模式缺失：安全检查（CSRF、限流、注入检测）应该用 FastAPI 中间件统一处理
- 观察者模式缺失：文档上传后的索引更新应该用事件驱动，而不是直接调用

---

### Q23: 你在项目中遇到的最大技术挑战是什么？怎么解决的？

**回答：**

**挑战一：对话记忆的线程安全与性能平衡**

问题：LLM 摘要调用需要 500ms+，如果在锁内执行会阻塞所有其他线程。

解决：设计了两阶段锁方案。第一阶段在锁内快速检查 token 预算、确定分割点、提取需要摘要的消息。第二阶段在锁外调用 LLM。第三阶段重新获取锁写回摘要。关键是要确保第二阶段提取的消息列表是独立副本，不受其他线程修改。

**挑战二：三路检索的分数归一化**

问题：FAISS 返回的是距离（越小越好），BM25 返回的是分数（越大越好），Neo4j 返回的是自定义权重（0-1）。三种分数不在同一尺度。

解决：统一做 min-max 归一化到 [0,1] 范围。对于 FAISS 的距离，先转换为相似度（`1 / (1 + distance)`），再归一化。当某个检索路无结果时，跳过该路并归一化剩余路的权重。

**挑战三：法律文档的结构化分块**

问题：法律文档有特殊的「第X条」结构，普通分块会切断完整的法律条文，影响检索效果。

解决：实现 `DocumentSplitter`，基于法律条文边界分块（三层策略：条文边界 > 标点符号 > 递归字符分割）。129 个法律术语加入 jieba 词典防止错误分词。

**改进建议：**
- 这些解决方案虽然有效，但缺乏量化评估。应该用 RAGAS 等框架做系统性的效果对比实验
- 两阶段锁方案复杂度高，如果引入异步 IO（asyncio），可以用 async/await 更优雅地解决

---

## 十一、补充高频问题

### Q24: RAG 系统有哪些常见的评估指标？你怎么评估你的系统？

**回答：**

RAG 系统的评估分为检索质量和生成质量两个维度：

**检索质量：**
- **Recall@K**：前 K 个检索结果中包含相关文档的比例
- **Precision@K**：前 K 个检索结果中相关文档的比例
- **MRR**（Mean Reciprocal Rank）：第一个相关文档的排名倒数
- **NDCG**：考虑排名位置的增益

**生成质量：**
- **Faithfulness**：回答是否忠实于检索到的文档（不幻觉）
- **Answer Relevancy**：回答是否与问题相关
- **Context Relevancy**：检索到的上下文是否与问题相关

我的项目中有 RAGAS 评估框架（`tests/eval/`），通过人工标注的测试集评估系统效果。三轮迭代从 81.7% 提升到 90.0%。

**改进建议：**
- 缺少自动化的回归测试，每次代码变更应该自动运行评估
- 评估集应该持续扩充，覆盖更多边界情况
- 缺少线上评估（基于用户反馈的自动评估）
- 应该区分不同查询类型（简单/复杂/多跳）的评估结果

---

### Q25: 你怎么处理法律领域特有的问题（如术语歧义、多义词、新法旧法）？

**回答：**

**术语歧义与多义词：**
- 129 个法律术语强制分词，确保「善意取得」不被切成「善意/取得」
- 查询分析时做口语→法律术语改写（如「老板欠工资」→「用人单位拖欠劳动报酬」）
- 知识图谱的概念节点提供术语的法律定义，辅助消歧

**新法旧法：**
- 当前系统没有版本管理机制，文档直接覆盖
- 知识图谱中没有法律效力层级关系

**改进建议：**
- 应该引入法律版本管理：每部法律标注生效日期和失效日期，查询时根据时间过滤
- 对于新法旧法并存的情况，应该建立「废止」关系边
- 法律术语的消歧可以引入 BERT-based 的上下文消歧模型
- 应该建立法律概念的层级关系（如「违约责任」→「继续履行」「赔偿损失」「违约金」）

---

### Q26: 你对这个项目有什么反思？如果重新做会有什么不同？

**回答：**

**架构层面：**
1. `app.py` 单体文件过大（1200+ 行），应该从一开始就做好模块划分
2. 缺少 API 层和业务逻辑层的分离，路由处理器直接操作 ORM
3. 配置管理应该用 Pydantic Settings，而不是散落在各处的 `os.getenv`

**技术选型：**
1. BM25 用 `rank_bm25` 库不支持增量更新，应该用 Elasticsearch
2. FAISS 不支持元数据过滤，如果需要按知识库过滤应该用 Milvus/Qdrant
3. 对话记忆用 Redis 做 L2 缓存，但 Redis 故障时的降级方案（本地内存）不适合多实例部署

**工程实践：**
1. 缺少类型注解和 Pydantic schema，代码可维护性不够
2. 测试覆盖率不足，只有核心模块有单元测试
3. 没有集成测试和端到端测试
4. 缺少 CI/CD 流水线（目前只有 lint + unit test）
5. 没有日志收集和性能监控

**如果重新做：**
- 用 FastAPI + Pydantic 从一开始就做好 API schema
- 模块划分：`core/`（RAG）、`api/`（路由）、`models/`（ORM）、`services/`（业务逻辑）、`utils/`（工具）
- 用 Docker Compose 做本地开发环境
- 从第一天就写测试，目标覆盖率 80%+
- 引入 LangSmith/LangFuse 做 LLM 调用的 tracing

---

> 本文档共 26 个问题，覆盖 RAG 架构、检索系统、对话记忆、LLM 应用、安全设计、性能优化、Agent 设计、部署运维、工程实践等维度。每个问题的「改进建议」部分可以作为后续优化的参考。
