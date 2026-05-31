# 项目改进建议完整汇总

> 基于面试QA文档中所有问题的改进建议，按模块分类整理，新增「法律缺失时人工介入」机制。

---

## 一、架构与工程实践

### 1.1 `app.py` 单体拆分

**问题：** `app.py` 1200+ 行，ORM 模型、认证、路由、后台任务全在一个文件中

**方案：**
```
law_assistant/
├── models/          # SQLAlchemy ORM 模型
│   ├── user.py
│   ├── chat.py
│   ├── message.py
│   ├── knowledge_base.py
│   └── document.py
├── auth/            # 认证与会话
│   ├── session.py       # HMAC 签名、token 管理
│   ├── csrf.py          # CSRF 双提交 Cookie
│   └── dependencies.py  # require_user, get_db
├── api/             # 路由处理器
│   ├── chat.py          # /api/chats CRUD
│   ├── upload.py        # /upload 文档上传
│   ├── ask.py           # /ask_stream 问答
│   └── admin.py         # /api/retrieval-weights
├── services/        # 业务逻辑
│   ├── rag.py           # DeepSeekApiRag（从 law_assistant/rag.py 迁移）
│   ├── memory.py        # ConversationMemory
│   └── processor.py     # DocumentProcessor
├── core/            # 核心检索组件
│   ├── bm25.py
│   ├── graph.py
│   ├── security.py
│   └── redis_utils.py
└── middleware/       # 中间件
    ├── rate_limit.py
    ├── security_headers.py
    └── injection.py
```

**收益：** 职责清晰、可独立测试、便于多人协作

---

### 1.2 数据库迁移管理

**问题：** `Base.metadata.create_all(engine)` 在模块导入时执行，无法管理 schema 变更

**方案：**
```bash
pip install alembic
alembic init alembic
```
- 用 Alembic 管理数据库版本
- `create_all()` 改为仅在开发环境执行
- 生产环境通过 `alembic upgrade head` 做增量迁移

---

### 1.3 配置管理标准化

**问题：** 散落在各处的 `os.getenv()` 调用，缺乏类型校验

**方案：**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM
    mimo_api_key: str
    deepseek_api_key: str | None = None
    
    # Embedding
    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    
    # Reranker
    reranker_provider: str = "local"
    reranker_model_path: str = "BAAI/bge-reranker-v2-m3"
    
    # Retrieval weights
    vector_retrieval_weight: float = 0.4
    bm25_retrieval_weight: float = 0.3
    graph_retrieval_weight: float = 0.3
    relevance_threshold: float = 0.15
    
    # Session
    session_secret: str
    session_expire_hours: int = 24
    
    # External services
    redis_url: str = "redis://localhost:6379/0"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()  # 启动时校验，缺失必填项直接报错
```

---

### 1.4 引入 Pydantic Schema

**问题：** 缺少类型注解和 API schema

**方案：** 为所有 API 请求/响应定义 Pydantic 模型：
```python
from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    chat_id: int
    query: str = Field(..., max_length=2000)
    knowledge_base_id: int | None = None

class AskResponse(BaseModel):
    content: str
    sources: list[SourceItem]
    message_id: int
```

**收益：** 自动生成 OpenAPI 文档、运行时类型校验、IDE 补全

---

### 1.5 引入中间件模式

**问题：** 安全检查（CSRF、限流、注入检测）散落在路由处理器中

**方案：** 用 FastAPI 中间件统一处理：
```python
@app.middleware("http")
async def security_middleware(request: Request, call_next):
    # 1. 限流检查
    # 2. CSRF 验证（状态变更请求）
    # 3. 注入检测（/ask_stream）
    response = await call_next(request)
    # 4. 添加安全头
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response
```

---

### 1.6 引入工厂模式

**问题：** Embedding/Reranker/LLM 的创建逻辑散落在 `__init__` 中

**方案：**
```python
class EmbeddingFactory:
    @staticmethod
    def create(provider: str, **kwargs):
        if provider == "dashscope":
            return DashScopeEmbedding(...)
        elif provider == "local":
            return HuggingFaceEmbedding(...)
        raise ValueError(f"Unknown provider: {provider}")

class RerankerFactory:
    @staticmethod
    def create(provider: str, **kwargs):
        if provider == "local":
            return CrossEncoderReranker(...)
        elif provider == "dashscope":
            return DashScopeReranker(...)
        raise ValueError(f"Unknown provider: {provider}")
```

---

### 1.7 引入事件驱动

**问题：** 文档上传后直接调用索引更新，耦合度高

**方案：** 用观察者模式解耦：
```python
class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = defaultdict(list)
    
    def subscribe(self, event: str, handler: Callable):
        self._handlers[event].append(handler)
    
    def publish(self, event: str, data: Any):
        for handler in self._handlers[event]:
            handler(data)

# 注册事件
event_bus.subscribe("document_uploaded", index_to_faiss)
event_bus.subscribe("document_uploaded", index_to_bm25)
event_bus.subscribe("document_uploaded", build_knowledge_graph)
event_bus.subscribe("document_deleted", remove_from_indexes)
```

---

## 二、检索系统

### 2.1 FAISS 索引重建从统一数据源

**问题：** `rebuild_faiss_if_dirty` 从 BM25 文档列表重建，可能导致数据不一致

**方案：** 维护一个统一的文档注册表（数据库），所有索引从同一数据源构建：
```python
def rebuild_faiss_if_dirty(self):
    if self._faiss_dirty:
        # 从数据库获取所有已索引文档，而非从 BM25
        all_docs = self.db_session.query(IndexedDocument).all()
        texts = [doc.content for doc in all_docs]
        metadatas = [doc.metadata for doc in all_docs]
        self.vector_db = FAISS.from_texts(texts, self.embedding_model, metadatas=metadatas)
        self._faiss_dirty = False
```

---

### 2.2 BM25 `remove_documents` 锁外预分词

**问题：** `remove_documents` 在锁内做分词，阻塞其他线程

**方案：**
```python
def remove_documents(self, target_texts):
    target_set = set(target_texts)
    
    # 锁外预处理
    with self._lock:
        remaining = [doc for doc in self.documents if doc not in target_set]
        remaining_pending = [doc for doc in self.pending_documents if doc not in target_set]
        self.documents = remaining
        self.pending_documents = remaining_pending
    
    # 锁外重建索引
    if remaining:
        tokenized = [self._tokenize(doc) for doc in remaining]
        new_bm25 = BM25Okapi(tokenized)
        
        with self._lock:
            self.bm25 = new_bm25
            self.tokenized_documents = tokenized
```

---

### 2.3 RRF 融合策略

**问题：** 加权求和依赖分数归一化，不同检索器的分数尺度不同

**方案：** 引入 Reciprocal Rank Fusion：
```python
def reciprocal_rank_fusion(results_list: list[list], k: int = 60) -> list:
    """
    results_list: 每个检索器返回的 [(doc, score), ...] 列表
    k: 常数，控制排名影响衰减速度
    """
    fused_scores = defaultdict(float)
    for results in results_list:
        for rank, (doc, _) in enumerate(results, 1):
            fused_scores[doc] += 1.0 / (k + rank)
    
    return sorted(fused_scores.items(), key=lambda x: -x[1])
```

**优势：** 不依赖分数归一化，对不同检索器更鲁棒

---

### 2.4 权重自适应学习

**问题：** 检索权重是静态经验值

**方案：** 基于用户反馈自动优化：
```python
def update_weights_from_feedback(self, query, selected_docs, feedback: bool):
    """
    feedback=True: 用户点了 thumbs up，被选中的文档所在的检索路权重上调
    feedback=False: 用户点了 thumbs down，权重下调
    """
    for doc in selected_docs:
        source = doc.metadata.get("retrieval_source")  # "vector" | "bm25" | "graph"
        if feedback:
            self.weights[source] *= 1.05  # 上调 5%
        else:
            self.weights[source] *= 0.95  # 下调 5%
    
    # 归一化
    total = sum(self.weights.values())
    self.weights = {k: v / total for k, v in self.weights.items()}
```

---

### 2.5 相关性阈值可配置

**问题：** `RELEVANCE_THRESHOLD = 0.15` 硬编码

**方案：** 加入环境变量 + 管理员 API 动态调整：
```python
self.relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.15"))

# 管理员 API
@app.post("/api/relevance-threshold")
async def update_threshold(threshold: float = Body(..., ge=0.0, le=1.0)):
    rag_model.relevance_threshold = threshold
    return {"threshold": threshold}
```

---

## 三、对话记忆

### 3.1 摘要截断修复

**问题：** `combined[-600:]` 取后 600 字符，丢失摘要开头

**方案：**
```python
# 改为保留前 600 字符
if len(combined) > self.SUMMARY_MAX_CHARS:
    combined = combined[:self.SUMMARY_MAX_CHARS]

# 或者更好的方案：让 LLM 重新总结
if len(combined) > self.SUMMARY_MAX_CHARS:
    combined = self.summarizer(combined)  # 用 LLM 压缩到 600 字符以内
```

---

### 3.2 L1 缓存添加 TTL

**问题：** L1 内存缓存没有 TTL，长时间不活跃的对话一直占用内存

**方案：**
```python
class ConversationMemory:
    def __init__(self, max_conversations=500, idle_timeout=3600):
        self._cache = OrderedDict()  # chat_id -> (data, last_access_time)
        self._idle_timeout = idle_timeout
    
    def _evict_idle(self):
        """淘汰空闲超过 idle_timeout 的对话"""
        now = time.time()
        while self._cache:
            chat_id, (_, last_access) = next(iter(self._cache.items()))
            if now - last_access > self._idle_timeout:
                self._cache.popitem(last=False)
            else:
                break
    
    def get_recent_history(self, conversation_id):
        self._evict_idle()  # 访问时清理
        # ... 原有逻辑
```

---

### 3.3 Token 估算精确化

**问题：** `len(text) * 2 // 3` 对中英混合文本误差 30-50%

**方案：** 引入 tiktoken：
```python
import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")  # 或针对中文优化的编码

def estimate_tokens(text: str) -> int:
    return len(_encoder.encode(text))
```

**对比：**

| 方法 | 中文文本 | 英文文本 | 中英混合 |
|------|---------|---------|---------|
| `len(text) * 2 // 3` | ~0.67 tokens/char | ~0.67 tokens/char | 误差大 |
| tiktoken cl100k | ~1.5 tokens/char | ~0.25 tokens/char | 准确 |

---

### 3.4 `_write_to_redis` 线程安全修复

**问题：** 在锁外调用 `_write_to_redis`，依赖 CPython GIL

**方案：**
```python
def add_message(self, conversation_id, role, content):
    with self._lock:
        history.append(...)
        snapshot = copy.deepcopy(memory)  # 在锁内做深拷贝
    
    # 锁外操作使用独立副本
    if should_summarize:
        summary = self.summarizer(old_messages)
        with self._lock:
            memory["summary"] = summary
    
    # Redis 写入用独立副本
    self._write_to_redis(conversation_id, snapshot)
```

---

### 3.5 摘要用小模型

**问题：** 摘要生成消耗 LLM 调用，成本高

**方案：**
- 用更小的模型做摘要（如 `qwen-turbo` 而非 `qwen-plus`）
- 或用抽取式摘要（extractive）替代生成式摘要（abstractive）：
```python
def extractive_summary(messages, max_chars=600):
    """简单抽取式摘要：保留每条消息的前 N 个字符"""
    result = []
    for msg in messages:
        truncated = msg["content"][:100]  # 每条消息保留前 100 字符
        result.append(f"{msg['role']}: {truncated}")
    return "\n".join(result)[:max_chars]
```

---

## 四、LLM 应用与 Prompt

### 4.1 查询分析输出校验

**问题：** LLM 输出不稳定，JSON 解析可能失败

**方案：**
```python
from pydantic import BaseModel, validator

class QueryAnalysisResult(BaseModel):
    rewritten_query: str
    sub_queries: list[str] = Field(max_length=3)
    hypothetical_doc: str
    
    @validator("hypothetical_doc")
    def check_length(cls, v):
        if len(v) < 100 or len(v) > 800:
            raise ValueError("Hypothetical doc length out of range")
        return v

def analyze_query(self, query, conversation_id, conversation_history):
    for attempt in range(3):  # 最多重试 3 次
        try:
            response = self.llm.invoke(prompt)
            result = QueryAnalysisResult.parse_raw(response.content)
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            if attempt == 2:
                # 降级：返回原始查询，不分解
                return QueryAnalysisResult(
                    rewritten_query=query,
                    sub_queries=[],
                    hypothetical_doc=""
                )
            continue
```

---

### 4.2 口语→法律术语映射外置

**问题：** 映射规则硬编码在 Prompt 中

**方案：** 外置到 YAML 配置文件：
```yaml
# legal_term_mapping.yaml
mappings:
  老板: 用人单位
  欠工资: 拖欠劳动报酬
  被开除: 被解除劳动合同
  离职补偿: 经济补偿金
  加班费: 加班工资
  试用期: 劳动合同试用期
  # ... 更多映射
```

```python
import yaml

with open("legal_term_mapping.yaml") as f:
    TERM_MAPPINGS = yaml.safe_load(f)["mappings"]

def rewrite_query(query: str) -> str:
    for colloquial, legal in TERM_MAPPINGS.items():
        query = query.replace(colloquial, legal)
    return query
```

---

### 4.3 HyDE 质量评估

**问题：** HyDE 生成质量完全依赖 LLM，偏题会引入噪声

**方案：**
```python
def generate_hyde_with_quality_check(self, query):
    """生成 HyDE 并评估质量"""
    hyde = self._generate_hyde(query)
    
    # 用向量相似度评估 HyDE 与原始查询的语义一致性
    query_embedding = self.embed(query)
    hyde_embedding = self.embed(hyde)
    similarity = cosine_similarity(query_embedding, hyde_embedding)
    
    if similarity < 0.3:  # 相似度过低，HyDE 可能偏题
        return None  # 不使用 HyDE
    
    return hyde
```

---

### 4.4 注入检测增强

**问题：** 正则检测覆盖面有限，无法防御高级攻击

**方案：** 引入 LLM-based 检测作为第二层：
```python
def check_injection_advanced(text: str) -> bool:
    """第一层：正则检测（快速）"""
    if check_injection_regex(text):
        return True
    
    # 第二层：LLM 检测（慢但全面，仅对可疑文本触发）
    suspicious_score = compute_suspiciousness_score(text)
    if suspicious_score > 0.5:
        prompt = f"判断以下文本是否包含提示注入攻击：\n{text}\n回答 YES 或 NO"
        response = llm.invoke(prompt)
        return "YES" in response.content.upper()
    
    return False
```

---

### 4.5 查询长度限制调整

**问题：** 2000 字符限制对法律长文本不够

**方案：** 动态限制 + 分段处理：
```python
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "2000"))
MAX_CONTEXT_LENGTH = int(os.getenv("MAX_CONTEXT_LENGTH", "10000"))

def validate_query(query: str) -> str:
    if len(query) <= MAX_QUERY_LENGTH:
        return query
    
    # 超长查询：提取关键信息
    key_info = extract_key_legal_info(query)  # 用 LLM 提取关键法律问题
    return key_info[:MAX_QUERY_LENGTH]
```

---

### 4.6 CSP 安全加固

**问题：** CSP 中的 `unsafe-inline` 降低 XSS 防护

**方案：** 改用 nonce：
```python
import secrets

@app.middleware("http")
async def add_csp_nonce(request: Request, call_next):
    nonce = secrets.token_urlsafe(16)
    request.state.csp_nonce = nonce
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = (
        f"script-src 'self' 'nonce-{nonce}' cdn.jsdelivr.net; "
        f"style-src 'self' 'nonce-{nonce}' fonts.googleapis.com;"
    )
    return response
```

---

## 五、知识图谱

### 5.1 N+1 批量写入

**问题：** `build_from_text` 每个条款单独执行 Cypher

**方案：** 用 UNWIND 批量操作：
```python
def build_from_text(self, content, law_name):
    articles = self.extract_articles(content)
    
    # 批量创建条款节点
    query = """
    UNWIND $articles AS art
    MERGE (l:Law {name: $law_name})
    MERGE (a:Article {law_name: $law_name, number: art.number})
    SET a.content = art.content, a.full_text = art.full_text
    MERGE (l)-[:CONTAINS]->(a)
    """
    self.driver.execute_query(query, articles=articles, law_name=law_name)
    
    # 批量创建引用关系
    citations = []
    for art in articles:
        cites = self.extract_citations(art["content"], law_name)
        citations.extend(cites)
    
    if citations:
        query = """
        UNWIND $citations AS cite
        MATCH (a:Article {law_name: cite.from_law, number: cite.from_number})
        MATCH (b:Article {law_name: cite.to_law, number: cite.to_number})
        MERGE (a)-[:CITES]->(b)
        """
        self.driver.execute_query(query, citations=citations)
```

---

### 5.2 清理死代码

**问题：** `extract_citations` 定义了 `clause` 类型但从未处理

**方案：** 删除未使用的 `clause` 模式，或补全处理逻辑：
```python
CITE_PATTERMS = {
    "self": r"依照本法第([零一二三四五六七八九十百千万\d]+)条",
    "short_self": r"本法第([零一二三四五六七八九十百千万\d]+)条",
    "cross": r"依据《(.+?)》第([零一二三四五六七八九十百千万\d]+)条",
    # 移除未使用的 "clause" 模式
}
```

---

### 5.3 Cypher 参数化修复

**问题：** `get_stats` 用 f-string 拼接 Cypher

**方案：** 用参数化查询：
```python
def get_stats(self):
    stats = {}
    for label in ["Law", "Chapter", "Article", "Concept"]:
        result = self.driver.execute_query(
            "MATCH (n:$label) RETURN count(n) AS count",
            label=label  # 注意：标签名不能参数化，但有白名单校验
        )
        stats[label] = result[0][0]["count"]
    return stats
```

---

### 5.4 Neo4j 连接池

**问题：** 单连接，高并发下成为瓶颈

**方案：**
```python
from neo4j import GraphDatabase

class LegalKnowledgeGraph:
    def __init__(self):
        self._driver = None
        self._max_connection_pool_size = 50
    
    def connect(self):
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            max_connection_pool_size=self._max_connection_pool_size,
            connection_acquisition_timeout=30,
        )
```

---

## 六、安全

### 6.1 文件上传 MIME 校验

**问题：** 不信任前端传的 file_type

**方案：**
```python
import magic

def validate_file_type(file_path: str) -> str:
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    
    ALLOWED_TYPES = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "text/plain": ".txt",
        "image/jpeg": ".jpg",
        "image/png": ".png",
    }
    
    if file_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"不支持的文件类型: {file_type}")
    
    return ALLOWED_TYPES[file_type]
```

---

### 6.2 文件名安全处理

**问题：** 文件名可能存在路径穿越

**方案：**
```python
import re
import uuid

def safe_filename(original_name: str) -> str:
    # 移除路径分隔符和特殊字符
    name = re.sub(r'[/\\:*?"<>|]', '', original_name)
    # 添加 UUID 防止冲突
    ext = os.path.splitext(name)[1]
    return f"{uuid.uuid4().hex}{ext}"
```

---

### 6.3 CSRF 覆盖 SSE

**问题：** `/ask_stream` 跳过了 CSRF 校验

**方案：** SSE 端点也做 CSRF 校验（从 cookie 读取 token，通过 query param 传递）：
```python
@app.get("/ask_stream")
async def ask_stream(
    request: Request,
    chat_id: int = Query(...),
    query: str = Query(...),
    csrf_token: str = Query(...),  # 从 URL 参数传递
):
    if not verify_csrf(request.cookies.get("session_token"), csrf_token):
        raise HTTPException(403, "CSRF validation failed")
    # ...
```

---

### 6.4 SESSION_SECRET 竞态修复

**问题：** 多 worker 并发时自动生成 SECRET 有竞态

**方案：** 启动前检查，不自动写入：
```python
def ensure_session_secret():
    """启动时检查，不自动写入 .env"""
    secret = os.getenv("SESSION_SECRET")
    if not secret:
        raise RuntimeError(
            "SESSION_SECRET 未设置。请在 .env 文件中配置 SESSION_SECRET。"
            "可以使用: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
        )
    return secret
```

---

## 七、性能优化

### 7.1 A/B 测试框架

**问题：** 无法量化各优化的效果

**方案：**
```python
class ABTestManager:
    def __init__(self):
        self.experiments = {}
    
    def assign_variant(self, user_id: int, experiment: str) -> str:
        """基于 user_id 哈希确定分组，保证同一用户始终在同一组"""
        hash_val = hash(f"{user_id}:{experiment}") % 100
        if hash_val < 50:
            return "control"
        return "treatment"
    
    def log_result(self, experiment: str, variant: str, metric: str, value: float):
        """记录实验结果"""
        # 存入数据库或 Redis
        pass
```

---

### 7.2 Redis Key 命名空间

**问题：** key 设计没有命名空间规范

**方案：** 统一 key 前缀：
```python
class RedisKey:
    PREFIX = "law_assistant:"
    
    # 缓存
    RETRIEVAL_CACHE = PREFIX + "retrieval:{query_hash}:{kb_id}"
    KB_TEXT_CACHE = PREFIX + "kb_text:{kb_id}"
    CONVERSATION = PREFIX + "conversation:{chat_id}"
    
    # 限流
    RATE_LIMIT = PREFIX + "ratelimit:{identifier}:{window}"
    
    # 黑名单
    TOKEN_BLACKLIST = PREFIX + "blacklist:{token_hash}"
    
    @staticmethod
    def format(key_template: str, **kwargs) -> str:
        return key_template.format(**kwargs)
```

---

### 7.3 连接池复用

**问题：** Neo4j、Redis 没有统一的连接管理

**方案：** 引入连接池管理器：
```python
class ConnectionPool:
    def __init__(self):
        self._redis = None
        self._neo4j = None
    
    def get_redis(self):
        if self._redis is None:
            self._redis = redis.ConnectionPool.from_url(settings.redis_url)
        return self._redis
    
    def get_neo4j(self):
        if self._neo4j is None:
            self._neo4j = GraphDatabase.driver(settings.neo4j_uri)
        return self._neo4j
    
    def close_all(self):
        if self._redis:
            self._redis.disconnect()
        if self._neo4j:
            self._neo4j.close()
```

---

## 八、部署与运维

### 8.1 Docker 化

**方案：**
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8080:8080"
    env_file: .env
    depends_on:
      - redis
      - postgres
      - neo4j
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: law_assistant
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
  
  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4jdata:/data

volumes:
  pgdata:
  neo4jdata:
```

---

### 8.2 日志与监控

**方案：**
```python
import logging
from prometheus_client import Counter, Histogram, generate_latest

# 结构化日志
logging.basicConfig(
    format='{"time":"%(asctime)s","level":"%(levelname)s","module":"%(module)s","message":"%(message)s"}',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Prometheus 指标
REQUEST_COUNT = Counter('http_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
RETRIEVAL_LATENCY = Histogram('retrieval_latency_seconds', 'Retrieval latency', ['path'])
LLM_CALL_COUNT = Counter('llm_calls_total', 'LLM API calls', ['model', 'status'])

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

---

### 8.3 异步初始化

**问题：** 索引重建阻塞启动

**方案：**
```python
@app.on_event("startup")
async def startup():
    # 先启动 HTTP 服务
    asyncio.create_task(initialize_indexes_async())

async def initialize_indexes_async():
    """后台异步初始化索引"""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, rag_model.initialize_vector_database)
    logger.info("索引初始化完成")
```

---

### 8.4 蓝绿部署

**方案：**
```yaml
# Kubernetes 部署
apiVersion: apps/v1
kind: Deployment
metadata:
  name: law-assistant-blue
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      containers:
        - name: app
          image: law-assistant:v2.0
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 30
```

---

## 九、评估与测试

### 9.1 自动化回归测试

**方案：**
```python
# tests/eval/regression.py
import pytest
from ragas import evaluate
from datasets import Dataset

@pytest.mark.eval
def test_rag_regression():
    """每次代码变更自动运行"""
    test_data = load_test_dataset("tests/eval/test_set.json")
    results = evaluate(
        dataset=test_data,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    
    # 质量门禁
    assert results["faithfulness"] >= 0.85, f"Faithfulness 下降: {results['faithfulness']}"
    assert results["answer_relevancy"] >= 0.80, f"Relevancy 下降: {results['answer_relevancy']}"
```

---

### 9.2 线上评估

**方案：** 基于用户反馈自动评估：
```python
def collect_online_metrics():
    """定期收集线上评估指标"""
    feedbacks = db.query(MessageFeedback).filter(
        MessageFeedback.created_at >= datetime.now() - timedelta(hours=24)
    ).all()
    
    upvotes = sum(1 for f in feedbacks if f.rating == "up")
    downvotes = sum(1 for f in feedbacks if f.rating == "down")
    
    satisfaction_rate = upvotes / (upvotes + downvotes) if (upvotes + downvotes) > 0 else 0
    
    # 写入 Prometheus
    ONLINE_SATISFACTION.set(satisfaction_rate)
```

---

### 9.3 区分查询类型评估

**方案：**
```python
QUERY_TYPES = {
    "simple": "单一法律问题，如「什么是善意取得」",
    "complex": "复合法律问题，如「劳动合同解除的条件和补偿」",
    "multi_hop": "多跳推理，如「如果公司破产，员工的工资和社保谁来承担」",
    "conversational": "多轮对话，需要上下文理解",
}

# 每种类型独立评估
for query_type, test_set in test_sets.items():
    results = evaluate(test_set)
    logger.info(f"{query_type}: {results}")
```

---

## 十、法律缺失时人工介入机制（新增）

### 10.1 设计目标

当系统无法从知识库中找到足够的法律依据来回答用户问题时，不应强行编造答案，而应：
1. 明确告知用户当前知识库的局限性
2. 请求人工介入（转接人工法律咨询）
3. 记录未覆盖的法律问题，用于后续知识库扩充

### 10.2 置信度评估

在生成回答前，评估检索结果的充分性：

```python
class ConfidenceEvaluator:
    """评估检索结果是否足够回答用户问题"""
    
    # 置信度阈值
    HIGH_CONFIDENCE = 0.7    # 高置信度：直接回答
    LOW_CONFIDENCE = 0.3     # 低置信度：部分回答 + 提示人工介入
    NO_CONFIDENCE = 0.0      # 无置信度：完全无法回答
    
    def evaluate(self, query: str, retrieved_docs: list, reranker_scores: list) -> dict:
        """
        评估检索结果的充分性
        返回: {"level": "high"|"low"|"none", "score": float, "reason": str}
        """
        if not retrieved_docs:
            return {
                "level": "none",
                "score": 0.0,
                "reason": "未检索到相关法律文档"
            }
        
        # 1. 最高 reranker 分数
        max_score = max(reranker_scores) if reranker_scores else 0.0
        
        # 2. 高分文档数量
        high_score_count = sum(1 for s in reranker_scores if s >= self.LOW_CONFIDENCE)
        
        # 3. 文档覆盖度（是否涉及多个法律领域）
        unique_laws = set(doc.metadata.get("law_name", "") for doc in retrieved_docs)
        
        # 综合评分
        coverage_score = min(high_score_count / 3, 1.0)  # 至少 3 篇高分文档
        diversity_score = min(len(unique_laws) / 2, 1.0)  # 至少涉及 2 部法律
        
        final_score = max_score * 0.5 + coverage_score * 0.3 + diversity_score * 0.2
        
        if final_score >= self.HIGH_CONFIDENCE:
            level = "high"
            reason = "检索到充分的法律依据"
        elif final_score >= self.LOW_CONFIDENCE:
            level = "low"
            reason = f"检索到部分相关文档（{high_score_count}篇），但覆盖不够全面"
        else:
            level = "none"
            reason = f"检索结果不足（最高分: {max_score:.2f}），可能缺少相关法律知识"
        
        return {"level": level, "score": final_score, "reason": reason}
```

### 10.3 分级响应策略

```python
class ResponseStrategy:
    """根据置信度选择响应策略"""
    
    def generate_response(self, query, confidence, retrieved_docs, conversation_history):
        if confidence["level"] == "high":
            # 高置信度：正常 RAG 回答
            return self._normal_rag_response(query, retrieved_docs, conversation_history)
        
        elif confidence["level"] == "low":
            # 低置信度：部分回答 + 免责声明 + 人工介入提示
            partial_answer = self._normal_rag_response(query, retrieved_docs, conversation_history)
            return self._append_disclaimer(partial_answer, confidence["reason"])
        
        else:
            # 无置信度：无法回答 + 人工介入请求
            return self._human_intervention_response(query, confidence["reason"])
    
    def _append_disclaimer(self, answer: str, reason: str) -> str:
        """追加免责声明和人工介入提示"""
        disclaimer = f"""
---

> **温馨提示：**
> 当前知识库可能未完全覆盖您的问题（{reason}）。
> 以上回答仅供参考，如需更专业的法律建议，请点击下方按钮联系人工律师咨询。
>
> [转人工咨询]
"""
        return answer + disclaimer
    
    def _human_intervention_response(self, query: str, reason: str) -> str:
        """完全无法回答时的响应"""
        return f"""
感谢您的提问。

**当前知识库暂未收录相关法律规定**，无法为您提供准确的法律建议。

可能的原因：
- 该问题涉及的地方法规或行业规定未纳入知识库
- 问题涉及的法律领域超出当前覆盖范围
- 问题表述较为复杂，需要人工分析

**建议您：**
1. 点击下方按钮 **[转人工咨询]**，我们的专业律师将为您提供帮助
2. 您的问题已被记录，我们会尽快补充相关法律知识

> 您的问题已被记录（ID: {{query_id}}），如有更新将第一时间通知您。
"""
```

### 10.4 人工介入转接

```python
class HumanInterventionManager:
    """管理人工介入请求"""
    
    def create_intervention_request(self, user_id: int, chat_id: int, query: str, 
                                     confidence: dict, retrieved_docs: list) -> int:
        """创建人工介入请求"""
        request = InterventionRequest(
            user_id=user_id,
            chat_id=chat_id,
            original_query=query,
            confidence_level=confidence["level"],
            confidence_score=confidence["score"],
            confidence_reason=confidence["reason"],
            retrieved_doc_count=len(retrieved_docs),
            status="pending",  # pending -> assigned -> completed
            created_at=datetime.now(),
        )
        db.add(request)
        db.commit()
        
        # 通知管理员（邮件/钉钉/微信）
        self._notify_admins(request)
        
        return request.id
    
    def _notify_admins(self, request: InterventionRequest):
        """通知管理员有新的人工介入请求"""
        admins = db.query(User).filter(User.role == "admin").all()
        for admin in admins:
            # 发送通知（邮件/站内信/WebSocket）
            send_notification(
                user_id=admin.id,
                title="新的人工法律咨询请求",
                content=f"用户 #{request.user_id} 的问题需要人工介入：{request.original_query[:100]}...",
                link=f"/admin/interventions/{request.id}"
            )
```

### 10.5 前端交互

```html
<!-- 人工介入按钮 -->
<div class="human-intervention-banner" id="interventionBanner" style="display: none;">
    <div class="banner-content">
        <span class="banner-icon">⚖️</span>
        <span class="banner-text">
            当前回答仅供参考，需要专业律师进一步确认？
        </span>
        <button class="btn btn-accent" onclick="requestHumanIntervention()">
            转人工咨询
        </button>
    </div>
</div>

<script>
function requestHumanIntervention() {
    fetch('/api/intervention', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': getCsrfToken()
        },
        body: JSON.stringify({
            chat_id: currentChatId,
            query_id: currentQueryId
        })
    }).then(res => res.json()).then(data => {
        if (data.success) {
            showToast('已提交人工咨询请求，律师将在24小时内回复');
            document.getElementById('interventionBanner').style.display = 'none';
        }
    });
}
</script>
```

### 10.6 知识库扩充闭环

```python
class KnowledgeGapTracker:
    """跟踪知识库缺失的法律问题"""
    
    def record_gap(self, query: str, confidence: dict, user_id: int):
        """记录知识库缺口"""
        gap = KnowledgeGap(
            query=query,
            confidence_level=confidence["level"],
            confidence_reason=confidence["reason"],
            user_id=user_id,
            frequency=1,
            status="open",  # open -> researched -> added
            created_at=datetime.now(),
        )
        
        # 检查是否已有类似缺口
        existing = db.query(KnowledgeGap).filter(
            KnowledgeGap.query.like(f"%{query[:20]}%"),
            KnowledgeGap.status == "open"
        ).first()
        
        if existing:
            existing.frequency += 1  # 频率 +1
        else:
            db.add(gap)
        
        db.commit()
    
    def get_top_gaps(self, limit: int = 20) -> list:
        """获取最高频的知识库缺口，用于指导知识库扩充"""
        return db.query(KnowledgeGap).filter(
            KnowledgeGap.status == "open"
        ).order_by(
            KnowledgeGap.frequency.desc()
        ).limit(limit).all()
```

### 10.7 数据库表设计

```python
class InterventionRequest(Base):
    """人工介入请求"""
    __tablename__ = "intervention_request"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"))
    chat_id = Column(Integer, ForeignKey("chat.id"))
    original_query = Column(Text)
    confidence_level = Column(String(20))  # high/low/none
    confidence_score = Column(Float)
    confidence_reason = Column(Text)
    retrieved_doc_count = Column(Integer)
    status = Column(String(20), default="pending")  # pending/assigned/completed
    assigned_to = Column(Integer, ForeignKey("user.id"), nullable=True)
    response = Column(Text, nullable=True)  # 人工回复
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, onupdate=datetime.now)


class KnowledgeGap(Base):
    """知识库缺口"""
    __tablename__ = "knowledge_gap"
    
    id = Column(Integer, primary_key=True)
    query = Column(Text)
    confidence_level = Column(String(20))
    confidence_reason = Column(Text)
    user_id = Column(Integer, ForeignKey("user.id"))
    frequency = Column(Integer, default=1)  # 被问到的次数
    status = Column(String(20), default="open")  # open/researched/added
    notes = Column(Text, nullable=True)  # 研究笔记
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, onupdate=datetime.now)
```

### 10.8 集成到主流程

```python
# rag.py:generate_response_stream() 中集成

def generate_response_stream(self, query, chat_id, conversation_id, ...):
    # 1. 查询分析
    analysis = self.analyze_query(query, conversation_id, conversation_history)
    
    # 2. 检索
    retrieved_docs = self.retrieve_documents(
        analysis.rewritten_query, top_k=10,
        sub_queries=analysis.sub_queries,
        hypothetical_doc=analysis.hypothetical_doc,
        knowledge_base_id=knowledge_base_id,
        db_session=db_session
    )
    
    # 3. 置信度评估（新增）
    confidence = self.confidence_evaluator.evaluate(
        query, retrieved_docs, [doc[1] for doc in retrieved_docs]
    )
    
    # 4. 记录知识库缺口（新增）
    if confidence["level"] in ("low", "none"):
        self.gap_tracker.record_gap(query, confidence, user_id)
    
    # 5. 根据置信度选择响应策略（新增）
    if confidence["level"] == "none":
        # 无置信度：不调用 LLM，直接返回人工介入提示
        response = self.response_strategy._human_intervention_response(query, confidence["reason"])
        # 创建人工介入请求（新增）
        intervention_id = self.intervention_manager.create_intervention_request(
            user_id, chat_id, query, confidence, retrieved_docs
        )
        yield json.dumps({"content": response, "intervention_id": intervention_id})
        yield json.dumps({"done": True})
        return
    
    # 6. 构建上下文（对 low 置信度追加免责声明）
    context = self._build_context(retrieved_docs, analysis.rewritten_query)
    
    # 7. 流式生成
    prompt = self._build_prompt(query, context, conversation_history)
    stream = self.llm.stream(prompt)
    
    for chunk in stream:
        yield json.dumps({"content": chunk.content})
    
    # 8. 对 low 置信度追加免责声明（新增）
    if confidence["level"] == "low":
        disclaimer = self.response_strategy._get_disclaimer(confidence["reason"])
        yield json.dumps({"content": disclaimer, "show_intervention_banner": True})
    
    yield json.dumps({"done": True, "confidence_level": confidence["level"]})
```

---

## 改进优先级排序

| 优先级 | 改进项 | 影响范围 | 工作量 |
|--------|--------|---------|--------|
| **P0** | 法律缺失人工介入机制 | 核心功能 | 中 |
| **P0** | 摘要截断修复 | 数据正确性 | 小 |
| **P0** | 文件名安全处理 | 安全 | 小 |
| **P1** | Token 估算精确化 | 检索质量 | 小 |
| **P1** | BM25 remove_documents 锁优化 | 性能 | 小 |
| **P1** | 查询分析输出校验+重试 | 稳定性 | 小 |
| **P1** | 文件上传 MIME 校验 | 安全 | 小 |
| **P2** | FAISS 从统一数据源重建 | 数据一致性 | 中 |
| **P2** | RRF 融合策略 | 检索质量 | 中 |
| **P2** | L1 缓存 TTL | 内存管理 | 小 |
| **P2** | 知识图谱批量写入 | 性能 | 中 |
| **P2** | 日志与监控 | 运维 | 中 |
| **P3** | app.py 模块拆分 | 可维护性 | 大 |
| **P3** | Pydantic Settings 配置管理 | 可维护性 | 中 |
| **P3** | Alembic 数据库迁移 | 运维 | 中 |
| **P3** | Docker 化 | 部署 | 中 |
| **P3** | 工厂模式/中间件模式 | 架构 | 大 |
| **P4** | A/B 测试框架 | 数据驱动 | 大 |
| **P4** | 线上评估 | 质量保证 | 中 |
| **P4** | 权重自适应学习 | 检索质量 | 大 |
| **P4** | Agent 系统升级 | 架构升级 | 极大 |

---

> 本文档共 10 大类 40+ 项改进建议，覆盖架构、检索、记忆、LLM、知识图谱、安全、性能、部署、测试、人工介入等维度。建议按优先级分批实施，P0 项应在下一迭代完成。
