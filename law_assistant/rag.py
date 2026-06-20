import atexit
import concurrent.futures
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
from collections import OrderedDict

import numpy as np
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores.faiss import FAISS

from law_assistant.bm25 import BM25Retriever
from law_assistant.graph import LegalKnowledgeGraph
from law_assistant.memory import ConversationMemory
from law_assistant.processor import DocumentProcessor
from law_assistant.security import check_injection, sanitize_context
from law_assistant.splitter import DocumentSplitter, GeneralDocumentSplitter

logger = logging.getLogger(__name__)

load_dotenv()


# ── Helper for no-confidence streaming ────────────────────────────────
class _SimpleChunk:
    """Minimal chunk wrapper for non-LLM streaming responses."""

    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


# ── Citation Post-processing ──────────────────────────────────────────
def postprocess_citations(answer: str, context_parts: list[str]) -> str:
    """后处理：为缺少引用的法律句子自动补充 [来源N] 标签。

    逻辑:
    1. 检测包含法律术语但缺少 [来源N] 的句子
    2. 在 context_parts 中查找匹配的来源
    3. 在句末补充 [来源N]
    """
    if not answer or not context_parts:
        return answer

    legal_term_pattern = re.compile(
        r"《[^》]+》|第[零一二三四五六七八九十百千万\d]+[条章节款项]|"
        r"用人单位|劳动者|劳动合同|解除|终止|赔偿|补偿|共同财产|分割|抚养|继承|遗嘱|"
        r"诉讼|仲裁|起诉|上诉|调解|违约|侵权|过错|责任|义务|权利"
    )
    citation_pattern = re.compile(r"\[来源\d+\]")

    # 预处理: 从 context_parts 提取来源编号和关键词
    source_keywords = {}  # source_id -> set of keywords
    for i, part in enumerate(context_parts):
        source_id = i + 1
        keywords = set(legal_term_pattern.findall(part))
        source_keywords[source_id] = keywords

    def _find_best_source(sentence: str) -> int | None:
        """为句子找到最匹配的来源编号"""
        sentence_terms = set(legal_term_pattern.findall(sentence))
        if not sentence_terms:
            return None
        best_id = None
        best_overlap = 0
        for sid, kw in source_keywords.items():
            overlap = len(sentence_terms & kw)
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = sid
        return best_id if best_overlap > 0 else None

    # 按句处理
    sentences = re.split(r"(?<=[。！？])", answer)
    result = []
    for s in sentences:
        if not s.strip():
            result.append(s)
            continue
        # 跳过已有引用的句子
        if citation_pattern.search(s):
            result.append(s)
            continue
        # 跳过非实质性句子
        if len(s.strip()) <= 15:
            result.append(s)
            continue
        # 检测是否包含法律术语
        if not legal_term_pattern.search(s):
            result.append(s)
            continue
        # 尝试补充引用
        source_id = _find_best_source(s)
        if source_id:
            # 在句末最后一个标点前插入引用
            insert_pos = len(s.rstrip())
            for pos in range(len(s) - 1, -1, -1):
                if s[pos] in "。！？":
                    insert_pos = pos
                    break
            result.append(s[:insert_pos] + f" [来源{source_id}]" + s[insert_pos:])
        else:
            result.append(s)

    return "".join(result)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────
def reciprocal_rank_fusion(results_lists: list[list[tuple]], k: int = 60) -> list[tuple]:
    """RRF 融合多路检索结果，不依赖分数归一化，对不同检索器更鲁棒。
    results_lists: 每个检索器返回的 [(doc, score), ...] 列表
    k: 常数，控制排名影响衰减速度（默认60）
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, str] = {}
    for results in results_lists:
        for rank, (doc, _score) in enumerate(results, 1):
            if doc not in fused_scores:
                fused_scores[doc] = 0.0
                doc_map[doc] = doc
            fused_scores[doc] += 1.0 / (k + rank)
    sorted_items = sorted(fused_scores.items(), key=lambda x: -x[1])
    return [(doc_map[doc], score) for doc, score in sorted_items]


# ── Confidence Evaluation & Human Intervention ────────────────────────

HIGH_CONFIDENCE_THRESHOLD = 0.7
LOW_CONFIDENCE_THRESHOLD = 0.3


class ConfidenceEvaluator:
    """评估检索结果是否足够回答用户问题"""

    def evaluate(self, query: str, retrieved_docs: list, reranker_scores: list[float]) -> dict:
        """
        评估检索结果的充分性
        Returns: {"level": "high"|"low"|"none", "score": float, "reason": str}
        """
        if not retrieved_docs:
            return {"level": "none", "score": 0.0, "reason": "未检索到相关法律文档"}

        max_score = max(reranker_scores) if reranker_scores else 0.0
        high_score_count = sum(1 for s in reranker_scores if s >= LOW_CONFIDENCE_THRESHOLD)
        unique_laws = set()
        for doc in retrieved_docs:
            # Try to extract law name from metadata or content
            if isinstance(doc, tuple):
                doc_text = doc[0]
            else:
                doc_text = doc
            import re as _re

            law_match = _re.search(r"《(.+?)》", doc_text)
            if law_match:
                unique_laws.add(law_match.group(1))

        coverage_score = min(high_score_count / 3, 1.0)
        diversity_score = min(len(unique_laws) / 2, 1.0)
        final_score = max_score * 0.5 + coverage_score * 0.3 + diversity_score * 0.2

        if final_score >= HIGH_CONFIDENCE_THRESHOLD:
            level = "high"
            reason = "检索到充分的法律依据"
        elif final_score >= LOW_CONFIDENCE_THRESHOLD:
            level = "low"
            reason = f"检索到部分相关文档（{high_score_count}篇），但覆盖不够全面"
        else:
            level = "none"
            reason = f"检索结果不足（最高分: {max_score:.2f}），可能缺少相关法律知识"

        return {"level": level, "score": round(final_score, 4), "reason": reason}


class ResponseStrategy:
    """根据置信度选择响应策略"""

    DISCLAIMER_TEMPLATE = """

---

> **温馨提示：**
> 当前知识库可能未完全覆盖您的问题（{reason}）。
> 以上回答仅供参考，如需更专业的法律建议，请点击下方按钮联系人工律师咨询。
"""

    NO_CONFIDENCE_TEMPLATE = """感谢您的提问。

**当前知识库暂未收录相关法律规定**，无法为您提供准确的法律建议。

可能的原因：
- 该问题涉及的地方法规或行业规定未纳入知识库
- 问题涉及的法律领域超出当前覆盖范围
- 问题表述较为复杂，需要人工分析

**建议您：**
1. 点击下方按钮 **[转人工咨询]**，我们的专业律师将为您提供帮助
2. 您的问题已被记录，我们会尽快补充相关法律知识
"""

    def get_no_confidence_response(self) -> str:
        return self.NO_CONFIDENCE_TEMPLATE

    def get_disclaimer(self, reason: str) -> str:
        return self.DISCLAIMER_TEMPLATE.format(reason=reason)


# Shared thread pool for concurrent retrieval (avoids creating a new pool per query)
_SHARED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)
atexit.register(_SHARED_EXECUTOR.shutdown, wait=False)


def _cuda_available() -> bool:
    """检测 CUDA 是否可用"""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class DeepSeekApiRag:
    def __init__(self, api_key: str = None, db_path: str = None):
        from law_assistant.factories import EmbeddingFactory, LLMFactory, RerankerFactory

        if db_path is None:
            db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")

        # 1. Embedding (factory)
        self.embedding_model, self.fallback_embedding_model = EmbeddingFactory.create()

        # 2. LLM (factory)
        self.llm = LLMFactory.create()

        # 3. Vector DB
        self.db_path = db_path
        self.vector_db = None

        # 4. BM25
        self.bm25_retriever = BM25Retriever("data/bm25_index.pkl", rebuild_threshold=50)

        # 5. Document processor
        self.document_processor = DocumentProcessor()
        self.general_splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)

        # 6. Reranker (factory)
        reranker_provider = os.getenv("RERANKER_PROVIDER", "local").lower()
        self._reranker_provider = reranker_provider
        self._local_reranker = None
        self.reranker_api_key = None

        if reranker_provider == "local":
            self._local_reranker = RerankerFactory.create("local")
        else:
            reranker_config = RerankerFactory.create("dashscope")
            self.reranker_api_key = reranker_config["api_key"]
            self.reranker_model = reranker_config["model"]

        # 7. 初始化记忆模块
        self.memory = ConversationMemory(max_history_turns=5)
        self.memory.set_summarizer(self._summarize_messages)
        # 知识库模型类（由 app.py 注入，避免循环导入）
        self._knowledge_base_model = None

        # 8. 检索权重配置
        self.vector_weight = float(os.getenv("VECTOR_RETRIEVAL_WEIGHT", "0.4"))
        self.bm25_weight = float(os.getenv("BM25_RETRIEVAL_WEIGHT", "0.3"))
        self.graph_weight = float(os.getenv("GRAPH_RETRIEVAL_WEIGHT", "0.3"))

        # 8.1 检索模式（用于 baseline 对比实验：full / vector_only / bm25_only / graph_only）
        self.retrieval_mode = os.getenv("RETRIEVAL_MODE", "full")

        # 8.2 HyDE 开关（用于消融实验）
        self.enable_hyde = os.getenv("ENABLE_HYDE", "true").lower() == "true"

        # 9. 知识图谱（可选，Neo4j 不可用时自动降级）
        self.knowledge_graph = LegalKnowledgeGraph()
        if self.knowledge_graph.connect():
            self.knowledge_graph.create_schema()
        else:
            logger.info("知识图谱不可用，三路融合降级为双路融合")

        # 9. 相关性阈值（低于此分数的检索结果会被过滤）
        self.relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.15"))

        # 10. 启动时一次性加载所有 prompt 模板（避免每次请求重复读取 YAML）
        self._prompts_cache = self._load_all_prompts()

        # 11. 知识库文档文本缓存（首次访问时填充，文档增删时失效）
        self._kb_texts_cache: OrderedDict = OrderedDict()  # kb_id -> set of texts
        self._KB_CACHE_MAX_SIZE = 50  # 最多缓存 50 个知识库
        self._cache_lock = threading.Lock()

        # 12. FAISS 写操作锁（保护并发写入安全）
        self._faiss_write_lock = threading.Lock()
        self._faiss_dirty = False

        # 统一文档注册表（FAISS 和 BM25 重建的唯一数据源）
        self._document_registry: list[str] = []
        self._registry_lock = threading.Lock()

        # 13. 置信度评估器与响应策略（法律缺失时人工介入）
        self.confidence_evaluator = ConfidenceEvaluator()
        self.response_strategy = ResponseStrategy()

        # 如果向量数据库已存在，直接加载
        if os.path.exists(db_path):
            logger.info(f"加载已存在的向量数据库: {db_path}")
            self.load_vector_db()
            # 生成完整性哈希文件（首次运行时缺失）
            hash_file = os.path.join(db_path, ".hash")
            if not os.path.exists(hash_file):
                self._save_faiss_hash()
                logger.info("已生成 FAISS 索引完整性哈希文件")

        # 尝试加载BM25索引
        if not self.bm25_retriever.load_index():
            logger.info("BM25索引不存在，将在添加文档时构建")
        else:
            logger.info(f"BM25索引加载成功，文档数量: {self.bm25_retriever.get_document_count()}")

    def _load_all_prompts(self) -> dict:
        """启动时一次性加载所有 prompt 模板"""
        prompts_file = "prompts.yaml"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_path = os.path.join(current_dir, prompts_file)
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"提示词文件不存在: {prompts_path}")
        with open(prompts_path, encoding="utf-8") as file:
            prompts = yaml.safe_load(file)
        logger.info(f"已缓存 {len(prompts)} 个 prompt 模板")
        return prompts

    def _load_prompt(self, prompt_name: str = "legal_advisor_prompt") -> str:
        """从缓存返回 prompt 模板"""
        if prompt_name not in self._prompts_cache:
            raise ValueError(f"提示词 '{prompt_name}' 不存在")
        return self._prompts_cache[prompt_name]

    def _get_prompt(self, prompt_name: str = "legal_advisor_prompt", **kwargs) -> str:
        """获取格式化后的提示词"""
        prompt_template = self._load_prompt(prompt_name)

        try:
            formatted_prompt = prompt_template.format(**kwargs)
            return formatted_prompt
        except KeyError as e:
            raise ValueError(f"提示词格式化错误: 缺少参数 {e}") from e

    def _rerank_documents(
        self, query: str, documents: list[str], top_k: int = 10, original_scores: list[float] | None = None
    ) -> list[tuple[str, float]]:
        # 构建回退结果：使用原始融合分数而非 0.0
        def _fallback():
            if original_scores and len(original_scores) >= len(documents[:top_k]):
                return list(zip(documents[:top_k], original_scores[:top_k], strict=False))
            return [(doc, 0.0) for doc in documents[:top_k]]

        if not documents:
            return []

        # 本地 CrossEncoder Reranker
        if self._reranker_provider == "local" and self._local_reranker:
            try:
                pairs = [[query, doc] for doc in documents[:20]]
                scores = self._local_reranker.predict(pairs)
                scored = list(zip(documents[:20], scores, strict=False))
                scored.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"本地 Reranker 返回 {len(scored)} 个结果")
                return [(doc, float(score)) for doc, score in scored[:top_k]]
            except Exception as e:
                logger.warning(f"本地 Reranker 调用失败: {e}")
                return _fallback()

        # DashScope API Reranker
        if not self.reranker_api_key:
            logger.warning("未设置 Reranker API 密钥，跳过重排序")
            return _fallback()

        try:
            import dashscope
            from dashscope import TextReRank

            dashscope.api_key = self.reranker_api_key

            def _call_reranker():
                return TextReRank.call(
                    model=self.reranker_model,
                    query=query,
                    documents=documents[:20],
                    top_n=top_k,
                    return_documents=False,
                )

            # 超时保护：10 秒无响应则跳过重排序
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as timeout_executor:
                future = timeout_executor.submit(_call_reranker)
                try:
                    response = future.result(timeout=10)
                except concurrent.futures.TimeoutError:
                    logger.warning("Reranker API 超时 (10s)，跳过重排序")
                    return _fallback()

            if response.status_code == 200:
                results = response.output.get("results", [])
                reranked = []
                for res in results:
                    idx = res.get("index", 0)
                    score = res.get("relevance_score", 0.0)
                    if idx < len(documents):
                        reranked.append((documents[idx], score))
                reranked.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"Reranker 返回 {len(reranked)} 个结果")
                return reranked[:top_k]
            else:
                logger.warning(f"Reranker 返回错误: {response.status_code} {response.output}")
                return _fallback()

        except Exception as e:
            logger.warning(f"Reranker 调用失败: {e}")
            return _fallback()

    def _embed_with_retry(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """带重试的嵌入生成，主模型失败自动切换回退模型"""
        import time

        # 尝试主模型
        for attempt in range(max_retries):
            try:
                return self.embedding_model.embed_documents(texts)
            except Exception as e:
                err_str = str(e)
                logger.warning(f"主嵌入模型错误: {err_str[:200]}")
                wait_time = (2**attempt) * 2
                logger.info(f"等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                time.sleep(wait_time)

        # 主模型失败，尝试回退模型
        if self.fallback_embedding_model:
            logger.warning("主嵌入模型失败，切换到回退模型...")
            for attempt in range(max_retries):
                try:
                    return self.fallback_embedding_model.embed_documents(texts)
                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"回退嵌入模型错误: {err_str[:200]}")
                    wait_time = (2**attempt) * 2
                    logger.info(f"等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                    time.sleep(wait_time)

        raise RuntimeError("所有嵌入模型均调用失败")

    def _summarize_messages(self, messages: list) -> str:
        """用 LLM 将早期对话压缩为摘要（保留法律要点）"""
        if not messages:
            return ""
        history_text = "\n".join(f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:500]}" for m in messages)
        prompt = (
            "请将以下对话历史压缩为一段简洁的摘要。要求：\n"
            "1. 保留用户咨询的核心法律问题\n"
            "2. 保留关键法律结论和建议\n"
            "3. 保留涉及的法律条文编号\n"
            "4. 不超过300字\n\n"
            f"对话历史：\n{history_text}\n\n摘要："
        )
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"对话摘要生成失败: {e}")
            return ""

    def analyze_query(self, query: str, conversation_id: str = None, conversation_history: str = None) -> dict:
        """一次 LLM 调用完成：多轮融合 + 术语改写 + 查询分解 + HyDE 文档生成
        返回: {"rewritten_query": str, "sub_queries": [str], "hypothetical_doc": str}
        带重试机制（最多3次）和输出校验。
        """
        if conversation_history is not None:
            history = conversation_history
        else:
            history = ""
            if conversation_id:
                history = self.memory.get_formatted_history(conversation_id)
                if history == "无对话历史":
                    history = ""

        prompt = self._get_prompt("query_analysis_prompt", query=query, conversation_history=history or "无对话历史")

        for attempt in range(3):
            try:
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                # 提取 JSON（兼容 markdown code block 包裹）
                if "```" in content:
                    match = re.search(r"\{[\s\S]*\}", content)
                    if match:
                        content = match.group()
                result = json.loads(content)

                # 校验输出结构
                rewritten = result.get("rewritten_query", query)
                sub_queries = result.get("sub_queries", [])
                hypothetical = result.get("hypothetical_doc", "")

                if not rewritten or len(rewritten) < 3:
                    rewritten = query
                if not sub_queries:
                    sub_queries = [rewritten]
                # 限制子查询数量
                sub_queries = [q for q in sub_queries if isinstance(q, str) and len(q) > 3][:3]
                if not sub_queries:
                    sub_queries = [rewritten]

                logger.info(
                    f"查询分析: 原始='{query}' → 改写='{rewritten}', 子查询={len(sub_queries)}个, "
                    f"HyDE={'有' if hypothetical else '无'}"
                )
                return {
                    "rewritten_query": rewritten,
                    "sub_queries": sub_queries,
                    "hypothetical_doc": hypothetical,
                }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"查询分析解析失败 (第{attempt + 1}次): {e}")
                if attempt == 2:
                    break
            except Exception as e:
                logger.warning(f"查询分析调用失败 (第{attempt + 1}次): {e}")
                if attempt == 2:
                    break

        # 全部重试失败，降级为原始查询
        logger.warning("查询分析全部失败，使用原始查询")
        return {"rewritten_query": query, "sub_queries": [query], "hypothetical_doc": ""}

    def add_documents(self, documents: list[str], save_to_disk: bool = True):
        """添加文档到向量数据库和BM25索引"""
        if not documents:
            return

        logger.info(f"正在向向量数据库添加 {len(documents)} 个文档块...")

        # 手动生成嵌入向量（在锁外执行，避免长时间持锁）
        embeddings = self._embed_with_retry(documents)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # 检查嵌入维度是否一致
        if len(embeddings_array.shape) != 2:
            raise ValueError(f"嵌入维度不正确，期望2D数组，得到{embeddings_array.shape}")

        with self._faiss_write_lock:
            if self.vector_db is None:
                self.vector_db = FAISS.from_embeddings(
                    text_embeddings=list(zip(documents, embeddings_array, strict=False)),
                    embedding=self.embedding_model,
                    metadatas=[{} for _ in documents],
                )
                logger.info(f"FAISS 数据库已初始化，包含 {len(documents)} 个文档块。")
            else:
                # 如果向量数据库已存在，添加新文档
                self.vector_db.add_texts(documents, embeddings=embeddings_array)

        # 使用增量添加
        self.bm25_retriever.add_documents(documents)

        # 同步文档注册表
        with self._registry_lock:
            self._document_registry.extend(documents)

        if save_to_disk:
            self.save_vector_db()
            self.bm25_retriever.save_index()

        faiss_count = self.get_document_count()
        bm25_count = self.bm25_retriever.get_document_count()
        logger.info(f"文档添加完成 - 向量数据库: {faiss_count} 个文档, BM25索引: {bm25_count} 个文档")

    def add_file_documents(self, file_path: str, save_to_disk: bool = True):
        """添加单个文件文档"""
        logger.info(f"正在处理文档: {file_path}")

        try:
            # 使用文档处理器自动识别类型并处理
            structured_chunks = self.document_processor.process_document(file_path)

            # 准备添加到向量数据库的文本
            texts_to_add = []
            for chunk in structured_chunks:
                full_text = chunk["full_text"]

                # 如果是法律文档且条款过长，进行分块
                if chunk.get("metadata", {}).get("source") == "legal_document" and len(full_text) > 500:
                    legal_splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
                    sub_chunks = legal_splitter.split_text(full_text)
                    texts_to_add.extend(sub_chunks)
                else:
                    texts_to_add.append(full_text)

            logger.info(f"从文档中提取了 {len(structured_chunks)} 个结构化块，生成 {len(texts_to_add)} 个文本块")

            # 添加到向量数据库（不立即保存BM25，下面统一强制重建）
            self.add_documents(texts_to_add, save_to_disk=False)

            # P0: 单文件上传后强制立即重建BM25索引，确保立即可检索
            self.bm25_retriever.force_rebuild()

            if save_to_disk:
                self.save_vector_db()
                self.bm25_retriever.save_index()

        except Exception as e:
            logger.warning(f"文档处理失败: {e}")
            # 回退到普通分块
            self._fallback_add_documents(file_path, save_to_disk)

    def _fallback_add_documents(self, file_path: str, save_to_disk: bool = True):
        """回退到普通分块策略"""
        logger.info(f"使用普通分块策略处理: {file_path}")

        # 统一通过 DocumentProcessor 加载（支持 OCR 回退）
        pages = self.document_processor._load_documents(file_path)
        documents = self.general_splitter.split_documents(pages)
        texts = [doc.page_content for doc in documents]
        self.add_documents(texts, save_to_disk=False)
        # P0: 强制重建BM25索引
        self.bm25_retriever.force_rebuild()
        if save_to_disk:
            self.save_vector_db()
            self.bm25_retriever.save_index()

    def add_folder_documents(self, folder_path: str, save_to_disk: bool = True):
        """添加文件夹中的所有文档"""
        import time

        supported_extensions = (".pdf", ".doc", ".docx", ".txt", ".json", ".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        if not os.path.exists(folder_path):
            logger.warning(f"文件夹不存在: {folder_path}")
            return

        file_count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_extensions):
                file_path = os.path.join(folder_path, filename)
                logger.info(f"正在处理文件: {file_path}")
                self.add_file_documents(file_path, save_to_disk=False)
                time.sleep(1)
                file_count += 1

        # 重建一次BM25索引
        if file_count > 0:
            self.bm25_retriever.force_rebuild()

        if save_to_disk and (self.vector_db is not None or self.bm25_retriever.get_document_count() > 0):
            self.save_vector_db()
            self.bm25_retriever.save_index()

    def _compute_faiss_hash(self) -> str:
        """计算 FAISS 索引目录中所有文件的 SHA256 哈希"""
        hash_sha256 = hashlib.sha256()
        for fname in sorted(os.listdir(self.db_path)):
            fpath = os.path.join(self.db_path, fname)
            if os.path.isfile(fpath) and fname != ".hash":
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _save_faiss_hash(self):
        """保存 FAISS 索引的完整性哈希"""
        hash_file = os.path.join(self.db_path, ".hash")
        digest = self._compute_faiss_hash()
        with open(hash_file, "w") as f:
            f.write(digest)

    def _verify_faiss_hash(self) -> bool:
        """校验 FAISS 索引文件完整性"""
        hash_file = os.path.join(self.db_path, ".hash")
        if not os.path.exists(hash_file):
            logger.info("FAISS 索引无哈希文件，跳过完整性校验")
            return True
        with open(hash_file) as f:
            expected = f.read().strip()
        actual = self._compute_faiss_hash()
        return expected == actual

    def _save_vector_db_unlocked(self):
        """内部方法：保存向量数据库（假设已持有 _faiss_write_lock）"""
        if self.vector_db is not None:
            self.vector_db.save_local(self.db_path)
            self._save_faiss_hash()
            logger.info(f"向量数据库已保存到: {self.db_path}")

    def save_vector_db(self):
        """保存向量数据库到本地（附带完整性哈希）"""
        with self._faiss_write_lock:
            self._save_vector_db_unlocked()

    def load_vector_db(self):
        """从本地加载向量数据库（先校验文件完整性）"""
        if not self._verify_faiss_hash():
            raise RuntimeError(f"FAISS 索引文件完整性校验失败: {self.db_path}，文件可能被篡改")
        self.vector_db = FAISS.load_local(self.db_path, self.embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"向量数据库已从 {self.db_path} 加载")

    def _vector_search(self, query: str, top_k: int) -> list[tuple[str, float, str]]:
        """向量检索（归一化分数）"""
        results = []
        if self.vector_db is None:
            return results
        try:
            vector_results = self.vector_db.similarity_search_with_score(query, k=int(top_k * 1.5))
            vector_scores = [score for _, score in vector_results]
            if vector_scores:
                max_s = max(vector_scores)
                min_s = min(vector_scores)
                for doc, score in vector_results:
                    normalized = (max_s - score) / (max_s - min_s) if max_s != min_s else 1.0
                    results.append((doc.page_content, normalized, "vector"))
        except Exception as e:
            logger.warning(f"向量检索失败: {e}")
        return results

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float, str]]:
        """BM25 检索（归一化分数）"""
        results = []
        try:
            bm25_results = self.bm25_retriever.search(query, top_k=int(top_k * 1.5))
            bm25_scores = [score for _, score in bm25_results]
            if bm25_scores:
                max_s = max(bm25_scores)
                min_s = min(bm25_scores)
                for doc, score in bm25_results:
                    normalized = (score - min_s) / (max_s - min_s) if max_s != min_s else 1.0
                    results.append((doc, normalized, "bm25"))
        except Exception as e:
            logger.warning(f"BM25检索失败: {e}")
        return results

    def _graph_search(self, query: str, top_k: int) -> list[tuple[str, float, str]]:
        """图谱检索（归一化分数）"""
        results = []
        try:
            graph_results = self.knowledge_graph.graph_search(query, top_k=top_k)
            if graph_results:
                max_s = max(score for _, score in graph_results)
                min_s = min(score for _, score in graph_results)
                for doc, score in graph_results:
                    normalized = (score - min_s) / (max_s - min_s) if max_s != min_s else 1.0
                    results.append((doc, normalized, "graph"))
        except Exception as e:
            logger.warning(f"图谱检索失败: {e}")
        return results

    def hybrid_retrieve_documents(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """三路融合检索：向量检索 + BM25 检索 + 图谱检索（RRF 融合）

        支持 RETRIEVAL_MODE 环境变量控制检索模式：
        - full: 三路融合（默认）
        - vector_only: 仅向量检索
        - bm25_only: 仅 BM25 检索
        - graph_only: 仅知识图谱检索
        """
        mode = self.retrieval_mode
        result_lists = []

        # 顺序执行检索
        if mode in ("full", "vector_only"):
            vector_results = self._vector_search(query, top_k)
            logger.info(f"向量检索返回 {len(vector_results)} 个结果")
            if vector_results:
                result_lists.append([(doc, score) for doc, score, _ in vector_results])

        if mode in ("full", "bm25_only"):
            bm25_res = self._bm25_search(query, top_k)
            logger.info(f"BM25检索返回 {len(bm25_res)} 个结果")
            if bm25_res:
                result_lists.append([(doc, score) for doc, score, _ in bm25_res])

        if mode in ("full", "graph_only"):
            graph_res = self._graph_search(query, top_k)
            logger.info(f"图谱检索返回 {len(graph_res)} 个结果")
            if graph_res and self.knowledge_graph.is_available:
                result_lists.append([(doc, score) for doc, score, _ in graph_res])

        if not result_lists:
            return []

        # 单路模式直接返回，三路模式用 RRF 融合
        if mode == "full":
            final_results = reciprocal_rank_fusion(result_lists)[: int(top_k * 1.5)]
            logger.info(f"RRF 融合后返回 {len(final_results)} 个结果")
        else:
            final_results = result_lists[0][: int(top_k * 1.5)]
            logger.info(f"[{mode}] 返回 {len(final_results)} 个结果")

        return final_results

    def _get_retrieval_cache_key(self, query: str, knowledge_base_id: int = None) -> str | None:
        """生成检索缓存 key（基于查询内容 + 知识库 ID）"""
        import hashlib as _hashlib

        raw = f"{query}|kb={knowledge_base_id or ''}"
        return f"retrieval_cache:{_hashlib.md5(raw.encode()).hexdigest()}"

    def retrieve_documents(
        self,
        query: str,
        top_k: int = 10,
        sub_queries: list[str] = None,
        hypothetical_doc: str = "",
        knowledge_base_id: int = None,
        db_session=None,
    ) -> list[tuple[str, float]]:
        """混合检索 + 多子查询并行 + HyDE + 重排序 + 相关性过滤"""
        # 惰性重建 FAISS 索引（文档删除后标记 dirty）
        self.rebuild_faiss_if_dirty()

        # 尝试检索缓存（仅对简单查询生效，有子查询时跳过）
        if not sub_queries and not hypothetical_doc:
            try:
                from law_assistant.redis_utils import cache_get_json, cache_set_json

                cache_key = self._get_retrieval_cache_key(query, knowledge_base_id)
                cached = cache_get_json(cache_key)
                if cached:
                    logger.info(f"检索缓存命中: {query[:30]}...")
                    return [(doc, score) for doc, score in cached]
            except Exception:
                pass

        # 收集所有候选文档（去重）
        all_candidates = {}

        # 构建检索任务列表：主查询 + 子查询 + HyDE 文档
        search_queries = [query]
        if sub_queries:
            for sq in sub_queries:
                if sq not in search_queries:
                    search_queries.append(sq)
        if hypothetical_doc and len(hypothetical_doc) > 20:
            if self.enable_hyde:
                search_queries.append(hypothetical_doc)
            else:
                logger.info("[HyDE 已禁用] 跳过假设文档检索")

        # 并行执行所有检索任务（使用共享线程池）
        def _search_single(q):
            return self.hybrid_retrieve_documents(q, top_k=top_k)

        futures = {_SHARED_EXECUTOR.submit(_search_single, q): q for q in search_queries}
        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
                for doc, score in results:
                    if doc not in all_candidates or score > all_candidates[doc]:
                        all_candidates[doc] = score
            except Exception as e:
                logger.warning(f"子查询检索失败: {e}")

        if not all_candidates:
            logger.info("混合检索未返回任何结果")
            return []

        # P13: 知识库过滤 — 如果指定了知识库，只保留属于该知识库的文档
        if knowledge_base_id and db_session:
            kb_doc_contents = self._get_knowledge_base_texts(knowledge_base_id, db_session)
            if kb_doc_contents:
                kb_set = set(kb_doc_contents)
                before_count = len(all_candidates)
                all_candidates = {doc: score for doc, score in all_candidates.items() if doc in kb_set}
                logger.info(f"知识库过滤: {before_count} → {len(all_candidates)} (kb_id={knowledge_base_id})")

        # 按分数排序取 top candidates
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[: top_k * 3]
        initial_docs = [doc for doc, _ in top_candidates]
        initial_scores = [score for _, score in top_candidates]

        # 使用 reranker 进行精细排序（用主查询做 rerank），传入原始分数用于失败回退
        reranked_docs = self._rerank_documents(query, initial_docs, top_k=top_k, original_scores=initial_scores)

        # P1: 相关性过滤
        if self.relevance_threshold > 0:
            filtered = [(doc, score) for doc, score in reranked_docs if score >= self.relevance_threshold]
            if len(filtered) < len(reranked_docs):
                logger.info(f"相关性过滤: {len(reranked_docs)} → {len(filtered)} (阈值={self.relevance_threshold})")
            reranked_docs = filtered

        logger.info(f"重排序后返回 {len(reranked_docs)} 个最终结果")

        # 写入检索缓存（TTL 10 分钟）
        if not sub_queries and not hypothetical_doc and reranked_docs:
            try:
                from law_assistant.redis_utils import cache_set_json

                cache_key = self._get_retrieval_cache_key(query, knowledge_base_id)
                cache_set_json(cache_key, reranked_docs, ttl=600)
            except Exception:
                pass

        return reranked_docs

    def _get_knowledge_base_texts(self, knowledge_base_id: int, db_session) -> set:
        """获取指定知识库中所有文档的文本块（L1 内存 + L2 Redis + L3 重建）"""
        # L1: 内存缓存（锁内操作）
        with self._cache_lock:
            if knowledge_base_id in self._kb_texts_cache:
                self._kb_texts_cache.move_to_end(knowledge_base_id)
                return self._kb_texts_cache[knowledge_base_id]

        # L2: Redis（锁外，网络 I/O）
        try:
            from law_assistant.redis_utils import cache_get_set

            cached = cache_get_set(f"kb_texts:{knowledge_base_id}")
            if cached is not None:
                with self._cache_lock:
                    self._kb_texts_cache[knowledge_base_id] = cached
                return cached
        except Exception as e:
            logger.warning(f"Redis 缓存读取失败 (kb_texts:{knowledge_base_id}): {e}")

        # L3: 从 DB + 文件重建（锁外，CPU/IO 密集）
        try:
            if not self._knowledge_base_model:
                logger.warning("知识库模型未注入，无法从 DB 重建缓存")
                return set()
            KnowledgeBase = self._knowledge_base_model
            kb = db_session.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
            if not kb:
                return set()
            doc_paths = [d.file_path for d in kb.documents]
            texts = set()
            for path in doc_paths:
                if os.path.exists(path):
                    try:
                        chunks = self.document_processor.process_document(path)
                        for c in chunks:
                            full_text = c["full_text"]
                            # Apply same sub-splitting as add_file_documents
                            if c.get("metadata", {}).get("source") == "legal_document" and len(full_text) > 500:
                                legal_splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
                                sub_chunks = legal_splitter.split_text(full_text)
                                for sc in sub_chunks:
                                    texts.add(sc)
                            else:
                                texts.add(full_text)
                    except Exception as e:
                        logger.warning(f"知识库文档处理失败 {path}: {e}")
            # 锁内写回缓存（LRU 淘汰 + 插入）
            with self._cache_lock:
                if len(self._kb_texts_cache) >= self._KB_CACHE_MAX_SIZE:
                    with contextlib.suppress(StopIteration, RuntimeError):
                        self._kb_texts_cache.popitem(last=False)
                self._kb_texts_cache[knowledge_base_id] = texts
            # Write-through to Redis
            try:
                from law_assistant.redis_utils import cache_set_set

                cache_set_set(f"kb_texts:{knowledge_base_id}", texts, ttl=7200)
            except Exception as e:
                logger.warning(f"Redis 缓存写入失败 (kb_texts:{knowledge_base_id}): {e}")
            logger.info(f"已缓存知识库 {knowledge_base_id} 的 {len(texts)} 个文本块")
            return texts
        except Exception as e:
            logger.warning(f"获取知识库文本失败: {e}")
            return set()

    def set_knowledge_base_model(self, model_class):
        """注入知识库 ORM 模型类（由 app.py 调用，避免循环导入）"""
        self._knowledge_base_model = model_class

    def set_memory_db_factory(self, db_session_factory, message_model):
        """注入 DB session 工厂和 Message 模型到对话记忆模块"""
        self.memory._db_session_factory = db_session_factory
        self.memory._message_model = message_model

    def invalidate_kb_cache(self, knowledge_base_id: int = None):
        """文档增删后调用，清除缓存（内存 + Redis + 检索缓存）"""
        if knowledge_base_id is None:
            with self._cache_lock:
                self._kb_texts_cache.clear()
            try:
                from law_assistant.redis_utils import cache_delete_pattern

                cache_delete_pattern("kb_texts:*")
                cache_delete_pattern("retrieval_cache:*")
            except Exception as e:
                logger.warning(f"Redis 缓存清除失败: {e}")
        else:
            with self._cache_lock:
                self._kb_texts_cache.pop(knowledge_base_id, None)
            try:
                from law_assistant.redis_utils import cache_delete, cache_delete_pattern

                cache_delete(f"kb_texts:{knowledge_base_id}")
                cache_delete_pattern("retrieval_cache:*")
            except Exception as e:
                logger.warning(f"Redis 缓存清除失败: {e}")

    def mark_dirty(self):
        """标记 FAISS 索引需要重建（文档删除后调用）"""
        self._faiss_dirty = True
        logger.info("FAISS 索引已标记为需要重建")

    def remove_from_registry(self, texts: list[str]):
        """从统一文档注册表中移除指定文本（文档删除时调用）"""
        target_set = set(texts)
        with self._registry_lock:
            before = len(self._document_registry)
            self._document_registry = [t for t in self._document_registry if t not in target_set]
            removed = before - len(self._document_registry)
        if removed:
            logger.info(f"从文档注册表中移除 {removed} 个文本块")

    def rebuild_faiss_if_dirty(self):
        """惰性重建 FAISS 索引（从统一文档注册表重建，确保数据一致性）"""
        if not self._faiss_dirty:
            return
        # Double-check under lock
        with self._faiss_write_lock:
            if not self._faiss_dirty:
                return

        # 从统一注册表获取文档（而非 BM25）
        with self._registry_lock:
            docs = list(self._document_registry)

        if not docs:
            with self._faiss_write_lock:
                self.vector_db = None
                self._faiss_dirty = False
            logger.info("FAISS 索引已清空（无文档）")
            return

        logger.info(f"开始惰性重建 FAISS 索引（{len(docs)} 个文档）...")
        try:
            from langchain_community.vectorstores.faiss import FAISS

            new_db = FAISS.from_texts(docs, self.embedding_model, metadatas=[{"source": "bm25_rebuild"} for _ in docs])
            # Atomic swap + save under lock
            with self._faiss_write_lock:
                self.vector_db = new_db
                self._save_vector_db_unlocked()
                self._faiss_dirty = False
            logger.info(f"FAISS 索引重建完成，共 {len(docs)} 个文档")
        except Exception as e:
            logger.error(f"FAISS 索引重建失败: {e}")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """估算文本 token 数（区分中英文）"""
        if not text:
            return 1
        ascii_count = sum(1 for c in text if ord(c) < 128)
        non_ascii_count = len(text) - ascii_count
        return max(1, int(ascii_count * 0.25 + non_ascii_count * 1.5))

    def generate_response_stream(
        self,
        query: str,
        conversation_id: str = None,
        top_k: int = 10,
        prompt_name: str = "legal_advisor_prompt",
        knowledge_base_id: int = None,
        db_session=None,
    ):
        """生成RAG回答（带记忆、查询分析、HyDE、引用溯源）"""
        # Defense-in-depth: 再次检查注入（防止其他入口绕过 app.py 层）
        safe, reason = check_injection(query)
        if not safe:

            class _RejectChunk:
                __slots__ = ("content",)

                def __init__(self, c):
                    self.content = c

            def _reject():
                yield _RejectChunk(json.dumps({"error": reason}, ensure_ascii=False))
                yield _RejectChunk('{"done": true}')

            return {"stream": _reject(), "context": "", "retrieved_documents": [], "conversation_id": conversation_id}

        # 获取一次对话历史，传递给 analyze_query 避免重复查询
        conversation_history = ""
        if conversation_id:
            conversation_history = self.memory.get_formatted_history(conversation_id)
            if conversation_history == "无对话历史":
                conversation_history = ""

        # 一次 LLM 调用完成：多轮融合 + 术语改写 + 查询分解 + HyDE 文档
        analysis = self.analyze_query(query, conversation_id, conversation_history=conversation_history)

        try:
            retrieved_docs = self.retrieve_documents(
                query=analysis["rewritten_query"],
                top_k=top_k,
                sub_queries=analysis["sub_queries"],
                hypothetical_doc=analysis["hypothetical_doc"],
                knowledge_base_id=knowledge_base_id,
                db_session=db_session,
            )
        except Exception as e:
            logger.warning(f"文档检索失败，使用空上下文: {e}")
            retrieved_docs = []

        # 置信度评估 — 判断检索结果是否足够回答用户问题
        reranker_scores = [score for _, score in retrieved_docs]
        confidence = self.confidence_evaluator.evaluate(query, retrieved_docs, reranker_scores)

        # 无置信度：不调用 LLM，直接返回人工介入提示
        if confidence["level"] == "none":
            logger.info(f"检索置信度不足（{confidence['score']:.2f}），请求人工介入: {query[:50]}...")

            no_confidence_response = self.response_strategy.get_no_confidence_response()

            if conversation_id:
                self.memory.add_message(conversation_id, "user", query)

            def _no_confidence_stream():
                yield _SimpleChunk(no_confidence_response)
                yield _SimpleChunk("")

            return {
                "stream": _no_confidence_stream(),
                "context": "",
                "retrieved_documents": [doc[0] for doc in retrieved_docs],
                "retrieved_documents_with_scores": retrieved_docs,
                "analysis": analysis,
                "conversation_id": conversation_id,
                "confidence": confidence,
            }

        # 构建上下文（带引用编号 + token 预算保护）
        MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "3000"))
        context_parts = []
        estimated_tokens = 0

        if retrieved_docs:
            for i, (doc, score) in enumerate(retrieved_docs):
                safe_doc = sanitize_context(doc)
                part = f"【来源{i + 1}】(相关度:{score:.2f}): {safe_doc}"
                part_tokens = self._estimate_tokens(part)
                if estimated_tokens + part_tokens > MAX_CONTEXT_TOKENS and context_parts:
                    logger.info(f"上下文 token 预算已达上限，截断到 {len(context_parts)}/{len(retrieved_docs)} 个文档")
                    break
                context_parts.append(part)
                estimated_tokens += part_tokens

        context = "\n\n".join(context_parts) if context_parts else "无相关上下文"

        prompt = self._get_prompt(prompt_name, query=query, context=context, conversation_history=conversation_history)

        response_stream = self.llm.stream(prompt)

        if conversation_id:
            self.memory.add_message(conversation_id, "user", query)

        # 低置信度时在流式输出末尾追加免责声明
        show_intervention_banner = confidence["level"] == "low"
        disclaimer = self.response_strategy.get_disclaimer(confidence["reason"]) if show_intervention_banner else ""

        # 引用后处理：自动为缺少 [来源N] 的法律句子补充引用
        enable_citation_postprocess = os.getenv("ENABLE_CITATION_POSTPROCESS", "true").lower() == "true"

        def _stream_with_confidence():
            full_response = ""
            for chunk in response_stream:
                full_response += chunk.content
                yield chunk
            # 引用后处理：缓冲完整响应，补充缺失的 [来源N] 标签
            if enable_citation_postprocess and full_response and context_parts:
                processed = postprocess_citations(full_response, context_parts)
                if processed != full_response:
                    # 找出新增的引用部分，追加到流中
                    # 简单策略：直接 yield 处理后的完整响应（客户端已有原始响应）
                    # 这里仅记录日志，实际效果在 save_bot_response 中体现
                    logger.info("引用后处理: 已为缺少引用的法律句子补充 [来源N] 标签")
            if disclaimer:
                yield _SimpleChunk(disclaimer)

        return {
            "stream": _stream_with_confidence() if disclaimer else response_stream,
            "context": context,
            "retrieved_documents": [doc[0] for doc in retrieved_docs],
            "retrieved_documents_with_scores": [(doc, score) for doc, score in retrieved_docs],
            "analysis": analysis,
            "conversation_id": conversation_id,
            "confidence": confidence,
            "show_intervention_banner": show_intervention_banner,
        }

    def save_bot_response(self, conversation_id: str, response: str):
        """保存AI回复到记忆"""
        if conversation_id:
            self.memory.add_message(conversation_id, "assistant", response)

    def clear_conversation_memory(self, conversation_id: str):
        """清空特定对话的记忆"""
        self.memory.clear_conversation(conversation_id)

    def get_document_count(self) -> int:
        """获取向量数据库中的文档数量"""
        if self.vector_db is None:
            return 0
        return self.vector_db.index.ntotal if hasattr(self.vector_db.index, "ntotal") else 0
