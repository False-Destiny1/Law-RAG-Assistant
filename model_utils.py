from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import os
import yaml
import json
import numpy as np
import concurrent.futures
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from DocumentSplitter import DocumentSplitter, GeneralDocumentSplitter
from BM25Retriever import BM25Retriever
from ConversationMemory import ConversationMemory
from DocumentProcessor import DocumentProcessor

load_dotenv()

# Shared thread pool for concurrent retrieval (avoids creating a new pool per query)
_SHARED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)


class DeepSeekApiRag:
    def __init__(self, api_key: str = None, db_path: str = None):
        # 从环境变量获取配置，如果参数为None则使用环境变量
        if db_path is None:
            db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")

        # 1. 初始化嵌入模型
        print("正在加载嵌入模型...")
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "dashscope").lower()
        if embedding_provider == "dashscope":
            embedding_api_key = os.getenv("EMBEDDING_API_KEY")
            embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-v2")
            fallback_model_name = os.getenv("EMBEDDING_FALLBACK_MODEL", "text-embedding-async-v2")
            print(f"使用 DashScope 嵌入模型: {embedding_model_name} (回退: {fallback_model_name})")
            self.embedding_model = OpenAIEmbeddings(
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                model=embedding_model_name,
                check_embedding_ctx_length=False,
                chunk_size=10,
            )
            self.fallback_embedding_model = OpenAIEmbeddings(
                api_key=embedding_api_key,
                base_url=embedding_base_url,
                model=fallback_model_name,
                check_embedding_ctx_length=False,
                chunk_size=10,
            )
        else:
            embedding_model_name = os.getenv("EMBEDDING_MODEL", "bge-small-zh-v1.5")
            # 支持本地路径或HuggingFace模型名
            model_path = embedding_model_name
            if not os.path.isabs(model_path) and os.path.isdir(embedding_model_name):
                model_path = os.path.abspath(embedding_model_name)
            elif not os.path.isabs(model_path) and not os.path.isdir(model_path):
                # 尝试项目目录下的本地路径
                local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), embedding_model_name)
                if os.path.isdir(local_path):
                    model_path = local_path
            print(f"使用本地嵌入模型: {model_path}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_path,
                model_kwargs={'device': 'cuda'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.fallback_embedding_model = None

        # 2. 初始化 LLM API（优先 MiMo，回退 DeepSeek）
        mimo_api_key = os.getenv("MIMO_API_KEY")
        if mimo_api_key:
            mimo_base_url = os.getenv("MIMO_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
            mimo_model = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")
            print(f"正在初始化 MiMo API (模型: {mimo_model})...")
            self.llm = ChatOpenAI(
                api_key=mimo_api_key,
                base_url=mimo_base_url,
                model=mimo_model,
            )
        else:
            deepseek_api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            print(f"正在初始化 DeepSeek API (模型: {deepseek_model})...")
            self.llm = ChatOpenAI(
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                model=deepseek_model,
            )

        # 3. 初始化向量数据库
        self.db_path = db_path
        self.vector_db = None

        # 4. 初始化BM25检索器
        self.bm25_retriever = BM25Retriever("bm25_index.pkl", rebuild_threshold=50)

        # 5. 初始化文档处理器
        self.document_processor = DocumentProcessor()
        self.general_splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)

        # 6. 初始化Reranker配置
        self.reranker_api_key = os.getenv("RERANKER_API_KEY")
        self.reranker_url = os.getenv("RERANKER_BASE_URL", "https://api.siliconflow.cn/v1/rerank")
        self.reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

        # 7. 初始化记忆模块
        self.memory = ConversationMemory(max_history_turns=5)

        # 8. 检索权重配置
        self.vector_weight = float(os.getenv("VECTOR_RETRIEVAL_WEIGHT", "0.6"))
        self.bm25_weight = float(os.getenv("BM25_RETRIEVAL_WEIGHT", "0.4"))

        # 9. 相关性阈值（低于此分数的检索结果会被过滤）
        self.relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.15"))

        # 10. 启动时一次性加载所有 prompt 模板（避免每次请求重复读取 YAML）
        self._prompts_cache = self._load_all_prompts()

        # 11. 知识库文档文本缓存（首次访问时填充，文档增删时失效）
        self._kb_texts_cache: dict = {}  # kb_id -> set of texts

        # 如果向量数据库已存在，直接加载
        if os.path.exists(db_path):
            print(f"加载已存在的向量数据库: {db_path}")
            self.load_vector_db()

        # 尝试加载BM25索引
        # 尝试加载BM25索引
        if not self.bm25_retriever.load_index():
            print("BM25索引不存在，将在添加文档时构建")
        else:
            print(f"BM25索引加载成功，文档数量: {self.bm25_retriever.get_document_count()}")

    def _load_all_prompts(self) -> dict:
        """启动时一次性加载所有 prompt 模板"""
        prompts_file = "prompts.yaml"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_path = os.path.join(current_dir, prompts_file)
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(f"提示词文件不存在: {prompts_path}")
        with open(prompts_path, 'r', encoding='utf-8') as file:
            prompts = yaml.safe_load(file)
        print(f"已缓存 {len(prompts)} 个 prompt 模板")
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
            raise ValueError(f"提示词格式化错误: 缺少参数 {e}")

    def _rerank_documents(
            self,
            query: str,
            documents: List[str],
            top_k: int = 10,
            original_scores: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        # 构建回退结果：使用原始融合分数而非 0.0
        def _fallback():
            if original_scores and len(original_scores) >= len(documents[:top_k]):
                return list(zip(documents[:top_k], original_scores[:top_k]))
            return [(doc, 0.0) for doc in documents[:top_k]]

        if not self.reranker_api_key:
            print("未设置 Reranker API 密钥，跳过重排序")
            return _fallback()

        try:
            from dashscope import TextReRank
            import dashscope
            dashscope.api_key = self.reranker_api_key

            def _call_reranker():
                return TextReRank.call(
                    model=self.reranker_model,
                    query=query,
                    documents=documents[:20],
                    top_n=top_k,
                    return_documents=False
                )

            # 超时保护：10 秒无响应则跳过重排序
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as timeout_executor:
                future = timeout_executor.submit(_call_reranker)
                try:
                    response = future.result(timeout=10)
                except concurrent.futures.TimeoutError:
                    print("Reranker API 超时 (10s)，跳过重排序")
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
                print(f"Reranker 返回 {len(reranked)} 个结果")
                return reranked[:top_k]
            else:
                print(f"Reranker 返回错误: {response.status_code} {response.output}")
                return _fallback()

        except Exception as e:
            print(f"Reranker 调用失败: {e}")
            return _fallback()

    def _embed_with_retry(self, texts: List[str], max_retries: int = 3) -> List[List[float]]:
        """带重试的嵌入生成，主模型失败自动切换回退模型"""
        import time
        # 尝试主模型
        for attempt in range(max_retries):
            try:
                return self.embedding_model.embed_documents(texts)
            except Exception as e:
                err_str = str(e)
                print(f"主嵌入模型错误: {err_str[:200]}")
                wait_time = (2 ** attempt) * 2
                print(f"等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                time.sleep(wait_time)

        # 主模型失败，尝试回退模型
        if self.fallback_embedding_model:
            print("主嵌入模型失败，切换到回退模型...")
            for attempt in range(max_retries):
                try:
                    return self.fallback_embedding_model.embed_documents(texts)
                except Exception as e:
                    err_str = str(e)
                    print(f"回退嵌入模型错误: {err_str[:200]}")
                    wait_time = (2 ** attempt) * 2
                    print(f"等待 {wait_time} 秒后重试 (第 {attempt + 1}/{max_retries} 次)...")
                    time.sleep(wait_time)

        raise RuntimeError("所有嵌入模型均调用失败")

    def analyze_query(self, query: str, conversation_id: str = None, conversation_history: str = None) -> dict:
        """一次 LLM 调用完成：多轮融合 + 术语改写 + 查询分解 + HyDE 文档生成
        返回: {"rewritten_query": str, "sub_queries": [str], "hypothetical_doc": str}
        """
        if conversation_history is not None:
            history = conversation_history
        else:
            history = ""
            if conversation_id:
                history = self.memory.get_formatted_history(conversation_id)
                if history == "无对话历史":
                    history = ""

        try:
            prompt = self._get_prompt(
                "query_analysis_prompt",
                query=query,
                conversation_history=history or "无对话历史"
            )
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            # 提取 JSON（兼容 markdown code block 包裹）
            if "```" in content:
                import re
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    content = match.group()
            result = json.loads(content)
            rewritten = result.get("rewritten_query", query)
            sub_queries = result.get("sub_queries", [rewritten])
            hypothetical = result.get("hypothetical_doc", "")
            print(f"查询分析: 原始='{query}' → 改写='{rewritten}', 子查询={len(sub_queries)}个, HyDE={'有' if hypothetical else '无'}")
            return {
                "rewritten_query": rewritten if len(rewritten) > 3 else query,
                "sub_queries": [q for q in sub_queries if len(q) > 3] or [rewritten],
                "hypothetical_doc": hypothetical
            }
        except Exception as e:
            print(f"查询分析失败，使用原始查询: {e}")
            return {"rewritten_query": query, "sub_queries": [query], "hypothetical_doc": ""}

    def add_documents(self, documents: List[str], save_to_disk: bool = True):
        """添加文档到向量数据库和BM25索引"""
        if not documents:
            return

        print(f"正在向向量数据库添加 {len(documents)} 个文档块...")

        # 手动生成嵌入向量并确保是numpy数组格式
        embeddings = self._embed_with_retry(documents)
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # 检查嵌入维度是否一致
        if len(embeddings_array.shape) != 2:
            raise ValueError(f"嵌入维度不正确，期望2D数组，得到{embeddings_array.shape}")

        if self.vector_db is None:
            # 使用FAISS.from_embeddings方法
            docs = [Document(page_content=text) for text in documents]

            self.vector_db = FAISS.from_embeddings(
                text_embeddings=list(zip(documents, embeddings_array)),
                embedding=self.embedding_model,
                metadatas=[{} for _ in documents]
            )
            print(f"FAISS 数据库已初始化，包含 {len(documents)} 个文档块。")
        else:
            # 如果向量数据库已存在，添加新文档
            self.vector_db.add_texts(documents, embeddings=embeddings_array)

        # 使用增量添加
        self.bm25_retriever.add_documents(documents)

        if save_to_disk:
            self.save_vector_db()
            self.bm25_retriever.save_index()

        print(
            f"文档添加完成 - 向量数据库: {self.get_document_count()} 个文档, BM25索引: {self.bm25_retriever.get_document_count()} 个文档")
    def add_file_documents(self, file_path: str, save_to_disk: bool = True):
        """添加单个文件文档"""
        print(f"正在处理文档: {file_path}")

        try:
            # 使用文档处理器自动识别类型并处理
            structured_chunks = self.document_processor.process_document(file_path)

            # 准备添加到向量数据库的文本
            texts_to_add = []
            for chunk in structured_chunks:
                full_text = chunk['full_text']

                # 如果是法律文档且条款过长，进行分块
                if chunk.get('metadata', {}).get('source') == 'legal_document' and len(full_text) > 500:
                    legal_splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
                    sub_chunks = legal_splitter.split_text(full_text)
                    texts_to_add.extend(sub_chunks)
                else:
                    texts_to_add.append(full_text)

            print(f"从文档中提取了 {len(structured_chunks)} 个结构化块，生成 {len(texts_to_add)} 个文本块")

            # 添加到向量数据库（不立即保存BM25，下面统一强制重建）
            self.add_documents(texts_to_add, save_to_disk=False)

            # P0: 单文件上传后强制立即重建BM25索引，确保立即可检索
            self.bm25_retriever.force_rebuild()

            if save_to_disk:
                self.save_vector_db()
                self.bm25_retriever.save_index()

        except Exception as e:
            print(f"文档处理失败: {e}")
            # 回退到普通分块
            self._fallback_add_documents(file_path, save_to_disk)

    def _fallback_add_documents(self, file_path: str, save_to_disk: bool = True):
        """回退到普通分块策略"""
        print(f"使用普通分块策略处理: {file_path}")

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
        supported_extensions = ('.pdf', '.doc', '.docx', '.txt', '.jpg', '.jpeg', '.png', '.bmp', '.tiff')

        if not os.path.exists(folder_path):
            print(f"文件夹不存在: {folder_path}")
            return

        file_count = 0
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_extensions):
                file_path = os.path.join(folder_path, filename)
                print(f"正在处理文件: {file_path}")
                self.add_file_documents(file_path, save_to_disk=False)
                time.sleep(1)
                file_count += 1

        # 重建一次BM25索引
        if file_count > 0:
            self.bm25_retriever.force_rebuild()

        if save_to_disk and (self.vector_db is not None or self.bm25_retriever.get_document_count() > 0):
            self.save_vector_db()
            self.bm25_retriever.save_index()

    def save_vector_db(self):
        """保存向量数据库到本地"""
        if self.vector_db is not None:
            self.vector_db.save_local(self.db_path)
            print(f"向量数据库已保存到: {self.db_path}")

    def load_vector_db(self):
        """从本地加载向量数据库"""
        self.vector_db = FAISS.load_local(
            self.db_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )
        print(f"向量数据库已从 {self.db_path} 加载")

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[str, float, str]]:
        """向量检索（归一化分数）"""
        results = []
        if self.vector_db is None:
            return results
        try:
            vector_results = self.vector_db.similarity_search_with_score(query, k=top_k * 2)
            vector_scores = [score for _, score in vector_results]
            if vector_scores:
                max_s = max(vector_scores)
                min_s = min(vector_scores)
                for doc, score in vector_results:
                    normalized = (max_s - score) / (max_s - min_s) if max_s != min_s else 1.0
                    results.append((doc.page_content, normalized, "vector"))
        except Exception as e:
            print(f"向量检索失败: {e}")
        return results

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float, str]]:
        """BM25 检索（归一化分数）"""
        results = []
        try:
            bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)
            bm25_scores = [score for _, score in bm25_results]
            if bm25_scores:
                max_s = max(bm25_scores)
                min_s = min(bm25_scores)
                for doc, score in bm25_results:
                    normalized = (score - min_s) / (max_s - min_s) if max_s != min_s else 1.0
                    results.append((doc, normalized, "bm25"))
        except Exception as e:
            print(f"BM25检索失败: {e}")
        return results

    def hybrid_retrieve_documents(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """混合检索：向量检索 + BM25 检索（并行执行）"""
        all_results = []

        # 向量检索和 BM25 检索并行执行（使用共享线程池）
        vector_future = _SHARED_EXECUTOR.submit(self._vector_search, query, top_k)
        bm25_future = _SHARED_EXECUTOR.submit(self._bm25_search, query, top_k)
        all_results.extend(vector_future.result())
        bm25_res = bm25_future.result()
        all_results.extend(bm25_res)
        print(f"向量检索返回 {sum(1 for _,_,m in all_results if m=='vector')} 个结果")
        print(f"BM25检索返回 {len(bm25_res)} 个结果")

        # 3. 结果融合（加权求和）
        fused_results = {}
        for doc, score, method in all_results:
            weight = self.vector_weight if method == "vector" else self.bm25_weight
            if doc not in fused_results:
                fused_results[doc] = score * weight
            else:
                # 同一文档被两种方法命中时累加权重（双方法命中的文档应得分更高）
                fused_results[doc] += score * weight

        # 4. 排序并返回top_k
        sorted_results = sorted(fused_results.items(), key=lambda x: x[1], reverse=True)

        final_results = [(doc, score) for doc, score in sorted_results[:top_k * 2]]
        print(f"混合检索融合后返回 {len(final_results)} 个结果")

        return final_results

    def retrieve_documents(self, query: str, top_k: int = 10,
                           sub_queries: List[str] = None,
                           hypothetical_doc: str = "",
                           knowledge_base_id: int = None,
                           db_session=None) -> List[Tuple[str, float]]:
        """混合检索 + 多子查询并行 + HyDE + 重排序 + 相关性过滤"""
        # 收集所有候选文档（去重）
        all_candidates = {}

        # 构建检索任务列表：主查询 + 子查询 + HyDE 文档
        search_queries = [query]
        if sub_queries:
            for sq in sub_queries:
                if sq not in search_queries:
                    search_queries.append(sq)
        if hypothetical_doc and len(hypothetical_doc) > 20:
            search_queries.append(hypothetical_doc)

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
                print(f"子查询检索失败: {e}")

        if not all_candidates:
            print("混合检索未返回任何结果")
            return []

        # P13: 知识库过滤 — 如果指定了知识库，只保留属于该知识库的文档
        if knowledge_base_id and db_session:
            kb_doc_contents = self._get_knowledge_base_texts(knowledge_base_id, db_session)
            if kb_doc_contents:
                kb_set = set(kb_doc_contents)
                before_count = len(all_candidates)
                all_candidates = {doc: score for doc, score in all_candidates.items() if doc in kb_set}
                print(f"知识库过滤: {before_count} → {len(all_candidates)} (kb_id={knowledge_base_id})")

        # 按分数排序取 top candidates
        sorted_candidates = sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:top_k * 3]
        initial_docs = [doc for doc, _ in top_candidates]
        initial_scores = [score for _, score in top_candidates]

        # 使用 reranker 进行精细排序（用主查询做 rerank），传入原始分数用于失败回退
        reranked_docs = self._rerank_documents(query, initial_docs, top_k=top_k, original_scores=initial_scores)

        # P1: 相关性过滤
        if self.relevance_threshold > 0:
            filtered = [(doc, score) for doc, score in reranked_docs if score >= self.relevance_threshold]
            if len(filtered) < len(reranked_docs):
                print(f"相关性过滤: {len(reranked_docs)} → {len(filtered)} (阈值={self.relevance_threshold})")
            reranked_docs = filtered

        print(f"重排序后返回 {len(reranked_docs)} 个最终结果")
        return reranked_docs

    def _get_knowledge_base_texts(self, knowledge_base_id: int, db_session) -> set:
        """获取指定知识库中所有文档的文本块（带缓存）"""
        if knowledge_base_id in self._kb_texts_cache:
            return self._kb_texts_cache[knowledge_base_id]
        try:
            from app import KnowledgeBase
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
                            texts.add(c['full_text'])
                    except Exception as e:
                        print(f"知识库文档处理失败 {path}: {e}")
            self._kb_texts_cache[knowledge_base_id] = texts
            print(f"已缓存知识库 {knowledge_base_id} 的 {len(texts)} 个文本块")
            return texts
        except Exception as e:
            print(f"获取知识库文本失败: {e}")
            return set()

    def invalidate_kb_cache(self, knowledge_base_id: int = None):
        """文档增删后调用，清除缓存"""
        if knowledge_base_id is None:
            self._kb_texts_cache.clear()
        elif knowledge_base_id in self._kb_texts_cache:
            del self._kb_texts_cache[knowledge_base_id]

    def generate_response_stream(self, query: str, conversation_id: str = None, top_k: int = 20,
                                 prompt_name: str = "legal_advisor_prompt",
                                 knowledge_base_id: int = None, db_session=None):
        """生成RAG回答（带记忆、查询分析、HyDE、引用溯源）"""
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
                db_session=db_session
            )
        except Exception as e:
            print(f"文档检索失败，使用空上下文: {e}")
            retrieved_docs = []

        # 构建上下文（带引用编号）
        context_parts = []

        if retrieved_docs:
            for i, (doc, score) in enumerate(retrieved_docs):
                context_parts.append(f"【来源{i + 1}】(相关度:{score:.2f}): {doc}")

        if conversation_history:
            context_parts.append(conversation_history)

        context = "\n\n".join(context_parts) if context_parts else "无相关上下文"

        prompt = self._get_prompt(
            prompt_name,
            query=query,
            context=context,
            conversation_history=conversation_history
        )

        response_stream = self.llm.stream(prompt)

        if conversation_id:
            self.memory.add_message(conversation_id, 'user', query)

        return {
            "stream": response_stream,
            "context": context,
            "retrieved_documents": [doc[0] for doc in retrieved_docs],
            "conversation_id": conversation_id
        }

    def save_bot_response(self, conversation_id: str, response: str):
        """保存AI回复到记忆"""
        if conversation_id:
            self.memory.add_message(conversation_id, 'assistant', response)

    def clear_conversation_memory(self, conversation_id: str):
        """清空特定对话的记忆"""
        self.memory.clear_conversation(conversation_id)

    def get_document_count(self) -> int:
        """获取向量数据库中的文档数量"""
        if self.vector_db is None:
            return 0
        return self.vector_db.index.ntotal if hasattr(self.vector_db.index, 'ntotal') else 0

    def get_bm25_document_count(self) -> int:
        """获取BM25索引中的文档数量"""
        return self.bm25_retriever.get_document_count()