import os
import pickle
from typing import List, Tuple

import jieba
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25检索器（优化版，支持批量添加）"""

    def __init__(self, bm25_index_path: str = "bm25_index.pkl", rebuild_threshold: int = 50):
        self.bm25_index_path = bm25_index_path
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.pending_documents = []  # 待添加文档缓存
        self.rebuild_threshold = rebuild_threshold  # 积累多少文档后重建索引

    def chinese_tokenize(self, text: str) -> List[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def build_index(self, documents: List[str]):
        """构建BM25索引（完整重建）"""
        if not documents:
            return

        self.documents = documents
        print(f"正在构建BM25索引，文档数量: {len(documents)}")

        # 分词处理
        self.tokenized_docs = [self.chinese_tokenize(doc) for doc in documents]

        # 构建BM25模型
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("BM25索引构建完成")

    def add_documents(self, new_documents: List[str], force_rebuild: bool = False):
        """添加文档（智能批量处理）"""
        if not new_documents:
            return

        # 添加到待处理队列
        self.pending_documents.extend(new_documents)
        print(f"已缓存 {len(new_documents)} 个文档，待处理文档总数: {len(self.pending_documents)}")

        # 判断是否需要重建索引
        should_rebuild = (force_rebuild or
                          len(self.pending_documents) >= self.rebuild_threshold or
                          self.bm25 is None)

        if should_rebuild:
            self._rebuild_with_pending()

    def _rebuild_with_pending(self):
        """使用待处理文档重建索引"""
        if not self.pending_documents and self.bm25 is not None:
            return

        # 合并所有文档
        all_documents = self.documents + self.pending_documents

        if not all_documents:
            return

        print(f"正在重建BM25索引，总文档数: {len(all_documents)}")

        # 重新构建索引
        self.build_index(all_documents)

        # 清空待处理队列
        pending_count = len(self.pending_documents)
        self.pending_documents = []

        print(f"索引重建完成，新增 {pending_count} 个文档")

    def force_rebuild(self):
        """强制立即重建索引"""
        self._rebuild_with_pending()

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25检索 — 不在搜索时强制重建索引（上传时已通过 force_rebuild 保证）"""
        if self.bm25 is None or not self.documents:
            # 仅在完全没有索引时才构建
            if self.pending_documents:
                self._rebuild_with_pending()
            else:
                return []

        # 查询分词
        tokenized_query = self.chinese_tokenize(query)

        # BM25评分
        scores = self.bm25.get_scores(tokenized_query)

        # 获取top_k结果
        doc_scores = list(zip(self.documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

    def save_index(self):
        """保存BM25索引（包含待处理文档）"""
        # 先确保所有文档都已索引
        if self.pending_documents:
            self._rebuild_with_pending()

        if self.bm25 is not None:
            with open(self.bm25_index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents,
                    'tokenized_docs': self.tokenized_docs
                }, f)
            print(f"BM25索引已保存到: {self.bm25_index_path}")

    def load_index(self) -> bool:
        """加载BM25索引"""
        if not os.path.exists(self.bm25_index_path):
            return False

        try:
            with open(self.bm25_index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['documents']
                self.tokenized_docs = data['tokenized_docs']
                self.pending_documents = []  # 加载时清空待处理队列
            print(f"BM25索引已从 {self.bm25_index_path} 加载，文档数: {len(self.documents)}")
            return True
        except Exception as e:
            print(f"加载BM25索引失败: {e}")
            return False

    def get_document_count(self) -> int:
        """获取总文档数量（包含待处理文档）"""
        return len(self.documents) + len(self.pending_documents)

    def remove_documents(self, target_texts: List[str]):
        """从索引中移除包含指定文本的文档（P1: 文档删除索引同步）"""
        if not self.documents:
            return

        # 先处理待处理文档
        if self.pending_documents:
            self.pending_documents = [d for d in self.pending_documents if d not in target_texts]

        # 从已索引文档中移除
        target_set = set(target_texts)
        new_documents = [d for d in self.documents if d not in target_set]
        removed_count = len(self.documents) - len(new_documents)

        if removed_count > 0:
            self.documents = new_documents
            # 重建索引
            if self.documents:
                self.tokenized_docs = [self.chinese_tokenize(doc) for doc in self.documents]
                self.bm25 = BM25Okapi(self.tokenized_docs)
            else:
                self.bm25 = None
                self.tokenized_docs = []
            print(f"BM25索引已移除 {removed_count} 个文档，剩余 {len(self.documents)} 个")
        else:
            print("BM25索引中未找到需要移除的文档")