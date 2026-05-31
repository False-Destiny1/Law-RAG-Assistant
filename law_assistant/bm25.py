import json
import logging
import os
import threading

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# 法律领域专业术语词典（词频, 词性）
_LEGAL_TERMS = {
    "善意取得": (9999, "n"),
    "违约金": (9999, "n"),
    "不可抗力": (9999, "n"),
    "劳动合同": (9999, "n"),
    "劳动关系": (9999, "n"),
    "劳动仲裁": (9999, "n"),
    "劳动争议": (9999, "n"),
    "经济补偿": (9999, "n"),
    "赔偿金": (9999, "n"),
    "解除合同": (9999, "v"),
    "终止合同": (9999, "v"),
    "合同无效": (9999, "n"),
    "合同解除": (9999, "n"),
    "合同终止": (9999, "n"),
    "合同违约": (9999, "n"),
    "连带责任": (9999, "n"),
    "补充责任": (9999, "n"),
    "过错责任": (9999, "n"),
    "无过错责任": (9999, "n"),
    "公平责任": (9999, "n"),
    "侵权责任": (9999, "n"),
    "侵权行为": (9999, "n"),
    "物权变动": (9999, "n"),
    "物权请求权": (9999, "n"),
    "抵押权": (9999, "n"),
    "质权": (9999, "n"),
    "留置权": (9999, "n"),
    "担保物权": (9999, "n"),
    "用益物权": (9999, "n"),
    "不当得利": (9999, "n"),
    "无因管理": (9999, "n"),
    "共同侵权": (9999, "n"),
    "共同危险行为": (9999, "n"),
    "安全保障义务": (9999, "n"),
    "注意义务": (9999, "n"),
    "合理使用": (9999, "n"),
    "法定许可": (9999, "n"),
    "知识产权": (9999, "n"),
    "专利权": (9999, "n"),
    "商标权": (9999, "n"),
    "著作权": (9999, "n"),
    "商业秘密": (9999, "n"),
    "法人": (9999, "n"),
    "非法人组织": (9999, "n"),
    "民事行为能力": (9999, "n"),
    "限制民事行为能力": (9999, "n"),
    "完全民事行为能力": (9999, "n"),
    "无民事行为能力": (9999, "n"),
    "诉讼时效": (9999, "n"),
    "除斥期间": (9999, "n"),
    "抗辩权": (9999, "n"),
    "形成权": (9999, "n"),
    "请求权": (9999, "n"),
    "代位权": (9999, "n"),
    "撤销权": (9999, "n"),
    "定金": (9999, "n"),
    "订金": (9999, "n"),
    "押金": (9999, "n"),
    "要约": (9999, "n"),
    "承诺": (9999, "n"),
    "要约邀请": (9999, "n"),
    "格式条款": (9999, "n"),
    "免责条款": (9999, "n"),
    "显失公平": (9999, "n"),
    "重大误解": (9999, "n"),
    "欺诈": (9999, "n"),
    "胁迫": (9999, "n"),
    "乘人之危": (9999, "n"),
    "善意第三人": (9999, "n"),
    "相对人": (9999, "n"),
    "用人单位": (9999, "n"),
    "劳动者": (9999, "n"),
    "工伤认定": (9999, "n"),
    "工伤保险": (9999, "n"),
    "竞业限制": (9999, "n"),
    "保密义务": (9999, "n"),
    "试用期": (9999, "n"),
    "加班费": (9999, "n"),
    "年休假": (9999, "n"),
    "社会保险": (9999, "n"),
    "住房公积金": (9999, "n"),
    "贪污罪": (9999, "n"),
    "受贿罪": (9999, "n"),
    "行贿罪": (9999, "n"),
    "盗窃罪": (9999, "n"),
    "诈骗罪": (9999, "n"),
    "抢劫罪": (9999, "n"),
    "故意伤害罪": (9999, "n"),
    "交通肇事罪": (9999, "n"),
    "正当防卫": (9999, "n"),
    "紧急避险": (9999, "n"),
    "犯罪未遂": (9999, "n"),
    "犯罪中止": (9999, "n"),
    "犯罪预备": (9999, "n"),
    "共同犯罪": (9999, "n"),
    "主犯": (9999, "n"),
    "从犯": (9999, "n"),
    "缓刑": (9999, "n"),
    "假释": (9999, "n"),
    "减刑": (9999, "n"),
    "自首": (9999, "n"),
    "立功": (9999, "n"),
    "行政许可": (9999, "n"),
    "行政处罚": (9999, "n"),
    "行政复议": (9999, "n"),
    "行政诉讼": (9999, "n"),
    "国家赔偿": (9999, "n"),
    "行政赔偿": (9999, "n"),
    "环境污染": (9999, "n"),
    "生态破坏": (9999, "n"),
    "个人信息": (9999, "n"),
    "隐私权": (9999, "n"),
    "网络侵权": (9999, "n"),
    "产品责任": (9999, "n"),
    "消费者权益": (9999, "n"),
    "食品安全": (9999, "n"),
}

for _word, (_freq, _pos) in _LEGAL_TERMS.items():
    jieba.add_word(_word, freq=_freq, tag=_pos)


class BM25Retriever:
    """BM25检索器（优化版，支持批量添加）"""

    def __init__(self, bm25_index_path: str = "bm25_index.pkl", rebuild_threshold: int = 50):
        self.bm25_index_path = bm25_index_path
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.pending_documents = []  # 待添加文档缓存
        self.rebuild_threshold = rebuild_threshold  # 积累多少文档后重建索引
        self._lock = threading.Lock()  # 保护 pending_documents 和索引状态

    def chinese_tokenize(self, text: str) -> list[str]:
        """中文分词"""
        return list(jieba.cut(text))

    def build_index(self, documents: list[str]):
        """构建BM25索引（完整重建，线程安全）"""
        if not documents:
            return

        logger.info(f"正在构建BM25索引，文档数量: {len(documents)}")

        # 分词处理（CPU 密集型，在锁外执行）
        tokenized_docs = [self.chinese_tokenize(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)

        # 原子性替换索引状态
        with self._lock:
            self.documents = documents
            self.tokenized_docs = tokenized_docs
            self.bm25 = bm25
        logger.info("BM25索引构建完成")

    def add_documents(self, new_documents: list[str], force_rebuild: bool = False):
        """添加文档（智能批量处理，线程安全）"""
        if not new_documents:
            return

        with self._lock:
            # 添加到待处理队列
            self.pending_documents.extend(new_documents)
            logger.info(f"已缓存 {len(new_documents)} 个文档，待处理文档总数: {len(self.pending_documents)}")

            # 判断是否需要重建索引
            should_rebuild = force_rebuild or len(self.pending_documents) >= self.rebuild_threshold or self.bm25 is None

        if should_rebuild:
            self._rebuild_with_pending()

    def _rebuild_with_pending(self):
        """使用待处理文档重建索引（需在 self._lock 外调用，内部自行加锁）"""
        with self._lock:
            if not self.pending_documents and self.bm25 is not None:
                return

            # 合并所有文档
            all_documents = self.documents + self.pending_documents

            if not all_documents:
                return

            # 清空待处理队列（先清空再构建，构建可能耗时）
            pending_count = len(self.pending_documents)
            self.pending_documents = []

        logger.info(f"正在重建BM25索引，总文档数: {len(all_documents)}")

        # 重新构建索引（CPU 密集型，在锁外执行）
        self.build_index(all_documents)

        logger.info(f"索引重建完成，新增 {pending_count} 个文档")

    def force_rebuild(self):
        """强制立即重建索引"""
        self._rebuild_with_pending()

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """BM25检索 — 线程安全"""
        with self._lock:
            if self.bm25 is None or not self.documents:
                # 仅在完全没有索引时才构建
                if self.pending_documents:
                    pass  # will rebuild below
                else:
                    return []

        if self.bm25 is None and self.pending_documents:
            self._rebuild_with_pending()

        # 查询分词
        tokenized_query = self.chinese_tokenize(query)

        # BM25评分（在锁内读取共享状态的一致快照）
        with self._lock:
            if self.bm25 is None:
                return []
            scores = self.bm25.get_scores(tokenized_query)
            documents = self.documents

        # 获取top_k结果
        doc_scores = list(zip(documents, scores, strict=False))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

    def save_index(self):
        """保存BM25索引（JSON格式，不存储pickle对象）"""
        if self.pending_documents:
            self._rebuild_with_pending()

        if self.bm25 is not None:
            with open(self.bm25_index_path, "w", encoding="utf-8") as f:
                json.dump({"documents": self.documents, "tokenized_docs": self.tokenized_docs}, f, ensure_ascii=False)
            logger.info(f"BM25索引已保存到: {self.bm25_index_path}")

    def load_index(self) -> bool:
        """加载BM25索引（从JSON重建BM25Okapi）"""
        if not os.path.exists(self.bm25_index_path):
            return False

        try:
            with open(self.bm25_index_path, encoding="utf-8") as f:
                data = json.load(f)
            self.documents = data["documents"]
            self.tokenized_docs = data["tokenized_docs"]
            self.bm25 = BM25Okapi(self.tokenized_docs) if self.tokenized_docs else None
            self.pending_documents = []
            logger.info(f"BM25索引已从 {self.bm25_index_path} 加载，文档数: {len(self.documents)}")
            return True
        except Exception as e:
            logger.warning(f"加载BM25索引失败: {e}")
            return False

    def get_document_count(self) -> int:
        """获取总文档数量（包含待处理文档）"""
        return len(self.documents) + len(self.pending_documents)

    def remove_documents(self, target_texts: list[str]):
        """从索引中移除包含指定文本的文档（线程安全）"""
        target_set = set(target_texts)

        # Phase 1: 锁内快速过滤（不 tokenize）
        with self._lock:
            if not self.documents:
                return

            if self.pending_documents:
                self.pending_documents = [d for d in self.pending_documents if d not in target_set]

            new_documents = [d for d in self.documents if d not in target_set]
            removed_count = len(self.documents) - len(new_documents)

            if removed_count == 0:
                logger.info("BM25索引中未找到需要移除的文档")
                return

            self.documents = new_documents

        # Phase 2: 锁外做 CPU 密集的 tokenize + 索引构建
        if new_documents:
            tokenized_docs = [self.chinese_tokenize(doc) for doc in new_documents]
            bm25 = BM25Okapi(tokenized_docs)
        else:
            tokenized_docs = []
            bm25 = None

        # Phase 3: 原子替换索引
        with self._lock:
            self.tokenized_docs = tokenized_docs
            self.bm25 = bm25

        logger.info(f"BM25索引已移除 {removed_count} 个文档，剩余 {len(new_documents)} 个")
