"""法律领域知识图谱 — 规则抽取 + Neo4j 存储 + 图谱检索

三路融合检索的图谱分支：
- 从法律文本中抽取 条文/章节/引用/概念 实体和关系
- 在 Neo4j 中构建知识图谱
- 通过实体链接 + 1-2 跳子图遍历实现图谱检索
"""

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# ─── 正则表达式 ──────────────────────────────────────────────

# 条文：第一百四十三条、第143条、第一百四十三条之一
ARTICLE_PATTERN = re.compile(
    r"第([零一二三四五六七八九十百千万\d]+)条"
    r"(?:之([零一二三四五六七八九十百千万\d]+))?"
)

# 章节：第一章 总则、第二编 物权
CHAPTER_PATTERN = re.compile(r"(第[零一二三四五六七八九十百千\d]+)([编篇章节])\s*(.*)")

# 引用模式
CITE_PATTERNS = [
    # 自引用：依照本法第五百八十四条
    (re.compile(r"依照本法第([零一二三四五六七八九十百千万\d]+)条"), "self"),
    # 自引用简写：本法第X条
    (re.compile(r"本法第([零一二三四五六七八九十百千万\d]+)条"), "self"),
    # 跨法律引用：依据《民法典》第一百四十三条
    (re.compile(r"《([^》]+)》第([零一二三四五六七八九十百千万\d]+)条"), "cross"),
    # 条款引用：第X条第Y款
    (re.compile(r"第([零一二三四五六七八九十百千万\d]+)条第([零一二三四五六七八九十百千万\d]+)款"), "clause"),
]

# 法律名称提取
LAW_NAME_PATTERN = re.compile(r"《([^》]+)》")

# 法律分类关键词
LAW_CATEGORIES = {
    "民事法律": ["民法", "合同", "婚姻", "继承", "物权", "侵权"],
    "刑事法律": ["刑法", "刑事诉讼"],
    "行政法律": ["行政", "行政处罚", "行政许可"],
    "经济法律": ["公司", "证券", "保险", "银行", "税"],
    "社会法律": ["劳动", "社会保", "环境", "教育"],
}

# 法律概念词典（高频术语）
LEGAL_CONCEPTS = [
    "善意取得",
    "违约责任",
    "侵权责任",
    "不当得利",
    "无因管理",
    "物权",
    "债权",
    "知识产权",
    "正当防卫",
    "紧急避险",
    "代理",
    "时效",
    "抵押",
    "质押",
    "留置",
    "保证",
    "要约",
    "承诺",
    "合同解除",
    "合同终止",
    "损害赔偿",
    "精神损害",
    "连带责任",
    "补充责任",
    "法人",
    "自然人",
    "合伙",
    "信托",
    "著作权",
    "专利权",
    "商标权",
    "遗嘱",
    "法定继承",
    "代位继承",
    "仲裁",
    "诉讼",
    "调解",
    "不可抗力",
    "情势变更",
    "显失公平",
    "格式条款",
    "免责条款",
]


def _cn_to_int(cn: str) -> int:
    """将中文数字转换为阿拉伯数字（支持 0-99999）"""
    cn = cn.strip()
    if cn.isdigit():
        return int(cn)

    table = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}

    result = 0
    current = 0
    wan_part = 0

    for ch in cn:
        if ch in table:
            current = table[ch]
        elif ch in units:
            u = units[ch]
            if ch == "万":
                wan_part = (wan_part + current) * u
                current = 0
            else:
                if current == 0 and ch == "十":
                    current = 1
                wan_part += current * u
                current = 0

    result = wan_part + current
    return result if result > 0 else 0


class LegalKnowledgeGraph:
    """法律领域知识图谱（Neo4j 后端）"""

    def __init__(
        self,
        uri: str = None,
        user: str = None,
        password: str = None,
        database: str = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "password")
        self._database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver = None
        self._available = False

    # ─── 连接管理 ───────────────────────────────────────────

    def connect(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """建立 Neo4j 连接（带重试），返回是否成功"""
        import time

        for attempt in range(1, max_retries + 1):
            try:
                from neo4j import GraphDatabase

                self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password), connection_timeout=10)
                self._driver.verify_connectivity()
                self._available = True
                logger.info(f"Neo4j 连接成功: {self._uri}")
                return True
            except ImportError:
                logger.warning("neo4j 驱动未安装，图谱功能不可用。pip install neo4j>=5.0.0")
                return False
            except Exception as e:
                if self._driver:
                    self._driver.close()
                    self._driver = None
                if attempt < max_retries:
                    logger.info(f"Neo4j 连接失败 (尝试 {attempt}/{max_retries}): {e}，{retry_delay}s 后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.warning(f"Neo4j 连接失败 (已重试 {max_retries} 次): {e}，图谱功能降级")
        return False

    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._driver is not None

    # ─── Schema 管理 ────────────────────────────────────────

    def create_schema(self):
        """创建约束和索引"""
        if not self.is_available:
            return
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Law) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE (a.law_name, a.number) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (ch:Chapter) ON (ch.law_name, ch.number)",
        ]
        with self._driver.session(database=self._database) as session:
            for cypher in constraints:
                try:
                    session.run(cypher)
                except Exception as e:
                    logger.warning(f"Schema 操作失败: {cypher[:60]}... -> {e}")
        logger.info("知识图谱 Schema 创建完成")

    # ─── 中文数字转换 ──────────────────────────────────────

    @staticmethod
    def _cn_to_int(cn: str) -> int:
        return _cn_to_int(cn)

    # ─── 实体/关系抽取 ─────────────────────────────────────

    @staticmethod
    def extract_law_name(content: str) -> str:
        """从文本中提取法律名称"""
        match = LAW_NAME_PATTERN.search(content)
        return match.group(1) if match else "未知法律"

    @staticmethod
    def classify_law(law_name: str) -> str:
        """根据名称对法律分类"""
        for category, keywords in LAW_CATEGORIES.items():
            for kw in keywords:
                if kw in law_name:
                    return category
        return "其他"

    @staticmethod
    def extract_chapters(content: str) -> list[dict[str, str]]:
        """提取章节信息"""
        chapters = []
        for match in CHAPTER_PATTERN.finditer(content):
            chapters.append(
                {
                    "number": match.group(1) + match.group(2),
                    "title": match.group(3).strip(),
                }
            )
        return chapters

    @staticmethod
    def extract_articles(content: str) -> list[dict[str, Any]]:
        """提取条文信息（编号 + 全文 + 位置）"""
        articles = []
        for match in ARTICLE_PATTERN.finditer(content):
            num_str = match.group(1)
            suffix = match.group(2)
            number = f"{num_str}条" + (f"之{suffix}" if suffix else "")
            # 提取条文全文：从当前位置到下一个条文
            start = match.start()
            next_match = ARTICLE_PATTERN.search(content, match.end())
            end = next_match.start() if next_match else len(content)
            text = content[start:end].strip()
            articles.append(
                {
                    "number": number,
                    "num_int": _cn_to_int(num_str),
                    "text": text,
                    "start": start,
                }
            )
        return articles

    @staticmethod
    def extract_citations(article_text: str, current_law: str) -> list[dict[str, str]]:
        """提取条文内的引用关系"""
        citations = []
        for pattern, cite_type in CITE_PATTERNS:
            for match in pattern.finditer(article_text):
                if cite_type == "cross":
                    cited_law = match.group(1)
                    cited_article = f"{match.group(2)}条"
                    citations.append(
                        {
                            "cited_law": cited_law,
                            "cited_article": cited_article,
                            "type": "cross",
                        }
                    )
                elif cite_type == "self":
                    cited_article = f"{match.group(1)}条"
                    citations.append(
                        {
                            "cited_law": current_law,
                            "cited_article": cited_article,
                            "type": "self",
                        }
                    )
        return citations

    @staticmethod
    def extract_concepts(article_text: str) -> list[str]:
        """从条文中匹配法律概念"""
        found = []
        for concept in LEGAL_CONCEPTS:
            if concept in article_text:
                found.append(concept)
        return found

    @staticmethod
    def extract_legal_entities(query: str) -> dict[str, list[str]]:
        """从用户查询中提取法律实体（用于图谱检索的实体链接）"""
        entities: dict[str, list[str]] = {
            "laws": [],
            "articles": [],
            "concepts": [],
        }

        # 法律名称：《民法典》《刑法》等
        entities["laws"] = LAW_NAME_PATTERN.findall(query)

        # 条文引用：第X条
        for match in ARTICLE_PATTERN.finditer(query):
            num = match.group(1)
            suffix = match.group(2)
            article_id = f"{num}条" + (f"之{suffix}" if suffix else "")
            entities["articles"].append(article_id)

        # 概念匹配
        for concept in LEGAL_CONCEPTS:
            if concept in query:
                entities["concepts"].append(concept)

        return entities

    # ─── 图谱构建 ──────────────────────────────────────────

    def build_from_text(self, content: str, law_name: str = None) -> dict[str, int]:
        """从一篇法律文本构建图谱节点和关系（批量操作，返回统计）"""
        if not self.is_available:
            return {"laws": 0, "chapters": 0, "articles": 0, "citations": 0, "concepts": 0}

        if law_name is None:
            law_name = self.extract_law_name(content)
        category = self.classify_law(law_name)
        chapters = self.extract_chapters(content)
        articles = self.extract_articles(content)

        stats = {"laws": 0, "chapters": 0, "articles": 0, "citations": 0, "concepts": 0}

        # 预计算每个条文所属章节
        chapter_positions = [(ch["number"], content.find(ch["number"])) for ch in chapters]

        with self._driver.session(database=self._database) as session:
            # 1. 创建 Law 节点
            full_name = f"中华人民共和国{law_name}" if "中华人民共和国" not in law_name else law_name
            session.run(
                "MERGE (l:Law {name: $name}) SET l.full_name = $full_name, l.category = $category",
                name=law_name, full_name=full_name, category=category,
            )
            stats["laws"] = 1

            # 2. 批量创建 Chapter 节点
            if chapters:
                chapter_params = [{"law": law_name, "num": ch["number"], "title": ch["title"]} for ch in chapters]
                session.run(
                    "UNWIND $chapters AS ch "
                    "MERGE (c:Chapter {law_name: ch.law, number: ch.num}) "
                    "SET c.title = ch.title "
                    "WITH c "
                    "MATCH (l:Law {name: ch.law}) "
                    "MERGE (l)-[:HAS_CHAPTER]->(c)",
                    chapters=chapter_params,
                )
                stats["chapters"] = len(chapters)

            # 3. 批量创建 Article 节点 + 关系
            all_citations = []
            all_concepts = []
            article_params = []

            for art in articles:
                current_chapter = ""
                art_pos = art.get("start", 0)
                for ch_num, ch_pos in chapter_positions:
                    if ch_pos < art_pos:
                        current_chapter = ch_num

                article_params.append({
                    "law": law_name,
                    "num": art["number"],
                    "text": art["text"][:2000],
                    "chapter": current_chapter,
                })

                # 收集引用和概念
                citations = self.extract_citations(art["text"], law_name)
                for cite in citations:
                    all_citations.append({
                        "law": law_name, "num": art["number"],
                        "cited_law": cite["cited_law"], "cited_num": cite["cited_article"],
                    })

                concepts = self.extract_concepts(art["text"])
                for concept in concepts:
                    all_concepts.append({
                        "name": concept, "law": law_name, "num": art["number"],
                    })

            # 批量写入条文
            if article_params:
                session.run(
                    "UNWIND $articles AS art "
                    "MERGE (a:Article {law_name: art.law, number: art.num}) "
                    "SET a.text = art.text, a.chapter = art.chapter "
                    "WITH a "
                    "MATCH (l:Law {name: art.law}) "
                    "MERGE (l)-[:CONTAINS]->(a) "
                    "FOREACH (_ IN CASE WHEN art.chapter <> '' THEN [1] ELSE [] END | "
                    "  MERGE (ch:Chapter {law_name: art.law, number: art.chapter}) "
                    "  MERGE (ch)-[:CONTAINS]->(a))",
                    articles=article_params,
                )
                stats["articles"] = len(article_params)

            # 4. 批量创建引用关系
            if all_citations:
                session.run(
                    "UNWIND $citations AS cite "
                    "MATCH (a:Article {law_name: cite.law, number: cite.num}) "
                    "MATCH (cited:Article {law_name: cite.cited_law, number: cite.cited_num}) "
                    "MERGE (a)-[:CITES]->(cited)",
                    citations=all_citations,
                )
                stats["citations"] = len(all_citations)

            # 5. 批量创建概念关系
            if all_concepts:
                session.run(
                    "UNWIND $concepts AS con "
                    "MERGE (c:Concept {name: con.name}) "
                    "WITH c "
                    "MATCH (a:Article {law_name: con.law, number: con.num}) "
                    "MERGE (a)-[:DEFINES]->(c)",
                    concepts=all_concepts,
                )
                stats["concepts"] = len(all_concepts)

        logger.info(
            f"图谱构建完成 [{law_name}]: "
            f"{stats['laws']} 法律, {stats['chapters']} 章节, "
            f"{stats['articles']} 条文, {stats['citations']} 引用, "
            f"{stats['concepts']} 概念"
        )
        return stats

    def build_from_folder(self, folder_path: str) -> dict[str, int]:
        """批量导入文件夹中的所有法律文档"""
        if not self.is_available:
            return {}

        supported = (".txt", ".pdf", ".doc", ".docx")
        total_stats = {"laws": 0, "chapters": 0, "articles": 0, "citations": 0, "concepts": 0, "files": 0}

        from law_assistant.processor import DocumentProcessor

        processor = DocumentProcessor()

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(supported):
                continue
            file_path = os.path.join(folder_path, filename)
            try:
                content = processor._load_file_content(file_path)
                if not content:
                    continue
                stats = self.build_from_text(content)
                for k in total_stats:
                    if k in stats:
                        total_stats[k] += stats[k]
                total_stats["files"] += 1
            except Exception as e:
                logger.warning(f"图谱构建跳过 {filename}: {e}")

        logger.info(
            f"批量图谱构建完成: {total_stats['files']} 文件, "
            f"{total_stats['laws']} 法律, {total_stats['articles']} 条文, "
            f"{total_stats['citations']} 引用, {total_stats['concepts']} 概念"
        )
        return total_stats

    # ─── 图谱检索 ──────────────────────────────────────────

    def graph_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """图谱检索：实体链接 + 1-2 跳子图遍历

        返回: [(条文全文, 分数), ...]
        """
        if not self.is_available:
            return []

        entities = self.extract_legal_entities(query)
        results: dict[str, float] = {}

        with self._driver.session(database=self._database) as session:
            # 1. 精确条文匹配（最高权重）
            for law_name in entities["laws"]:
                for art_num in entities["articles"]:
                    records = session.run(
                        "MATCH (a:Article {law_name: $law, number: $num}) RETURN a.text AS text",
                        law=law_name,
                        num=art_num,
                    )
                    for rec in records:
                        text = rec["text"]
                        if text and text not in results:
                            results[text] = 1.0

            # 2. 法律全条文检索（如果查询指定了法律但没指定条文）
            for law_name in entities["laws"]:
                if not entities["articles"]:
                    records = session.run(
                        "MATCH (a:Article {law_name: $law}) RETURN a.text AS text LIMIT $limit",
                        law=law_name,
                        limit=top_k,
                    )
                    for rec in records:
                        text = rec["text"]
                        if text and text not in results:
                            results[text] = 0.8

            # 3. 概念关联检索（2 跳）
            for concept in entities["concepts"]:
                records = session.run(
                    "MATCH (c:Concept {name: $name}) "
                    "MATCH (a:Article)-[:DEFINES]->(c) "
                    "OPTIONAL MATCH (a)-[:CITES]->(cited:Article) "
                    "RETURN a.text AS text, collect(DISTINCT cited.text) AS cited_texts",
                    name=concept,
                )
                for rec in records:
                    text = rec["text"]
                    if text and text not in results:
                        results[text] = 0.9
                    for cited_text in rec["cited_texts"] or []:
                        if cited_text and cited_text not in results:
                            results[cited_text] = 0.7

            # 4. 全文模糊搜索（兜底）
            if not results:
                # 用关键词搜索概念
                for concept in entities["concepts"]:
                    records = session.run(
                        "MATCH (a:Article) WHERE a.text CONTAINS $keyword RETURN a.text AS text LIMIT $limit",
                        keyword=concept,
                        limit=top_k,
                    )
                    for rec in records:
                        text = rec["text"]
                        if text and text not in results:
                            results[text] = 0.5

                # 如果仍无结果，尝试全文关键词
                if not results:
                    keywords = re.findall(r"[一-鿿]{2,}", query)
                    for kw in keywords[:3]:
                        records = session.run(
                            "MATCH (a:Article) WHERE a.text CONTAINS $keyword RETURN a.text AS text LIMIT $limit",
                            keyword=kw,
                            limit=top_k // 3 + 1,
                        )
                        for rec in records:
                            text = rec["text"]
                            if text and text not in results:
                                results[text] = 0.3

        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    # ─── 图谱统计 ──────────────────────────────────────────

    def get_stats(self) -> dict[str, int]:
        """获取图谱统计信息"""
        if not self.is_available:
            return {}
        stats = {}
        _VALID_LABELS = {"Law", "Chapter", "Article", "Concept"}
        with self._driver.session(database=self._database) as session:
            for label in _VALID_LABELS:
                rec = session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()
                stats[label] = rec["cnt"]
            rec = session.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()
            stats["relationships"] = rec["cnt"]
        return stats
