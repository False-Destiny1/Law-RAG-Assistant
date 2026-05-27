"""Unit tests for law_assistant.graph module."""
import pytest
from unittest.mock import MagicMock
from law_assistant.graph import (
    LegalKnowledgeGraph,
    _cn_to_int,
    ARTICLE_PATTERN,
    CHAPTER_PATTERN,
    LAW_CATEGORIES,
    LEGAL_CONCEPTS,
)


# ─── _cn_to_int ─────────────────────────────────────────────

class TestCnToInt:
    def test_digit_passthrough(self):
        assert _cn_to_int("143") == 143

    def test_simple_numbers(self):
        assert _cn_to_int("一") == 1
        assert _cn_to_int("十") == 10
        assert _cn_to_int("二十") == 20
        assert _cn_to_int("五") == 5

    def test_compound_numbers(self):
        assert _cn_to_int("一百") == 100
        assert _cn_to_int("一百四十三") == 143
        assert _cn_to_int("五百八十四") == 584
        assert _cn_to_int("一千二百") == 1200

    def test_wan_unit(self):
        assert _cn_to_int("一万") == 10000
        assert _cn_to_int("五万八千") == 58000

    def test_complex_numbers(self):
        assert _cn_to_int("三百六十五") == 365
        assert _cn_to_int("二千零二十四") == 2024

    def test_zero_edge_case(self):
        # "零" alone should map to 0 via the table, but _cn_to_int returns 0 for zero
        result = _cn_to_int("零")
        assert result == 0


# ─── extract_law_name ───────────────────────────────────────

class TestExtractLawName:
    def test_basic(self):
        text = "根据《中华人民共和国民法典》第一百四十三条"
        assert LegalKnowledgeGraph.extract_law_name(text) == "中华人民共和国民法典"

    def test_short_name(self):
        text = "《刑法》第二百三十二条"
        assert LegalKnowledgeGraph.extract_law_name(text) == "刑法"

    def test_no_name(self):
        assert LegalKnowledgeGraph.extract_law_name("没有法律名称的文本") == "未知法律"

    def test_multiple_names(self):
        text = "《民法典》和《合同法》的关系"
        # 应返回第一个匹配
        assert LegalKnowledgeGraph.extract_law_name(text) == "民法典"


# ─── classify_law ───────────────────────────────────────────

class TestClassifyLaw:
    def test_civil(self):
        assert LegalKnowledgeGraph.classify_law("民法典") == "民事法律"
        assert LegalKnowledgeGraph.classify_law("合同法") == "民事法律"
        assert LegalKnowledgeGraph.classify_law("婚姻法") == "民事法律"
        assert LegalKnowledgeGraph.classify_law("物权法") == "民事法律"

    def test_criminal(self):
        assert LegalKnowledgeGraph.classify_law("刑法") == "刑事法律"
        assert LegalKnowledgeGraph.classify_law("刑事诉讼法") == "刑事法律"

    def test_administrative(self):
        assert LegalKnowledgeGraph.classify_law("行政处罚法") == "行政法律"
        assert LegalKnowledgeGraph.classify_law("行政许可法") == "行政法律"

    def test_economic(self):
        assert LegalKnowledgeGraph.classify_law("公司法") == "经济法律"
        assert LegalKnowledgeGraph.classify_law("证券法") == "经济法律"
        assert LegalKnowledgeGraph.classify_law("税法") == "经济法律"

    def test_social(self):
        assert LegalKnowledgeGraph.classify_law("劳动法") == "社会法律"
        assert LegalKnowledgeGraph.classify_law("环境保护法") == "社会法律"

    def test_unknown(self):
        assert LegalKnowledgeGraph.classify_law("科学技术进步法") == "其他"


# ─── extract_chapters ───────────────────────────────────────

class TestExtractChapters:
    def test_basic(self):
        text = "第一章 总则\n第一条 ...\n第二章 物权\n第一百一十四条 ..."
        chapters = LegalKnowledgeGraph.extract_chapters(text)
        assert len(chapters) == 2
        assert chapters[0]["number"] == "第一章"
        assert chapters[0]["title"] == "总则"
        assert chapters[1]["number"] == "第二章"
        assert chapters[1]["title"] == "物权"

    def test_pian_unit(self):
        text = "第一编 总则\n第二编 物权"
        chapters = LegalKnowledgeGraph.extract_chapters(text)
        assert len(chapters) == 2
        assert chapters[0]["number"] == "第一编"

    def test_no_chapters(self):
        text = "没有章节的文本"
        assert LegalKnowledgeGraph.extract_chapters(text) == []


# ─── extract_articles ───────────────────────────────────────

class TestExtractArticles:
    def test_basic(self):
        text = "第一条 为了保护民事主体的合法权益。第二条 民事主体的人身权利受法律保护。"
        articles = LegalKnowledgeGraph.extract_articles(text)
        assert len(articles) == 2
        assert articles[0]["number"] == "一条"
        assert "保护" in articles[0]["text"]
        assert articles[1]["number"] == "二条"

    def test_complex_numbers(self):
        text = "第一百四十三条 具备下列条件的民事法律行为有效。第一百四十四条 无民事行为能力人实施的民事法律行为无效。"
        articles = LegalKnowledgeGraph.extract_articles(text)
        assert len(articles) == 2
        assert articles[0]["number"] == "一百四十三条"
        assert articles[1]["number"] == "一百四十四条"

    def test_digit_format(self):
        text = "第143条 条文内容。第144条 另一条文。"
        articles = LegalKnowledgeGraph.extract_articles(text)
        assert len(articles) == 2
        assert articles[0]["number"] == "143条"

    def test_no_articles(self):
        text = "这是普通文本，没有法律条文。"
        assert LegalKnowledgeGraph.extract_articles(text) == []

    def test_article_with_suffix(self):
        text = "第一百四十三条之一 补充规定。第一百四十四条 正文。"
        articles = LegalKnowledgeGraph.extract_articles(text)
        assert len(articles) == 2
        assert articles[0]["number"] == "一百四十三条之一"


# ─── extract_citations ──────────────────────────────────────

class TestExtractCitations:
    def test_self_cite(self):
        text = "依照本法第五百八十四条的规定承担赔偿责任"
        cites = LegalKnowledgeGraph.extract_citations(text, "民法典")
        assert len(cites) >= 1
        assert any(c["type"] == "self" and "五百八十四" in c["cited_article"] for c in cites)

    def test_self_cite_short(self):
        text = "本法第一百四十三条"
        cites = LegalKnowledgeGraph.extract_citations(text, "民法典")
        assert len(cites) >= 1
        assert cites[0]["cited_law"] == "民法典"

    def test_cross_cite(self):
        text = "依据《合同法》第一百零七条的规定"
        cites = LegalKnowledgeGraph.extract_citations(text, "民法典")
        assert len(cites) >= 1
        cross = [c for c in cites if c["type"] == "cross"]
        assert len(cross) >= 1
        assert cross[0]["cited_law"] == "合同法"

    def test_no_citation(self):
        text = "本条没有任何引用"
        assert LegalKnowledgeGraph.extract_citations(text, "民法典") == []


# ─── extract_concepts ───────────────────────────────────────

class TestExtractConcepts:
    def test_basic(self):
        text = "善意取得的，受让人取得不动产或者动产的所有权。"
        concepts = LegalKnowledgeGraph.extract_concepts(text)
        assert "善意取得" in concepts

    def test_multiple(self):
        text = "因违约责任产生的损害赔偿请求权，适用诉讼时效的规定。"
        concepts = LegalKnowledgeGraph.extract_concepts(text)
        assert "违约责任" in concepts
        assert "时效" in concepts

    def test_no_concepts(self):
        text = "这是普通文本"
        assert LegalKnowledgeGraph.extract_concepts(text) == []

    def test_concept_list_non_empty(self):
        assert len(LEGAL_CONCEPTS) >= 20


# ─── extract_legal_entities ─────────────────────────────────

class TestExtractLegalEntities:
    def test_full_query(self):
        query = "《民法典》第一百四十三条关于善意取得的规定是什么？"
        entities = LegalKnowledgeGraph.extract_legal_entities(query)
        assert "民法典" in entities["laws"]
        assert any("一百四十三" in a for a in entities["articles"])
        assert "善意取得" in entities["concepts"]

    def test_no_entities(self):
        query = "今天天气怎么样？"
        entities = LegalKnowledgeGraph.extract_legal_entities(query)
        assert entities["laws"] == []
        assert entities["articles"] == []
        assert entities["concepts"] == []

    def test_multiple_laws(self):
        query = "《民法典》和《合同法》在违约责任方面有什么关系？"
        entities = LegalKnowledgeGraph.extract_legal_entities(query)
        assert "民法典" in entities["laws"]
        assert "合同法" in entities["laws"]
        assert "违约责任" in entities["concepts"]


# ─── LegalKnowledgeGraph (mocked Neo4j) ─────────────────────

class TestLegalKnowledgeGraphInit:
    def test_default_config(self):
        graph = LegalKnowledgeGraph()
        assert graph._uri == "bolt://localhost:7687"
        assert graph._user == "neo4j"
        assert graph.is_available is False

    def test_custom_config(self):
        graph = LegalKnowledgeGraph(uri="bolt://remote:7687", user="admin", password="secret")
        assert graph._uri == "bolt://remote:7687"
        assert graph._user == "admin"

    def test_connect_failure_graceful(self):
        """连接失败时应优雅降级，不抛异常"""
        graph = LegalKnowledgeGraph(uri="bolt://nonexistent:9999")
        result = graph.connect()
        assert result is False
        assert graph.is_available is False

    def test_close_without_connect(self):
        graph = LegalKnowledgeGraph()
        graph.close()  # 不应抛异常


class TestGraphSearchWithoutNeo4j:
    def test_search_returns_empty(self):
        graph = LegalKnowledgeGraph()
        results = graph.graph_search("《民法典》第一百四十三条")
        assert results == []

    def test_build_returns_empty(self):
        graph = LegalKnowledgeGraph()
        stats = graph.build_from_text("第一条 测试条文")
        assert stats == {"laws": 0, "chapters": 0, "articles": 0, "citations": 0, "concepts": 0}

    def test_get_stats_empty(self):
        graph = LegalKnowledgeGraph()
        assert graph.get_stats() == {}


class TestBuildFromText:
    def test_build_calls_neo4j(self):
        """验证 build_from_text 正确调用 Neo4j session"""
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        graph = LegalKnowledgeGraph()
        graph._driver = mock_driver
        graph._available = True

        text = (
            "《中华人民共和国民法典》\n"
            "第一章 总则\n"
            "第一条 为了保护民事主体的合法权益。\n"
            "第二条 民事主体的人身权利受法律保护。\n"
        )
        stats = graph.build_from_text(text, law_name="民法典")

        assert stats["laws"] == 1
        assert stats["chapters"] == 1
        assert stats["articles"] == 2
        assert mock_session.run.called


class TestGraphSearchMocked:
    def _make_graph_with_results(self, mock_records):
        """Helper: 创建一个 mock 的可用图谱"""
        graph = LegalKnowledgeGraph()
        graph._available = True
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter(mock_records))
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        graph._driver = mock_driver
        return graph, mock_session

    def test_exact_article_match(self):
        """精确条文匹配应返回高分"""
        mock_record = {"text": "第一百四十三条 具备下列条件的民事法律行为有效"}
        graph, _ = self._make_graph_with_results([mock_record])

        results = graph.graph_search("《民法典》第一百四十三条")
        assert len(results) == 1
        assert results[0][1] == 1.0  # 最高分

    def test_concept_search(self):
        """概念搜索应返回相关条文"""
        mock_record = {"text": "善意取得的，受让人取得所有权", "cited_texts": []}
        graph, _ = self._make_graph_with_results([mock_record])

        results = graph.graph_search("什么是善意取得？")
        assert len(results) >= 1


# ─── ARTICLE_PATTERN regex ──────────────────────────────────

class TestArticlePattern:
    def test_chinese_number(self):
        match = ARTICLE_PATTERN.search("第一百四十三条")
        assert match is not None
        assert match.group(1) == "一百四十三"

    def test_digit_number(self):
        match = ARTICLE_PATTERN.search("第143条")
        assert match is not None
        assert match.group(1) == "143"

    def test_with_suffix(self):
        match = ARTICLE_PATTERN.search("第一百四十三条之一")
        assert match is not None
        assert match.group(1) == "一百四十三"
        assert match.group(2) == "一"

    def test_no_match(self):
        assert ARTICLE_PATTERN.search("没有条文") is None


# ─── CHAPTER_PATTERN regex ──────────────────────────────────

class TestChapterPattern:
    def test_chapter(self):
        match = CHAPTER_PATTERN.search("第一章 总则")
        assert match is not None
        assert match.group(2) == "章"
        assert match.group(3) == "总则"

    def test_pian(self):
        match = CHAPTER_PATTERN.search("第一编 总则")
        assert match is not None
        assert match.group(2) == "编"

    def test_no_match(self):
        assert CHAPTER_PATTERN.search("普通文本") is None
