"""Unit tests for law_assistant.bm25 module."""

from law_assistant.bm25 import BM25Retriever


class TestBM25Retriever:
    def test_add_and_search(self):
        retriever = BM25Retriever(rebuild_threshold=5)
        retriever.add_documents(
            ["合同法规定了合同的订立和效力", "刑法规定了犯罪和刑罚", "民法调整平等主体之间的人身关系"]
        )
        retriever.force_rebuild()
        results = retriever.search("合同", top_k=2)
        assert len(results) > 0
        assert "合同" in results[0][0]

    def test_chinese_tokenize(self):
        retriever = BM25Retriever()
        tokens = retriever.chinese_tokenize("劳动合同解除")
        assert "劳动合同" in tokens or "劳动" in tokens

    def test_legal_terms_tokenization(self):
        retriever = BM25Retriever()
        tokens = retriever.chinese_tokenize("善意取得的构成要件")
        assert "善意取得" in tokens

    def test_empty_search(self):
        retriever = BM25Retriever()
        results = retriever.search("test", top_k=5)
        assert results == []

    def test_remove_documents(self):
        retriever = BM25Retriever(rebuild_threshold=5)
        retriever.add_documents(["文档一", "文档二", "文档三"])
        retriever.force_rebuild()
        retriever.remove_documents(["文档二"])
        assert retriever.get_document_count() == 2

    def test_build_index(self):
        retriever = BM25Retriever()
        retriever.build_index(["hello world", "foo bar"])
        assert retriever.get_document_count() == 2
        assert retriever.bm25 is not None

    def test_pending_documents_initial_build(self):
        retriever = BM25Retriever(rebuild_threshold=100)
        retriever.add_documents(["doc1", "doc2"])
        # First add triggers build since bm25 is None
        assert retriever.get_document_count() == 2
        assert retriever.bm25 is not None

    def test_pending_documents_accumulate(self):
        retriever = BM25Retriever(rebuild_threshold=100)
        retriever.build_index(["existing doc"])
        retriever.add_documents(["doc1", "doc2"])
        # Below threshold, should not rebuild
        assert retriever.get_document_count() == 3  # 1 existing + 2 pending

    def test_save_and_load_index(self, tmp_path):
        index_path = str(tmp_path / "test_bm25.json")
        retriever = BM25Retriever(bm25_index_path=index_path, rebuild_threshold=5)
        retriever.add_documents(["合同法", "刑法", "民法"])
        retriever.force_rebuild()
        retriever.save_index()

        retriever2 = BM25Retriever(bm25_index_path=index_path)
        assert retriever2.load_index() is True
        assert retriever2.get_document_count() == 3
        results = retriever2.search("合同", top_k=1)
        assert len(results) > 0
