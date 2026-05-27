"""Unit tests for law_assistant.splitter module."""

from law_assistant.splitter import DocumentSplitter, GeneralDocumentSplitter


class TestDocumentSplitter:
    def test_split_by_article(self):
        splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
        text = "第一条 为了保护民事主体的合法权益。第二条 民事主体的人身权利、财产权利以及其他合法权益受法律保护。"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert any("第一条" in c for c in chunks)

    def test_empty_text(self):
        splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
        chunks = splitter.split_text("")
        assert chunks == []

    def test_single_article(self):
        splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
        text = "第一条 这是一个简短的条款。"
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1


class TestGeneralDocumentSplitter:
    def test_split_long_text(self):
        splitter = GeneralDocumentSplitter(chunk_size=100, chunk_overlap=20)
        text = "这是一段很长的文本。" * 50
        chunks = splitter.split_text(text)
        assert len(chunks) > 1

    def test_short_text_single_chunk(self):
        splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)
        text = "这是一段短文本。"
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
