import re

from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter


class DocumentSplitter(TextSplitter):
    """文档分块，按条款分割"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_text(self, text: str) -> list[str]:
        """按法律条款分割文本"""

        # 匹配法律条款的正则表达式
        # 匹配 "第X条"、"第X条规定"、"《法律名称》第X条" 等格式
        article_pattern = r"第[零一二三四五六七八九十百千万\d]+条"

        # 找到所有条款的位置
        articles = []
        for match in re.finditer(article_pattern, text):
            articles.append({"start": match.start(), "text": match.group(), "content": ""})

        # 如果没有找到条款，回退到普通分块
        if not articles:
            return self._fallback_split(text)

        # 为每个条款提取内容
        chunks = []
        for i, article in enumerate(articles):
            start_pos = article["start"]

            # 确定当前条款的结束位置（下一个条款的开始或文本结尾）
            if i < len(articles) - 1:
                end_pos = articles[i + 1]["start"]
            else:
                end_pos = len(text)

            # 提取条款内容
            article_content = text[start_pos:end_pos].strip()

            # 如果条款内容过长，再进行细分
            if len(article_content) > self._chunk_size:
                sub_chunks = self._split_long_article(article_content)
                chunks.extend(sub_chunks)
            else:
                chunks.append(article_content)

        return chunks

    def _split_long_article(self, article_content: str) -> list[str]:
        """对过长的条款进行进一步分割（保留原始标点）"""
        # 按句号、分号等标点分割，使用捕获组保留分隔符
        parts = re.split(r"([。；;])", article_content)

        chunks = []
        current_chunk = ""

        for i in range(0, len(parts), 2):
            sentence = parts[i].strip()
            # 分隔符在 i+1 位置（如果存在）
            delimiter = parts[i + 1] if i + 1 < len(parts) else ""

            if not sentence:
                continue

            # 重新组合句子和其后的分隔符
            sentence_with_delim = sentence + delimiter

            # 单个句子就超过限制：用 RecursiveCharacterTextSplitter 二次分割
            if len(sentence_with_delim) > self._chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                fallback = RecursiveCharacterTextSplitter(
                    chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
                )
                chunks.extend(fallback.split_text(sentence_with_delim))
                continue

            # 如果当前块加上新句子不会超过限制
            if len(current_chunk) + len(sentence_with_delim) <= self._chunk_size:
                current_chunk += sentence_with_delim
            else:
                # 保存当前块并开始新块（附带上一块末尾的重叠）
                if current_chunk:
                    chunks.append(current_chunk)
                    # 重叠：取上一块末尾的 N 个字符作为上下文
                    overlap_text = (
                        current_chunk[-self._chunk_overlap :]
                        if len(current_chunk) > self._chunk_overlap
                        else current_chunk
                    )
                    # Ensure overlap + sentence doesn't exceed chunk_size
                    if len(overlap_text) + len(sentence_with_delim) > self._chunk_size:
                        overlap_text = overlap_text[: self._chunk_size - len(sentence_with_delim)]
                    current_chunk = overlap_text + sentence_with_delim
                else:
                    chunks.append(sentence_with_delim)
                    current_chunk = ""

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _fallback_split(self, text: str) -> list[str]:
        """回退到普通分块策略"""
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
        )
        return fallback_splitter.split_text(text)


class GeneralDocumentSplitter:
    """通用文档分块器"""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )

    def split_text(self, text: str) -> list[str]:
        """通用文档分块"""
        return self.splitter.split_text(text)

    def split_documents(self, documents: list) -> list:
        """分割文档对象"""
        return self.splitter.split_documents(documents)
