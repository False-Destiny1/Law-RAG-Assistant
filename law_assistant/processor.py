import logging
import os
import re
from typing import Any

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader

from law_assistant.splitter import DocumentSplitter, GeneralDocumentSplitter

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器（支持 OCR 预处理扫描文档）"""

    def __init__(self):
        self.legal_splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
        self.general_splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)
        self._ocr_engine = None  # 延迟加载 PaddleOCR

    def is_legal_document(self, file_path: str) -> bool:
        """判断是否为法律文档（独立使用时会加载文件，推荐用 process_document 避免重复加载）"""
        filename = os.path.basename(file_path)
        legal_keywords = ["法", "条例", "规定", "办法", "细则", "章程", "规范", "法律"]
        filename_lower = filename.lower()

        # 根据文件名判断
        for keyword in legal_keywords:
            if keyword in filename_lower:
                return True

        # 如果文件名无法判断，检查文件内容
        try:
            content = self._load_file_content(file_path)
            if self._has_legal_characteristics(content):
                return True
        except Exception:
            pass

        return False

    def _is_legal_content(self, file_path: str, content: str) -> bool:
        """基于文件名和已加载内容判断是否为法律文档（不重复加载文件）"""
        # JSON 数据集（CAIL、问答对等）不走法律文档路径
        if file_path.lower().endswith(".json"):
            return False

        filename = os.path.basename(file_path)
        legal_keywords = ["法", "条例", "规定", "办法", "细则", "章程", "规范", "法律"]
        filename_lower = filename.lower()

        for keyword in legal_keywords:
            if keyword in filename_lower:
                return True

        return bool(self._has_legal_characteristics(content))

    def _has_legal_characteristics(self, content: str) -> bool:
        """检查内容是否具有法律文档特征"""
        legal_patterns = [
            r"第[零一二三四五六七八九十百千万\d]+条",
            r"《[^》]+》",
            r"第[零一二三四五六七八九十百千\d]+[章节]",
        ]

        return any(re.search(pattern, content) for pattern in legal_patterns)

    def _load_file_content(self, file_path: str) -> str:
        """加载文件内容（支持 OCR 回退）"""
        if file_path.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        elif file_path.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                if self._needs_ocr(documents):
                    logger.info(f"[OCR] 文件内容检测文本不足，启用OCR: {file_path}")
                    documents = self._ocr_pdf_to_documents(file_path)
            except Exception:
                logger.info(f"[OCR] PDF加载失败，启用OCR: {file_path}")
                documents = self._ocr_pdf_to_documents(file_path)
        elif file_path.lower().endswith((".doc", ".docx")):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        elif file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            documents = self._load_image_with_ocr(file_path)
        else:
            return ""

        return "\n".join([doc.page_content for doc in documents])

    def _get_ocr_engine(self):
        """延迟加载 PaddleOCR 引擎"""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR

                logger.info("[OCR] 正在初始化 PaddleOCR 引擎（首次加载较慢）...")
                self._ocr_engine = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang="ch",
                )
                logger.info("[OCR] PaddleOCR 引擎初始化完成")
            except ImportError:
                raise ImportError(
                    "需要安装 PaddleOCR 才能处理扫描文档。请运行: pip install paddleocr paddlepaddle"
                ) from None
        return self._ocr_engine

    def _needs_ocr(self, pages: list) -> bool:
        """检测 PDF 文本层是否不足以，需要 OCR"""
        if not pages:
            return True
        total_chars = sum(len(doc.page_content.strip()) for doc in pages)
        avg_chars = total_chars / len(pages)
        return avg_chars < 50

    def _pdf_to_images(self, file_path: str) -> list:
        """将 PDF 页面转换为 PIL Image 列表（兼容旧调用）"""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError("需要安装 pdf2image 才能处理扫描PDF。请运行: pip install pdf2image") from None
        try:
            images = convert_from_path(file_path, dpi=200, fmt="png")
            return images
        except Exception as e:
            raise RuntimeError(
                f"PDF转图片失败，请确保已安装 poppler。Windows: conda install -c conda-forge poppler。错误: {e}"
            ) from e

    def _ocr_pdf_to_documents(self, file_path: str) -> list:
        """逐页将 PDF 转图片并 OCR，返回 Document 列表（逐页释放内存，避免 OOM）"""
        import gc

        import numpy as np
        from langchain_core.documents import Document
        from pdf2image import convert_from_path, pdfinfo_from_path

        ocr = self._get_ocr_engine()
        try:
            info = pdfinfo_from_path(file_path)
            total_pages = info.get("Pages", 0)
        except Exception:
            total_pages = 0

        if total_pages == 0:
            logger.warning(f"无法获取 PDF 页数，回退到全量加载: {file_path}")
            images = self._pdf_to_images(file_path)
            return self._ocr_images_to_documents(images, file_path)

        documents = []
        for page_num in range(1, total_pages + 1):
            try:
                images = convert_from_path(file_path, dpi=200, fmt="png", first_page=page_num, last_page=page_num)
                if not images:
                    continue
                img = images[0]
                img_array = np.array(img)
                results = ocr.predict(img_array)
                page_lines = []
                for result in results:
                    if hasattr(result, "rec_texts"):
                        page_lines.extend(result.rec_texts)
                page_text = "\n".join(page_lines)
                if page_text.strip():
                    documents.append(
                        Document(page_content=page_text, metadata={"source": file_path, "page": page_num - 1})
                    )
                else:
                    logger.warning(f"[OCR] 第 {page_num} 页 OCR 未识别到文本")
                del img, img_array, images
            except Exception as e:
                logger.warning(f"[OCR] 第 {page_num} 页处理失败: {e}")
            if page_num % 10 == 0:
                gc.collect()
        gc.collect()
        logger.info(f"[OCR] 扫描完成，提取 {len(documents)}/{total_pages} 页有效文本")
        return documents

    def _ocr_images_to_documents(self, images: list, file_path: str) -> list:
        """对图片列表执行 OCR，返回 LangChain Document 列表。处理后释放图片内存。"""
        import gc

        import numpy as np
        from langchain_core.documents import Document

        ocr = self._get_ocr_engine()
        documents = []
        for i, img in enumerate(images):
            img_array = np.array(img)
            results = ocr.predict(img_array)
            page_lines = []
            for result in results:
                if hasattr(result, "rec_texts"):
                    page_lines.extend(result.rec_texts)
            page_text = "\n".join(page_lines)
            if page_text.strip():
                documents.append(Document(page_content=page_text, metadata={"source": file_path, "page": i}))
            else:
                logger.warning(f"[OCR] 第 {i + 1} 页 OCR 未识别到文本")
            del img, img_array
        del images
        gc.collect()
        logger.info(f"[OCR] 扫描完成，提取 {len(documents)} 页有效文本")
        return documents

    def _load_image_with_ocr(self, file_path: str) -> list:
        """对单张图片执行 OCR，返回 Document 列表"""
        import numpy as np
        from langchain_core.documents import Document
        from PIL import Image

        ocr = self._get_ocr_engine()
        img = Image.open(file_path)
        img_array = np.array(img)
        results = ocr.predict(img_array)
        lines = []
        for result in results:
            if hasattr(result, "rec_texts"):
                lines.extend(result.rec_texts)
        text = "\n".join(lines)
        logger.info(f"[OCR] 图片 OCR 完成: {file_path}, 提取 {len(text)} 字符")
        return [Document(page_content=text, metadata={"source": file_path, "page": 0})]

    def process_document(self, file_path: str) -> list[dict[str, Any]]:
        """处理文档，自动识别类型并采用相应分块策略（文件只加载一次）"""
        # 一次性加载文档
        documents = self._load_documents(file_path)
        content = "\n".join([doc.page_content for doc in documents])

        # 基于已加载内容判断类型（避免二次加载）
        if self._is_legal_content(file_path, content):
            return self._process_legal_from_docs(documents)
        else:
            return self._process_general_from_docs(documents)

    def _process_legal_from_docs(self, documents: list) -> list[dict[str, Any]]:
        """从已加载的文档列表中提取结构化法律条款"""
        structured_articles = []
        for doc in documents:
            content = doc.page_content
            articles = self._extract_structured_articles(content)
            structured_articles.extend(articles)
        return structured_articles

    def _process_general_from_docs(self, documents: list) -> list[dict[str, Any]]:
        """从已加载的文档列表中生成通用分块"""
        chunks = self.general_splitter.split_documents(documents)

        structured_chunks = []
        for i, chunk in enumerate(chunks):
            structured_chunks.append(
                {
                    "document_type": "general",
                    "chunk_number": i + 1,
                    "content": chunk.page_content,
                    "full_text": chunk.page_content,
                    "metadata": {"source": "general_document", "chunk_type": "text_segment"},
                }
            )

        return structured_chunks

    def _load_documents(self, file_path: str) -> list:
        """加载文档（支持 OCR 回退）"""
        if file_path.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            return loader.load()
        elif file_path.lower().endswith(".pdf"):
            # 快速路径：先尝试提取文本层
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                if not self._needs_ocr(pages):
                    return pages
            except Exception:
                pass  # PyPDFLoader 失败，走 OCR 路径

            # 慢速路径：扫描 PDF 检测或提取失败
            logger.info(f"[OCR] PDF文本层不足，启用OCR: {file_path}")
            return self._ocr_pdf_to_documents(file_path)
        elif file_path.lower().endswith((".doc", ".docx")):
            loader = Docx2txtLoader(file_path)
            return loader.load()
        elif file_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            return self._load_image_with_ocr(file_path)
        elif file_path.lower().endswith(".json"):
            return self._load_json_documents(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

    def _load_json_documents(self, file_path: str) -> list:
        """加载 JSON/JSONL 格式的法律数据集

        支持两种格式:
        - JSON 数组: [{"id": ..., "input": ..., "output": ...}, ...]
        - JSONL (每行一个 JSON): 支持 CAIL 格式 {fact, meta} 和问答格式 {input, output}
        """
        import json as json_mod

        from langchain_core.documents import Document

        documents = []

        with open(file_path, encoding="utf-8") as f:
            content = f.read().strip()

        # 尝试 JSON 数组格式
        if content.startswith("["):
            try:
                records = json_mod.loads(content)
            except json_mod.JSONDecodeError:
                records = []
        else:
            # JSONL 格式：逐行解析
            records = []
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    try:
                        records.append(json_mod.loads(line))
                    except json_mod.JSONDecodeError:
                        continue

        for i, record in enumerate(records):
            # CAIL 格式：{fact, meta}
            if "fact" in record and "meta" in record:
                meta = record["meta"]
                accusations = ", ".join(meta.get("accusation", []))
                articles = ", ".join(str(a) for a in meta.get("relevant_articles", []))
                text = f"案件事实：{record['fact']}"
                if accusations:
                    text += f"\n罪名：{accusations}"
                if articles:
                    text += f"\n相关法条：{articles}"
                doc_id = f"cail_{i}"
            # 问答格式：{id, input, output}
            elif "input" in record and "output" in record:
                text = f"问题：{record['input']}\n回答：{record['output']}"
                doc_id = record.get("id", f"qa_{i}")
            # 通用格式：尝试拼接所有字段
            else:
                text = "\n".join(f"{k}: {v}" for k, v in record.items() if v)
                doc_id = record.get("id", f"doc_{i}")

            documents.append(Document(page_content=text, metadata={"source": file_path, "id": doc_id}))

        logger.info(f"[JSON] 从 {file_path} 加载了 {len(documents)} 条记录")
        return documents

    def _extract_structured_articles(self, content: str) -> list[dict[str, Any]]:
        """从文本中提取结构化的法律条款"""

        # 匹配法律名称
        law_name_match = re.search(r"《([^》]+)》", content)
        law_name = law_name_match.group(1) if law_name_match else "未知法律"

        # 匹配条款（使用前瞻匹配到下一个"第X条"或字符串结尾）
        article_pattern = r"(第[零一二三四五六七八九十百千万\d]+条(?:之[一二三四五六七八九十百千万\d]+)?[\s\S]*?)(?=第[零一二三四五六七八九十百千万\d]+条(?:之[一二三四五六七八九十百千万\d]+)?|$)"
        articles = re.findall(article_pattern, content)

        structured_articles = []

        for i, article in enumerate(articles):
            # 清理条款文本
            article_clean = article.strip()

            # 提取条款编号
            article_num_match = re.search(r"第([零一二三四五六七八九十百千万\d]+)条", article_clean)
            article_num = article_num_match.group(1) if article_num_match else str(i + 1)

            # 提取条款内容（去掉编号部分）
            content_start = article_num_match.end() if article_num_match else 0
            article_content = article_clean[content_start:].strip()

            # 如果内容以"，"开头，去掉
            if article_content.startswith("，"):
                article_content = article_content[1:].strip()

            structured_articles.append(
                {
                    "law_name": law_name,
                    "article_number": article_num,
                    "article_content": article_content,
                    "full_text": f"《{law_name}》第{article_num}条 {article_content}",
                    "metadata": {"source": "legal_document", "article_type": "clause"},
                }
            )

        return structured_articles
