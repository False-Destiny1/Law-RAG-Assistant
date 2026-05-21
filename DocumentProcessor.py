import re
from typing import List, Dict, Any
import os

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

from DocumentSplitter import DocumentSplitter, GeneralDocumentSplitter


class DocumentProcessor:
    """文档处理器（支持 OCR 预处理扫描文档）"""

    def __init__(self):
        self.legal_splitter = DocumentSplitter(chunk_size=400, chunk_overlap=30)
        self.general_splitter = GeneralDocumentSplitter(chunk_size=200, chunk_overlap=20)
        self._ocr_engine = None  # 延迟加载 PaddleOCR

    def is_legal_document(self, file_path: str) -> bool:
        """判断是否为法律文档"""
        filename = os.path.basename(file_path)
        legal_keywords = ['法', '条例', '规定', '办法', '细则', '章程', '规范', '法律']
        filename_lower = filename.lower()

        # 根据文件名判断
        for keyword in legal_keywords:
            if keyword in filename_lower:
                return True

        # 如果文件名无法判断，可以进一步检查文件内容
        try:
            content = self._load_file_content(file_path)
            # 检查内容中是否包含法律特征
            if self._has_legal_characteristics(content):
                return True
        except Exception:
            pass

        return False

    def _has_legal_characteristics(self, content: str) -> bool:
        """检查内容是否具有法律文档特征"""
        legal_patterns = [
            r'第[零一二三四五六七八九十百千万\d]+条',
            r'《[^》]+》',
            r'第一章|第二章|第三章|第四章|第五章|第六章|第七章|第八章|第九章|第十章',
            r'第一条|第二条|第三条|第四条|第五条'
        ]

        for pattern in legal_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _load_file_content(self, file_path: str) -> str:
        """加载文件内容（支持 OCR 回退）"""
        if file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        elif file_path.lower().endswith('.pdf'):
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                if self._needs_ocr(documents):
                    print(f"[OCR] 文件内容检测文本不足，启用OCR: {file_path}")
                    images = self._pdf_to_images(file_path)
                    documents = self._ocr_images_to_documents(images, file_path)
            except Exception:
                print(f"[OCR] PDF加载失败，启用OCR: {file_path}")
                images = self._pdf_to_images(file_path)
                documents = self._ocr_images_to_documents(images, file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            documents = self._load_image_with_ocr(file_path)
        else:
            return ""

        return "\n".join([doc.page_content for doc in documents])

    def _get_ocr_engine(self):
        """延迟加载 PaddleOCR 引擎"""
        if self._ocr_engine is None:
            try:
                from paddleocr import PaddleOCR
                print("[OCR] 正在初始化 PaddleOCR 引擎（首次加载较慢）...")
                self._ocr_engine = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang='ch'
                )
                print("[OCR] PaddleOCR 引擎初始化完成")
            except ImportError:
                raise ImportError(
                    "需要安装 PaddleOCR 才能处理扫描文档。"
                    "请运行: pip install paddleocr paddlepaddle"
                )
        return self._ocr_engine

    def _needs_ocr(self, pages: list) -> bool:
        """检测 PDF 文本层是否不足以，需要 OCR"""
        if not pages:
            return True
        total_chars = sum(len(doc.page_content.strip()) for doc in pages)
        avg_chars = total_chars / len(pages)
        return avg_chars < 50

    def _pdf_to_images(self, file_path: str) -> list:
        """将 PDF 页面转换为 PIL Image 列表（逐页转换避免 OOM）"""
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise ImportError(
                "需要安装 pdf2image 才能处理扫描PDF。"
                "请运行: pip install pdf2image"
            )
        try:
            # 逐页转换，避免一次性加载所有页到内存
            images = convert_from_path(file_path, dpi=200, fmt='png')
            return images
        except Exception as e:
            raise RuntimeError(
                f"PDF转图片失败，请确保已安装 poppler。"
                f"Windows: conda install -c conda-forge poppler。错误: {e}"
            )

    def _ocr_images_to_documents(self, images: list, file_path: str) -> list:
        """对图片列表执行 OCR，返回 LangChain Document 列表。处理后释放图片内存。"""
        import numpy as np
        from langchain_core.documents import Document
        import gc

        ocr = self._get_ocr_engine()
        documents = []
        for i, img in enumerate(images):
            img_array = np.array(img)
            results = ocr.predict(img_array)
            page_lines = []
            for result in results:
                if hasattr(result, 'rec_texts'):
                    page_lines.extend(result.rec_texts)
            page_text = "\n".join(page_lines)
            if page_text.strip():
                documents.append(Document(
                    page_content=page_text,
                    metadata={"source": file_path, "page": i}
                ))
            else:
                print(f"[OCR] 警告: 第 {i + 1} 页 OCR 未识别到文本")
            # Release image memory after processing
            del img, img_array
        del images
        gc.collect()
        print(f"[OCR] 扫描完成，提取 {len(documents)} 页有效文本")
        return documents

    def _load_image_with_ocr(self, file_path: str) -> list:
        """对单张图片执行 OCR，返回 Document 列表"""
        import numpy as np
        from PIL import Image
        from langchain_core.documents import Document

        ocr = self._get_ocr_engine()
        img = Image.open(file_path)
        img_array = np.array(img)
        results = ocr.predict(img_array)
        lines = []
        for result in results:
            if hasattr(result, 'rec_texts'):
                lines.extend(result.rec_texts)
        text = "\n".join(lines)
        print(f"[OCR] 图片 OCR 完成: {file_path}, 提取 {len(text)} 字符")
        return [Document(page_content=text, metadata={"source": file_path, "page": 0})]

    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """处理文档，自动识别类型并采用相应分块策略"""
        if self.is_legal_document(file_path):
            return self.process_legal_document(file_path)
        else:
            return self.process_general_document(file_path)

    def process_legal_document(self, file_path: str) -> List[Dict[str, Any]]:
        """处理法律文档，返回结构化的条款"""

        # 加载文档
        documents = self._load_documents(file_path)
        structured_articles = []

        for doc in documents:
            content = doc.page_content
            articles = self._extract_structured_articles(content)
            structured_articles.extend(articles)

        return structured_articles

    def process_general_document(self, file_path: str) -> List[Dict[str, Any]]:
        """处理普通文档"""
        documents = self._load_documents(file_path)
        chunks = self.general_splitter.split_documents(documents)

        structured_chunks = []
        for i, chunk in enumerate(chunks):
            structured_chunks.append({
                'document_type': 'general',
                'chunk_number': i + 1,
                'content': chunk.page_content,
                'full_text': chunk.page_content,
                'metadata': {
                    'source': 'general_document',
                    'chunk_type': 'text_segment'
                }
            })

        return structured_chunks

    def _load_documents(self, file_path: str) -> List:
        """加载文档（支持 OCR 回退）"""
        if file_path.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        elif file_path.lower().endswith('.pdf'):
            # 快速路径：先尝试提取文本层
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                if not self._needs_ocr(pages):
                    return pages
            except Exception:
                pass  # PyPDFLoader 失败，走 OCR 路径

            # 慢速路径：扫描 PDF 检测或提取失败
            print(f"[OCR] PDF文本层不足，启用OCR: {file_path}")
            images = self._pdf_to_images(file_path)
            return self._ocr_images_to_documents(images, file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
            return loader.load()
        elif file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            return self._load_image_with_ocr(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")

    def _extract_structured_articles(self, content: str) -> List[Dict[str, Any]]:
        """从文本中提取结构化的法律条款"""

        # 匹配法律名称
        law_name_match = re.search(r'《([^》]+)》', content)
        law_name = law_name_match.group(1) if law_name_match else "未知法律"

        # 匹配条款（使用前瞻匹配到下一个"第X条"或字符串结尾）
        article_pattern = r'(第[零一二三四五六七八九十百千万\d]+条(?:之[一二三四五六七八九十百千万\d]+)?[\s\S]*?)(?=第[零一二三四五六七八九十百千万\d]+条(?:之[一二三四五六七八九十百千万\d]+)?|$)'
        articles = re.findall(article_pattern, content)

        structured_articles = []

        for i, article in enumerate(articles):
            # 清理条款文本
            article_clean = article.strip()

            # 提取条款编号
            article_num_match = re.search(r'第([零一二三四五六七八九十百千万\d]+)条', article_clean)
            article_num = article_num_match.group(1) if article_num_match else str(i + 1)

            # 提取条款内容（去掉编号部分）
            content_start = article_num_match.end() if article_num_match else 0
            article_content = article_clean[content_start:].strip()

            # 如果内容以"，"开头，去掉
            if article_content.startswith('，'):
                article_content = article_content[1:].strip()

            structured_articles.append({
                'law_name': law_name,
                'article_number': article_num,
                'article_content': article_content,
                'full_text': f"《{law_name}》第{article_num}条 {article_content}",
                'metadata': {
                    'source': 'legal_document',
                    'article_type': 'clause'
                }
            })

        return structured_articles