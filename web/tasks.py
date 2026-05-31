"""Task and utility functions extracted from app.py."""

import logging

from sqlalchemy.orm import Session

from web.models import KnowledgeGap, UploadedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level placeholder for rag_model (set by app_factory at startup)
# ---------------------------------------------------------------------------
rag_model = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# File validation helpers
# ---------------------------------------------------------------------------


def _safe_int(value, default=None):
    """Safely convert to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# MIME type validation via magic bytes (no external dependency)
_MAGIC_BYTES = {
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/zip",  # .docx is ZIP-based
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG": "image/png",
    b"BM": "image/bmp",
    b"II\x2a\x00": "image/tiff",
    b"MM\x00\x2a": "image/tiff",
}

_EXT_TO_MIME = {
    "pdf": "application/pdf",
    "docx": "application/zip",
    "txt": "text/plain",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


def _validate_file_magic(file_path: str, expected_ext: str) -> str | None:
    """校验文件实际 MIME 类型是否匹配扩展名。返回错误信息，通过则返回 None。"""
    try:
        with open(file_path, "rb") as f:
            header = f.read(16)
    except OSError:
        return "无法读取文件内容"

    detected_mime = None
    for magic, mime in _MAGIC_BYTES.items():
        if header.startswith(magic):
            detected_mime = mime
            break

    # TXT 没有 magic bytes，跳过校验
    if expected_ext == "txt":
        return None

    expected_mime = _EXT_TO_MIME.get(expected_ext)
    if detected_mime and expected_mime and detected_mime != expected_mime:
        return f"文件内容与扩展名不匹配: 扩展名 .{expected_ext}，实际类型 {detected_mime}"
    return None


# ---------------------------------------------------------------------------
# Background document processing
# ---------------------------------------------------------------------------


def _process_document_async(file_path: str, doc_id: int, knowledge_base_id: int = None):
    """后台异步处理文档索引（FAISS + BM25 + 知识图谱）"""
    from web.db import SessionLocal

    db = SessionLocal()
    try:
        doc = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id).first()
        if doc:
            doc.status = "processing"
            db.commit()
        logger.info(f"[异步] 开始处理文档索引: {file_path}")
        rag_model.add_file_documents(file_path)
        # 将文档内容写入知识图谱
        try:
            from law_assistant.processor import DocumentProcessor

            content = DocumentProcessor()._load_file_content(file_path)
            if content and rag_model.knowledge_graph.is_available:
                rag_model.knowledge_graph.build_from_text(content)
                logger.info(f"[异步] 知识图谱更新完成: {file_path}")
        except Exception as ge:
            logger.warning(f"[异步] 知识图谱更新失败（不影响主流程）: {ge}")
        rag_model.invalidate_kb_cache(knowledge_base_id)
        if doc:
            doc.status = "completed"
            db.commit()
        logger.info(f"[异步] 文档索引完成: {file_path}")
    except Exception as e:
        logger.error(f"[异步] 文档索引失败: {file_path}, 错误: {e}")
        doc = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id).first()
        if doc:
            doc.status = "failed"
            db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Document removal / knowledge gap tracking
# ---------------------------------------------------------------------------


def _remove_document_from_texts(texts: list[str], kb_id: int, filename: str):
    """后台任务：从 BM25 索引和文档注册表中移除已删除文档的文本块"""
    try:
        rag_model.bm25_retriever.remove_documents(texts)
        rag_model.bm25_retriever.save_index()
        rag_model.remove_from_registry(texts)
        rag_model.invalidate_kb_cache(kb_id)
        logger.info(f"已从BM25索引中移除文档 {filename} 的 {len(texts)} 个文本块")
    except Exception as e:
        logger.warning(f"从索引中移除文档失败: {e}")


def _track_knowledge_gap(db: Session, user_id: int, query: str, confidence: dict):
    """记录知识库缺口（相同前缀的查询合并计数）"""
    try:
        query_prefix = query[:50]
        existing = db.query(KnowledgeGap).filter(
            KnowledgeGap.query.like(f"%{query_prefix}%"),
            KnowledgeGap.status == "open",
        ).first()
        if existing:
            existing.frequency += 1
        else:
            db.add(KnowledgeGap(
                query=query,
                confidence_level=confidence.get("level", "none"),
                confidence_reason=confidence.get("reason", ""),
                user_id=user_id,
            ))
        db.commit()
    except Exception as e:
        logger.warning(f"记录知识库缺口失败: {e}")
