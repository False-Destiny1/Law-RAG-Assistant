"""All route handlers extracted from app.py.

Uses FastAPI APIRouter; the ``router`` instance is mounted by the
application factory in ``web/app_factory.py``.
"""

import json as _json
import logging
import os
import re
import time as _time
import uuid
from contextlib import suppress
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session, joinedload, subqueryload

from web.auth import require_user
from web.db import SessionLocal, get_db
from web.metrics import metrics
from web.models import (
    Chat, InterventionRequest, KnowledgeBase, KnowledgeGap,
    Message, MessageFeedback, UploadedDocument, User,
)
from web.security import check_injection
from web.tasks import (
    _process_document_async,
    _remove_document_from_texts,
    _safe_int,
    _track_knowledge_gap,
    _validate_file_magic,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level placeholders — set by app_factory at startup
# ---------------------------------------------------------------------------
rag_model = None  # type: ignore[assignment]
templates = None  # type: ignore[assignment]

router = APIRouter()

# ── Paths ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads")
KNOWLEDGE_BASE_FOLDER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge_base")


# ══════════════════════════════════════════════════════════════════════
# Auth routes
# ══════════════════════════════════════════════════════════════════════


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, registered: str = None):
    context = {}
    if registered:
        context["success"] = "注册成功，请登录"
    return templates.TemplateResponse(request, "login.html", context)


@router.post("/login")
def login_submit(
    request: Request,
    identifier: str = Form(...),
    password: str = Form(...),
    remember: str | None = Form(None),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.phone == identifier).first()
    if not user or not user.check_password(password):
        return templates.TemplateResponse(request, "login.html", {"error": "手机号或密码错误"})

    from web.auth import create_session_token, SESSION_EXPIRE_HOURS
    token = create_session_token(user.id)
    response = RedirectResponse(url="/", status_code=303)
    max_age = SESSION_EXPIRE_HOURS * 3600 if remember else None
    is_production = os.getenv("ENV", "").lower() == "production"
    response.set_cookie("session_token", token, httponly=True, max_age=max_age, samesite="lax", secure=is_production)
    return response


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html")


@router.post("/register")
def register_submit(
    request: Request,
    phone: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
    db: Session = Depends(get_db),
):
    errors = []
    if not all([phone, username, password, confirm_password]):
        errors.append("请填写所有必填字段")
    if not re.match(r"^1[3-9]\d{9}$", phone):
        errors.append("请输入有效的11位手机号")
    if len(password) < 8:
        errors.append("密码长度至少为8位")
    if not re.search(r"[a-zA-Z]", password) or not re.search(r"\d", password):
        errors.append("密码必须包含字母和数字")
    if password != confirm_password:
        errors.append("两次输入的密码不一致")
    if db.query(User).filter(User.phone == phone).first():
        errors.append("该手机号已注册")
    if db.query(User).filter(User.username == username).first():
        errors.append("该用户名已存在")

    if errors:
        return templates.TemplateResponse(request, "register.html", {"errors": errors})

    new_user = User(phone=phone, username=username, role="user")
    new_user.set_password(password)
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login?registered=1", status_code=303)


@router.post("/logout")
def logout(request: Request):
    token = request.cookies.get("session_token", "")
    if token:
        try:
            from law_assistant.redis_utils import blacklist_token
            from web.auth import SESSION_EXPIRE_HOURS
            blacklist_token(token, ttl_seconds=SESSION_EXPIRE_HOURS * 3600)
        except Exception as e:
            logger.warning(f"Token 黑名单写入失败: {e}")
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token")
    response.delete_cookie("csrf_token")
    return response


@router.post("/api/change-password")
def change_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    if not user.check_password(old_password):
        return JSONResponse({"error": "当前密码错误"}, status_code=400)
    if len(new_password) < 8:
        return JSONResponse({"error": "新密码长度不能少于8位"}, status_code=400)
    if not re.search(r"[a-zA-Z]", new_password) or not re.search(r"\d", new_password):
        return JSONResponse({"error": "新密码必须包含字母和数字"}, status_code=400)
    user.set_password(new_password)
    db.commit()
    return {"success": True, "message": "密码修改成功"}


# ══════════════════════════════════════════════════════════════════════
# Page routes
# ══════════════════════════════════════════════════════════════════════


@router.get("/", response_class=HTMLResponse)
def home(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse(request, "index.html", {"user": user})


@router.get("/knowledge-bases", response_class=HTMLResponse)
def knowledge_bases_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    kb_list = (
        db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user.id).order_by(KnowledgeBase.updated_at.desc()).all()
    )
    return templates.TemplateResponse(request, "knowledge_base.html", {"user": user, "knowledge_bases": kb_list})


@router.get("/knowledge-base/create", response_class=HTMLResponse)
def create_kb_page(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse(request, "create_knowledge_base.html", {"user": user})


@router.post("/knowledge-base/create")
def create_kb_submit(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    new_kb = KnowledgeBase(user_id=user.id, name=name, description=description)
    db.add(new_kb)
    db.commit()
    return RedirectResponse(url="/knowledge-bases", status_code=303)


@router.get("/knowledge-base/{kb_id}/edit", response_class=HTMLResponse)
def edit_kb_page(kb_id: int, request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user.id).first()
    if not kb:
        return RedirectResponse(url="/knowledge-bases", status_code=303)
    return templates.TemplateResponse(request, "edit_knowledge_base.html", {"user": user, "kb": kb})


@router.post("/knowledge-base/{kb_id}/edit")
def edit_kb_submit(
    kb_id: int,
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user.id).first()
    if not kb:
        return RedirectResponse(url="/knowledge-bases", status_code=303)
    kb.name = name
    kb.description = description
    db.commit()
    return RedirectResponse(url="/knowledge-bases", status_code=303)


@router.delete("/knowledge-base/{kb_id}")
def delete_kb(
    kb_id: int,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user.id).first()
    if not kb:
        return JSONResponse({"error": "知识库不存在"}, status_code=404)
    # 删除前提取所有文档的文本（文件删除后就无法再读取）
    all_texts = []
    for doc in kb.documents:
        if os.path.exists(doc.file_path):
            try:
                from law_assistant.processor import DocumentProcessor

                chunks = DocumentProcessor().process_document(doc.file_path)
                all_texts.extend([c["full_text"] for c in chunks])
            except Exception as e:
                logger.warning(f"提取文档 {doc.filename} 文本失败: {e}")
            with suppress(Exception):
                os.remove(doc.file_path)
    # 后台清理 BM25 索引
    if all_texts:
        background_tasks.add_task(_remove_document_from_texts, all_texts, kb_id, f"知识库#{kb_id}")
    rag_model.invalidate_kb_cache(kb_id)
    rag_model.mark_dirty()
    db.delete(kb)
    db.commit()
    return JSONResponse({"success": True})


@router.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    if user.role not in ["expert", "admin"]:
        return RedirectResponse(url="/", status_code=303)
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user.id).order_by(KnowledgeBase.name).all()
    docs = (
        db.query(UploadedDocument)
        .filter(UploadedDocument.user_id == user.id)
        .order_by(UploadedDocument.uploaded_at.desc())
        .all()
    )
    return templates.TemplateResponse(
        request, "upload.html", {"user": user, "knowledge_bases": kbs, "uploaded_docs": docs}
    )


@router.post("/upload")
async def upload_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    knowledge_base_id: str | None = Form(None),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    if user.role not in ["expert", "admin"]:
        return RedirectResponse(url="/", status_code=303)

    allowed_extensions = {"pdf", "docx", "txt", "jpg", "jpeg", "png", "bmp", "tiff"}
    if not file.filename:
        return JSONResponse({"error": "请选择文件"}, status_code=400)
    file_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        supported = ", ".join(sorted(allowed_extensions))
        return JSONResponse({"error": f"不支持的文件格式: .{file_ext}，支持: {supported}"}, status_code=400)

    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20MB
    # Chunked reading to check size without buffering entire file in memory
    chunks = []
    total_size = 0
    while True:
        chunk = await file.read(8192)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_UPLOAD_SIZE:
            return JSONResponse({"error": "文件大小超过20MB限制"}, status_code=413)
        chunks.append(chunk)
    content = b"".join(chunks)
    with open(file_path, "wb") as f:
        f.write(content)

    # MIME 校验：检查文件实际内容是否匹配扩展名
    mime_error = _validate_file_magic(file_path, file_ext)
    if mime_error:
        with suppress(Exception):
            os.remove(file_path)
        return JSONResponse({"error": mime_error}, status_code=400)

    new_doc = UploadedDocument(
        user_id=user.id,
        knowledge_base_id=_safe_int(knowledge_base_id),
        filename=file.filename,
        file_path=file_path,
        file_type=file_ext,
        file_size=len(content),
    )
    db.add(new_doc)
    db.commit()

    # P10: 异步处理文档索引，不阻塞上传响应
    background_tasks.add_task(_process_document_async, file_path, new_doc.id, new_doc.knowledge_base_id)

    return RedirectResponse(url="/upload", status_code=303)


@router.delete("/upload/{doc_id}")
def delete_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    doc = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id, UploadedDocument.user_id == user.id).first()
    if not doc:
        return JSONResponse({"error": "文档不存在"}, status_code=404)
    # 删除前提取文本（文件删除后就无法再读取）
    file_path = doc.file_path
    kb_id = doc.knowledge_base_id
    filename = doc.filename
    texts_to_remove = []
    if os.path.exists(file_path):
        try:
            from law_assistant.processor import DocumentProcessor

            chunks = DocumentProcessor().process_document(file_path)
            texts_to_remove = [c["full_text"] for c in chunks]
        except Exception as e:
            logger.warning(f"提取文档文本失败: {e}")
        with suppress(Exception):
            os.remove(file_path)
    # 后台清理 BM25 索引
    if texts_to_remove:
        background_tasks.add_task(_remove_document_from_texts, texts_to_remove, kb_id, filename)
    rag_model.mark_dirty()
    db.delete(doc)
    db.commit()
    return JSONResponse({"success": True})


@router.post("/upload/{doc_id}/reprocess")
def reprocess_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    if user.role not in ["expert", "admin"]:
        return JSONResponse({"error": "权限不足"}, status_code=403)
    doc = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id, UploadedDocument.user_id == user.id).first()
    if not doc:
        return JSONResponse({"error": "文档不存在"}, status_code=404)
    if not os.path.exists(doc.file_path):
        return JSONResponse({"error": "文件已丢失，无法重新处理"}, status_code=404)
    doc.status = "pending"
    db.commit()
    background_tasks.add_task(_process_document_async, doc.file_path, doc.id, doc.knowledge_base_id)
    return JSONResponse({"success": True, "status": "pending"})


# ══════════════════════════════════════════════════════════════════════
# API routes
# ══════════════════════════════════════════════════════════════════════


@router.get("/api/chats")
def get_chats(user: User = Depends(require_user), db: Session = Depends(get_db)):
    chats = (
        db.query(Chat)
        .options(joinedload(Chat.knowledge_base))
        .filter(Chat.user_id == user.id)
        .order_by(Chat.updated_at.desc())
        .all()
    )
    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
            "knowledge_base_id": c.knowledge_base_id,
            "knowledge_base_name": c.knowledge_base.name if c.knowledge_base else None,
        }
        for c in chats
    ]


@router.get("/api/chats/{chat_id}")
def get_chat_messages(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = (
        db.query(Chat)
        .options(joinedload(Chat.knowledge_base))
        .filter(Chat.id == chat_id, Chat.user_id == user.id)
        .first()
    )
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
    return {
        "id": chat.id,
        "title": chat.title,
        "knowledge_base_id": chat.knowledge_base_id,
        "knowledge_base_name": chat.knowledge_base.name if chat.knowledge_base else None,
        "messages": [
            {"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in messages
        ],
    }


@router.post("/api/chats")
def create_chat(user: User = Depends(require_user), db: Session = Depends(get_db)):
    new_chat = Chat(user_id=user.id, title="新对话")
    db.add(new_chat)
    db.flush()

    initial_msg = Message(chat_id=new_chat.id, role="bot", content="您好！我是智能法律助手，请问有什么可以帮您的吗？")
    db.add(initial_msg)
    db.commit()
    db.refresh(new_chat)

    return {
        "id": new_chat.id,
        "title": new_chat.title,
        "created_at": new_chat.created_at.isoformat(),
        "updated_at": new_chat.updated_at.isoformat() if new_chat.updated_at else new_chat.created_at.isoformat(),
    }


@router.put("/api/chats/{chat_id}")
async def update_chat(
    chat_id: int, request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)
):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)

    data = await request.json()
    if "title" in data:
        chat.title = data["title"]
    if "knowledge_base_id" in data:
        kb_id_val = data["knowledge_base_id"]
        if kb_id_val is not None:
            # Verify KB belongs to current user
            kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id_val, KnowledgeBase.user_id == user.id).first()
            if not kb:
                return JSONResponse({"error": "知识库不存在或无权访问"}, status_code=403)
        chat.knowledge_base_id = kb_id_val
    chat.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {
        "id": chat.id,
        "title": chat.title,
        "knowledge_base_id": chat.knowledge_base_id,
        "updated_at": chat.updated_at.isoformat() if chat.updated_at else None,
    }


@router.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    rag_model.clear_conversation_memory(f"chat_{chat_id}")
    db.delete(chat)
    db.commit()
    return {"success": True}


@router.get("/api/chats/{chat_id}/export")
def export_chat(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at.asc()).all()
    lines = [f"# {chat.title or '对话记录'}\n"]
    kb_name = chat.knowledge_base.name if chat.knowledge_base else "未指定"
    lines.append(f"知识库: {kb_name}\n")
    lines.append(f"导出时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n---\n")
    for msg in messages:
        role = "用户" if msg.role == "user" else "法律助手"
        ts = msg.created_at.strftime("%Y-%m-%d %H:%M") if msg.created_at else ""
        lines.append(f"**{role}** ({ts})\n\n{msg.content}\n\n---\n")
    content = "\n".join(lines)
    return StreamingResponse(
        iter([content]),
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=chat_{chat_id}.md"},
    )


@router.post("/api/chats/{chat_id}/clear_memory")
def clear_chat_memory(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    rag_model.clear_conversation_memory(f"chat_{chat_id}")
    return {"success": True, "message": "对话记忆已清空"}


@router.get("/api/knowledge-bases")
def get_knowledge_bases_api(user: User = Depends(require_user), db: Session = Depends(get_db)):
    kbs = (
        db.query(KnowledgeBase)
        .options(subqueryload(KnowledgeBase.documents))
        .filter(KnowledgeBase.user_id == user.id)
        .order_by(KnowledgeBase.updated_at.desc())
        .all()
    )
    return [{"id": k.id, "name": k.name, "description": k.description, "document_count": len(k.documents)} for k in kbs]


@router.post("/api/retrieval-weights")
def set_retrieval_weights(
    vector_weight: float = Form(...),
    bm25_weight: float = Form(...),
    graph_weight: float = Form(0.3),
    user: User = Depends(require_user),
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    if vector_weight < 0 or bm25_weight < 0 or graph_weight < 0:
        raise HTTPException(status_code=400, detail="权重不能为负数")
    if vector_weight + bm25_weight + graph_weight <= 0:
        raise HTTPException(status_code=400, detail="权重之和必须大于 0")
    rag_model.vector_weight = vector_weight
    rag_model.bm25_weight = bm25_weight
    rag_model.graph_weight = graph_weight
    return {"vector_weight": vector_weight, "bm25_weight": bm25_weight, "graph_weight": graph_weight}


@router.post("/api/feedback")
def submit_feedback(
    message_id: int = Form(...),
    chat_id: int = Form(...),
    rating: str = Form(...),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    if rating not in ("up", "down"):
        return JSONResponse({"error": "rating 必须为 up 或 down"}, status_code=400)
    existing = (
        db.query(MessageFeedback)
        .filter(MessageFeedback.user_id == user.id, MessageFeedback.message_id == message_id)
        .first()
    )
    if existing:
        existing.rating = rating
    else:
        db.add(MessageFeedback(user_id=user.id, message_id=message_id, chat_id=chat_id, rating=rating))
    db.commit()

    # Track online evaluation metrics
    from web.online_eval import online_eval
    online_eval.record_feedback(rating, confidence_level="high")

    # Adaptive weight learning (thumbs up/down adjusts retrieval weights)
    from web.weight_adaptation import weight_adapter
    weight_adapter.update_from_feedback(["vector", "bm25", "graph"], rating)

    return {"success": True, "rating": rating}


@router.post("/api/intervention")
def create_intervention(
    chat_id: int = Form(...),
    query: str = Form(""),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """创建人工法律咨询介入请求"""
    chat_obj = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat_obj:
        return JSONResponse({"error": "对话不存在"}, status_code=404)

    # 取最近一条用户消息作为原始问题
    if not query:
        last_msg = (
            db.query(Message)
            .filter(Message.chat_id == chat_id, Message.role == "user")
            .order_by(Message.created_at.desc())
            .first()
        )
        query = last_msg.content if last_msg else ""

    intervention = InterventionRequest(
        user_id=user.id,
        chat_id=chat_id,
        original_query=query,
        confidence_level="low",
        confidence_score=0.0,
        confidence_reason="用户主动请求人工介入",
        status="pending",
    )
    db.add(intervention)
    db.commit()
    db.refresh(intervention)

    logger.info(f"人工介入请求已创建: user={user.id}, chat={chat_id}, id={intervention.id}")
    return {"success": True, "intervention_id": intervention.id}


@router.get("/api/interventions")
def list_interventions(user: User = Depends(require_user), db: Session = Depends(get_db)):
    """获取当前用户的介入请求列表"""
    interventions = (
        db.query(InterventionRequest)
        .filter(InterventionRequest.user_id == user.id)
        .order_by(InterventionRequest.created_at.desc())
        .limit(20)
        .all()
    )
    return [
        {
            "id": iv.id,
            "original_query": iv.original_query[:100],
            "status": iv.status,
            "response": iv.response,
            "created_at": iv.created_at.isoformat(),
        }
        for iv in interventions
    ]


@router.get("/api/admin/knowledge-gaps")
def list_knowledge_gaps(user: User = Depends(require_user), db: Session = Depends(get_db)):
    """管理员查看知识库缺口（按频率排序）"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    gaps = (
        db.query(KnowledgeGap)
        .filter(KnowledgeGap.status == "open")
        .order_by(KnowledgeGap.frequency.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "id": gap.id,
            "query": gap.query,
            "frequency": gap.frequency,
            "confidence_level": gap.confidence_level,
            "created_at": gap.created_at.isoformat(),
        }
        for gap in gaps
    ]


@router.put("/api/admin/knowledge-gaps/{gap_id}")
def update_knowledge_gap(
    gap_id: int,
    status: str = Form(...),
    notes: str = Form(""),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    """管理员更新知识库缺口状态"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    if status not in ("open", "researched", "added"):
        return JSONResponse({"error": "状态无效"}, status_code=400)
    gap = db.query(KnowledgeGap).filter(KnowledgeGap.id == gap_id).first()
    if not gap:
        return JSONResponse({"error": "缺口记录不存在"}, status_code=404)
    gap.status = status
    gap.notes = notes
    db.commit()
    return {"success": True}


@router.get("/api/admin/ab-tests")
def list_ab_tests(user: User = Depends(require_user)):
    """管理员查看 A/B 测试结果"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    from web.ab_test import ab_manager
    results = {}
    for name in ab_manager._experiments:
        results[name] = ab_manager.get_results(name)
    return results


@router.get("/api/admin/online-eval")
def get_online_eval(user: User = Depends(require_user)):
    """管理员查看线上满意度指标"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    from web.online_eval import online_eval
    return online_eval.get_all_metrics()


@router.get("/api/admin/adaptive-weights")
def get_adaptive_weights(user: User = Depends(require_user)):
    """管理员查看自适应学习后的检索权重"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    from web.weight_adaptation import weight_adapter
    return {"weights": weight_adapter.get_weights(), "update_count": weight_adapter._update_count}


@router.post("/api/admin/adaptive-weights/reset")
def reset_adaptive_weights(user: User = Depends(require_user)):
    """重置自适应权重为默认值"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    from web.weight_adaptation import weight_adapter
    with weight_adapter._lock:
        weight_adapter._weights = {"vector": 0.4, "bm25": 0.3, "graph": 0.3}
        weight_adapter._update_count = 0
    return {"success": True, "weights": weight_adapter.get_weights()}


# ══════════════════════════════════════════════════════════════════════
# Streaming chat
# ══════════════════════════════════════════════════════════════════════


@router.api_route("/ask_stream", methods=["GET", "POST"])
async def ask_stream(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    request_start = _time.time()

    if request.method == "GET":
        params = request.query_params
        user_input = params.get("user_input", "")
        chat_id = params.get("chat_id", "")
        kb_id = params.get("knowledge_base_id", "")
    else:
        form = await request.form()
        user_input = form.get("user_input", "")
        chat_id = form.get("chat_id", "")
        kb_id = form.get("knowledge_base_id", "")

    if not user_input or not chat_id:
        return JSONResponse({"error": "缺少必要参数"}, status_code=400)

    safe, reason = check_injection(user_input)
    if not safe:

        def _reject():
            yield f"data: {_json.dumps({'error': reason}, ensure_ascii=False)}\n\n"
            yield 'data: {"done": true}\n\n'

        return StreamingResponse(
            _reject(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    # Verify chat belongs to current user
    try:
        chat_id_int = int(chat_id)
    except (ValueError, TypeError):
        return JSONResponse({"error": "无效的对话ID"}, status_code=400)
    chat_obj = db.query(Chat).filter(Chat.id == chat_id_int, Chat.user_id == user.id).first()
    if not chat_obj:
        return JSONResponse({"error": "对话不存在或无权访问"}, status_code=404)

    conversation_id = f"chat_{chat_id}"

    # P13: 直接传递 knowledge_base_id 给 RAG 模型做 metadata 过滤，不再创建临时实例
    kb_id_int = _safe_int(kb_id) if user.role in ["expert", "admin"] else None
    result = rag_model.generate_response_stream(
        user_input, conversation_id=conversation_id, knowledge_base_id=kb_id_int, db_session=db
    )

    # 知识库缺口追踪（低/无置信度时记录）
    confidence = result.get("confidence", {})
    if confidence.get("level") in ("low", "none"):
        _track_knowledge_gap(db, user.id, user_input, confidence)

    # Save user message and update chat timestamp
    try:
        msg = Message(chat_id=int(chat_id), role="user", content=user_input)
        db.add(msg)
        chat_obj.updated_at = datetime.now(timezone.utc)
        db.commit()
    except Exception as e:
        logger.error(f"保存用户消息失败: {e}")
        db.rollback()

    def generate():
        full_response = ""
        try:
            for chunk in result["stream"]:
                content = chunk.content
                full_response += content
                yield f"data: {_json.dumps({'content': content}, ensure_ascii=False)}\n\n"

            rag_model.save_bot_response(conversation_id, full_response)

            bot_msg_id = None
            db2 = SessionLocal()
            try:
                bot_msg = Message(chat_id=int(chat_id), role="bot", content=full_response)
                db2.add(bot_msg)
                chat = db2.get(Chat, int(chat_id))
                if chat:
                    chat.updated_at = datetime.now(timezone.utc)
                db2.commit()
                db2.refresh(bot_msg)
                bot_msg_id = bot_msg.id
            finally:
                db2.close()

            conf_level = confidence.get("level", "high")
            show_banner = str(result.get("show_intervention_banner", False)).lower()
            done_payload = (
                f'{{"done": true, "message_id": {bot_msg_id}, "chat_id": {chat_id}, '
                f'"confidence_level": "{conf_level}", "show_intervention_banner": {show_banner}}}'
            )
            yield f"data: {done_payload}\n\n"

            # Metrics: track response time and confidence
            elapsed = _time.time() - request_start
            metrics.inc("ask_stream_total")
            metrics.observe("ask_stream_duration_seconds", elapsed)
            metrics.inc(f"confidence_{conf_level}")
        except Exception as e:
            logger.error(f"生成回复时出现错误: {e}", exc_info=True)
            error_msg = "抱歉，生成回复时出现错误，请稍后重试"
            rag_model.save_bot_response(conversation_id, error_msg)
            # Persist error message to DB so it survives page refresh
            db_err = SessionLocal()
            try:
                bot_msg = Message(chat_id=int(chat_id), role="bot", content=error_msg)
                db_err.add(bot_msg)
                chat = db_err.get(Chat, int(chat_id))
                if chat:
                    chat.updated_at = datetime.now(timezone.utc)
                db_err.commit()
                db_err.refresh(bot_msg)
                yield f'data: {{"done": true, "message_id": {bot_msg.id}, "chat_id": {chat_id}}}\n\n'
            except Exception:
                db_err.rollback()
                yield f"data: {_json.dumps({'error': error_msg}, ensure_ascii=False)}\n\n"
                yield 'data: {"done": true}\n\n'
            finally:
                db_err.close()

    return StreamingResponse(
        generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


# ══════════════════════════════════════════════════════════════════════
# Health / Metrics
# ══════════════════════════════════════════════════════════════════════


@router.get("/health")
def health_check():
    from law_assistant.redis_utils import is_available
    from sqlalchemy import text as _text

    redis_ok = is_available()
    db_ok = False
    try:
        db = SessionLocal()
        db.execute(_text("SELECT 1"))
        db_ok = True
        db.close()
    except Exception:
        pass
    status = "healthy" if (db_ok and rag_model is not None) else "degraded"
    return {"status": status, "database": db_ok, "redis": redis_ok, "rag_ready": rag_model is not None}


@router.get("/metrics")
def metrics_endpoint():
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(metrics.render(), media_type="text/plain")
