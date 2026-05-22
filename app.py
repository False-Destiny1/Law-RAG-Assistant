import os
import uuid
import secrets
import hmac
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import FastAPI, Request, Form, UploadFile, File, Depends, HTTPException, Response, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session, joinedload, subqueryload
import bcrypt
from dotenv import load_dotenv

from model_utils import DeepSeekApiRag

load_dotenv()

import json as _json


def _safe_int(value, default=None):
    """Safely convert to int, returning default on failure."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(title="智能法律助手")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def _get_rate_limit_id(request: Request) -> str:
    """从 session cookie 提取 user_id，无 cookie 时 fallback 到 IP"""
    token = request.cookies.get("session_token")
    if token:
        try:
            parts = token.split(":")
            if len(parts) >= 2:
                return f"user:{parts[0]}"
        except Exception:
            pass
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    return f"ip:{ip}"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/static"):
        return await call_next(request)
    try:
        from redis_utils import rate_limit_check
        identifier = _get_rate_limit_id(request)
        path = request.url.path
        if path.startswith("/ask_stream"):
            limit = 30
        elif path.startswith("/api/"):
            limit = 60
        elif path in ("/login", "/register") and request.method == "POST":
            limit = 10
        else:
            limit = 120
        allowed, count, remaining = rate_limit_check(identifier, limit, window=60)
        if not allowed:
            return JSONResponse(
                {"error": "请求过于频繁，请稍后再试"},
                status_code=429,
                headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"}
            )
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
    except Exception:
        return await call_next(request)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ── Security ─────────────────────────────────────────────────────────
SESSION_SECRET = os.getenv("SESSION_SECRET")
if not SESSION_SECRET:
    SESSION_SECRET = secrets.token_hex(32)
    # Persist to .env so it survives restarts
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path, "a", encoding="utf-8") as _f:
            _f.write(f"\nSESSION_SECRET={SESSION_SECRET}\n")
    print("WARNING: SESSION_SECRET not set, generated a random one (saved to .env)")
SESSION_EXPIRE_HOURS = 24

# ── Database ─────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///user.db")
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True, autoincrement=True)
    phone = Column(String(20), unique=True, nullable=False)
    username = Column(String(50), nullable=False)
    password_hash = Column(String(128), nullable=False)
    role = Column(String(20), nullable=False, default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    chats = relationship("Chat", backref="user", cascade="all, delete-orphan")
    knowledge_bases = relationship("KnowledgeBase", backref="user", cascade="all, delete-orphan")
    uploaded_documents = relationship("UploadedDocument", backref="user", cascade="all, delete-orphan")

    def set_password(self, password: str):
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    documents = relationship("UploadedDocument", backref="knowledge_base", cascade="all, delete-orphan")
    chats = relationship("Chat", backref="knowledge_base")


class Chat(Base):
    __tablename__ = "chat"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    title = Column(String(100), nullable=False, default="新对话")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_base.id"), nullable=True)
    messages = relationship("Message", backref="chat", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "message"
    id = Column(Integer, primary_key=True, autoincrement=True)
    chat_id = Column(Integer, ForeignKey("chat.id"), nullable=False)
    role = Column(String(10), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class UploadedDocument(Base):
    __tablename__ = "uploaded_document"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_base.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# 数据库索引（加速查询）
Index("ix_chat_user_updated", Chat.user_id, Chat.updated_at)
Index("ix_message_chat", Message.chat_id)
Index("ix_kb_user", KnowledgeBase.user_id)
Index("ix_doc_user_kb", UploadedDocument.user_id, UploadedDocument.knowledge_base_id)


Base.metadata.create_all(engine)
# 确保索引存在（create_all 不会更新已存在的表）
from sqlalchemy import text as _text
_index_sqls = [
    'CREATE INDEX IF NOT EXISTS ix_chat_user_updated ON chat (user_id, updated_at)',
    'CREATE INDEX IF NOT EXISTS ix_message_chat ON message (chat_id)',
    'CREATE INDEX IF NOT EXISTS ix_kb_user ON knowledge_base (user_id)',
    'CREATE INDEX IF NOT EXISTS ix_doc_user_kb ON uploaded_document (user_id, knowledge_base_id)',
]
with engine.connect() as _conn:
    for _sql in _index_sqls:
        _conn.execute(_text(_sql))
    _conn.commit()


# ── Session helpers ──────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_session_token(user_id: int) -> str:
    payload = f"{user_id}:{secrets.token_hex(16)}"
    signature = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}:{signature}"


def get_current_user(request: Request, db: Session = Depends(get_db)) -> Optional[User]:
    token = request.cookies.get("session_token")
    if not token:
        return None
    try:
        parts = token.split(":")
        if len(parts) != 3:
            return None
        user_id_str, nonce, sig = parts
        expected = hmac.new(SESSION_SECRET.encode(), f"{user_id_str}:{nonce}".encode(), hashlib.sha256).hexdigest()[:32]
        if not hmac.compare_digest(sig, expected):
            return None

        user_id = int(user_id_str)

        # Try Redis cache first
        try:
            from redis_utils import cache_get_json
            cached = cache_get_json(f"user:{user_id}")
            if cached:
                user = User(
                    id=cached["id"],
                    phone=cached["phone"],
                    username=cached["username"],
                    role=cached["role"],
                )
                return user
        except Exception:
            pass

        # Fallback: DB query
        user = db.get(User, user_id)
        if user:
            try:
                from redis_utils import cache_set_json
                cache_set_json(f"user:{user_id}", {
                    "id": user.id,
                    "phone": user.phone,
                    "username": user.username,
                    "role": user.role,
                }, ttl=3600)
            except Exception:
                pass
        return user
    except (ValueError, AttributeError):
        return None


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@app.exception_handler(401)
async def auth_redirect_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/") or request.url.path.startswith("/ask_stream"):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return RedirectResponse(url="/login", status_code=303)


# ── RAG Model ────────────────────────────────────────────────────────
api_key = os.getenv("MIMO_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")
rag_model = None

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
KNOWLEDGE_BASE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(KNOWLEDGE_BASE_FOLDER, exist_ok=True)


def initialize_vector_database():
    global rag_model
    rag_model = DeepSeekApiRag(api_key, db_path)

    if not os.path.exists(db_path):
        print("向量数据库不存在，开始构建...")
        if os.path.exists(KNOWLEDGE_BASE_FOLDER) and os.listdir(KNOWLEDGE_BASE_FOLDER):
            print(f"正在处理 knowledge_base 文件夹: {KNOWLEDGE_BASE_FOLDER}")
            rag_model.add_folder_documents(KNOWLEDGE_BASE_FOLDER)

        db = SessionLocal()
        try:
            all_docs = db.query(UploadedDocument).all()
            if all_docs:
                all_texts = []
                for doc in all_docs:
                    if os.path.exists(doc.file_path):
                        try:
                            # 统一通过 DocumentProcessor 加载（支持 OCR 回退）
                            pages = rag_model.document_processor._load_documents(doc.file_path)
                            documents = rag_model.text_splitter.split_documents(pages)
                            all_texts.extend([d.page_content for d in documents])
                        except Exception as e:
                            print(f"处理文档 {doc.filename} 失败: {e}")
                if all_texts:
                    rag_model.add_documents(all_texts)
        finally:
            db.close()

        print(f"向量数据库构建完成")
    else:
        print("向量数据库已存在，跳过初始化构建")


initialize_vector_database()


def ensure_admin_exists():
    """启动时确保默认管理员账号存在"""
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == "admin").first()
        if not admin:
            admin_user = User(phone="admin", username="管理员", role="admin")
            admin_user.set_password("admin123")
            db.add(admin_user)
            db.commit()
            print("已创建默认管理员账号: admin / admin123")
        else:
            print(f"管理员账号已存在: {admin.phone}")
    finally:
        db.close()


ensure_admin_exists()


# ── Auth routes ──────────────────────────────────────────────────────
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, registered: str = None):
    context = {}
    if registered:
        context["success"] = "注册成功，请登录"
    return templates.TemplateResponse(request, "login.html", context)


@app.post("/login")
def login_submit(
    request: Request,
    identifier: str = Form(...),
    password: str = Form(...),
    remember: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.phone == identifier).first()
    if not user or not user.check_password(password):
        return templates.TemplateResponse(request, "login.html", {
            "error": "手机号或密码错误"
        })

    token = create_session_token(user.id)
    response = RedirectResponse(url="/", status_code=303)
    max_age = SESSION_EXPIRE_HOURS * 3600 if remember else None
    is_production = os.getenv("ENV", "").lower() == "production"
    response.set_cookie("session_token", token, httponly=True, max_age=max_age, samesite="lax", secure=is_production)
    return response


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html")


@app.post("/register")
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
    if len(password) < 6:
        errors.append("密码长度至少为6位")
    if password != confirm_password:
        errors.append("两次输入的密码不一致")
    if db.query(User).filter(User.phone == phone).first():
        errors.append("该手机号已注册")
    if db.query(User).filter(User.username == username).first():
        errors.append("该用户名已存在")

    if errors:
        return templates.TemplateResponse(request, "register.html", {
            "errors": errors
        })

    new_user = User(phone=phone, username=username, role="user")
    new_user.set_password(password)
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login?registered=1", status_code=303)


@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token")
    return response


# ── Page routes ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse(request, "index.html", {"user": user})


@app.get("/knowledge-bases", response_class=HTMLResponse)
def knowledge_bases_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    kb_list = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user.id).order_by(KnowledgeBase.updated_at.desc()).all()
    return templates.TemplateResponse(request, "knowledge_base.html", {
        "user": user, "knowledge_bases": kb_list
    })


@app.get("/knowledge-base/create", response_class=HTMLResponse)
def create_kb_page(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse(request, "create_knowledge_base.html", {"user": user})


@app.post("/knowledge-base/create")
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


@app.get("/knowledge-base/{kb_id}/edit", response_class=HTMLResponse)
def edit_kb_page(kb_id: int, request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user.id).first()
    if not kb:
        return RedirectResponse(url="/knowledge-bases", status_code=303)
    return templates.TemplateResponse(request, "edit_knowledge_base.html", {
        "user": user, "kb": kb
    })


@app.post("/knowledge-base/{kb_id}/edit")
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


@app.post("/knowledge-base/{kb_id}/delete")
def delete_kb(kb_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id, KnowledgeBase.user_id == user.id).first()
    if kb:
        for doc in kb.documents:
            if os.path.exists(doc.file_path):
                try:
                    os.remove(doc.file_path)
                except Exception:
                    pass
        db.delete(kb)
        db.commit()
    return RedirectResponse(url="/knowledge-bases", status_code=303)


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    if user.role not in ["expert", "admin"]:
        return RedirectResponse(url="/", status_code=303)
    kbs = db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user.id).order_by(KnowledgeBase.name).all()
    docs = db.query(UploadedDocument).filter(UploadedDocument.user_id == user.id).order_by(UploadedDocument.uploaded_at.desc()).all()
    return templates.TemplateResponse(request, "upload.html", {
        "user": user, "knowledge_bases": kbs, "uploaded_docs": docs
    })


def _process_document_async(file_path: str, doc_id: int, knowledge_base_id: int = None):
    """后台异步处理文档索引"""
    try:
        print(f"[异步] 开始处理文档索引: {file_path}")
        rag_model.add_file_documents(file_path)
        rag_model.invalidate_kb_cache(knowledge_base_id)
        print(f"[异步] 文档索引完成: {file_path}")
    except Exception as e:
        print(f"[异步] 文档索引失败: {file_path}, 错误: {e}")


@app.post("/upload")
async def upload_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    knowledge_base_id: Optional[str] = Form(None),
    user: User = Depends(require_user),
    db: Session = Depends(get_db),
):
    if user.role not in ["expert", "admin"]:
        return RedirectResponse(url="/", status_code=303)

    allowed_extensions = {"pdf", "docx", "txt", "jpg", "jpeg", "png", "bmp", "tiff"}
    file_ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if file_ext not in allowed_extensions:
        return RedirectResponse(url="/upload", status_code=303)

    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        return JSONResponse({"error": "文件大小超过50MB限制"}, status_code=413)
    with open(file_path, "wb") as f:
        f.write(content)

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


@app.post("/upload/{doc_id}/delete")
def delete_document(doc_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    doc = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id, UploadedDocument.user_id == user.id).first()
    if doc:
        # P1: 删除前先从索引中移除该文档的内容
        try:
            if os.path.exists(doc.file_path):
                from DocumentProcessor import DocumentProcessor
                processor = DocumentProcessor()
                chunks = processor.process_document(doc.file_path)
                texts_to_remove = [c['full_text'] for c in chunks]
                if texts_to_remove:
                    rag_model.bm25_retriever.remove_documents(texts_to_remove)
                    rag_model.bm25_retriever.save_index()
                    rag_model.invalidate_kb_cache(doc.knowledge_base_id)
                    print(f"已从BM25索引中移除文档 {doc.filename} 的 {len(texts_to_remove)} 个文本块")
        except Exception as e:
            print(f"从索引中移除文档失败: {e}")

        if os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except Exception:
                pass
        db.delete(doc)
        db.commit()
    return RedirectResponse(url="/upload", status_code=303)


# ── API routes ───────────────────────────────────────────────────────
@app.get("/api/chats")
def get_chats(user: User = Depends(require_user), db: Session = Depends(get_db)):
    chats = db.query(Chat).options(joinedload(Chat.knowledge_base)).filter(Chat.user_id == user.id).order_by(Chat.updated_at.desc()).all()
    return [{
        "id": c.id, "title": c.title,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
        "knowledge_base_id": c.knowledge_base_id,
        "knowledge_base_name": c.knowledge_base.name if c.knowledge_base else None,
    } for c in chats]


@app.get("/api/chats/{chat_id}")
def get_chat_messages(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).options(joinedload(Chat.knowledge_base)).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.created_at).all()
    return {
        "id": chat.id, "title": chat.title,
        "knowledge_base_id": chat.knowledge_base_id,
        "knowledge_base_name": chat.knowledge_base.name if chat.knowledge_base else None,
        "messages": [{"id": m.id, "role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in messages],
    }


@app.post("/api/chats")
def create_chat(user: User = Depends(require_user), db: Session = Depends(get_db)):
    new_chat = Chat(user_id=user.id, title="新对话")
    db.add(new_chat)
    db.flush()

    initial_msg = Message(chat_id=new_chat.id, role="bot", content="您好！我是智能法律助手，请问有什么可以帮您的吗？")
    db.add(initial_msg)
    db.commit()
    db.refresh(new_chat)

    return {"id": new_chat.id, "title": new_chat.title, "created_at": new_chat.created_at.isoformat()}


@app.put("/api/chats/{chat_id}")
async def update_chat(chat_id: int, request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
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

    return {"id": chat.id, "title": chat.title, "knowledge_base_id": chat.knowledge_base_id}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    rag_model.clear_conversation_memory(f"chat_{chat_id}")
    db.delete(chat)
    db.commit()
    return {"success": True}


@app.post("/api/chats/{chat_id}/clear_memory")
def clear_chat_memory(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    rag_model.clear_conversation_memory(f"chat_{chat_id}")
    return {"success": True, "message": "对话记忆已清空"}


@app.get("/api/knowledge-bases")
def get_knowledge_bases_api(user: User = Depends(require_user), db: Session = Depends(get_db)):
    kbs = db.query(KnowledgeBase).options(subqueryload(KnowledgeBase.documents)).filter(KnowledgeBase.user_id == user.id).order_by(KnowledgeBase.updated_at.desc()).all()
    return [{"id": k.id, "name": k.name, "description": k.description, "document_count": len(k.documents)} for k in kbs]


@app.post("/api/retrieval-weights")
def set_retrieval_weights(
    vector_weight: float = Form(...),
    bm25_weight: float = Form(...),
    user: User = Depends(require_user),
):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    if vector_weight + bm25_weight <= 0:
        raise HTTPException(status_code=400, detail="权重之和必须大于 0")
    rag_model.vector_weight = vector_weight
    rag_model.bm25_weight = bm25_weight
    return {"vector_weight": vector_weight, "bm25_weight": bm25_weight}


# ── Streaming chat ──────────────────────────────────────────────────
@app.api_route("/ask_stream", methods=["GET", "POST"])
async def ask_stream(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
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
        user_input,
        conversation_id=conversation_id,
        knowledge_base_id=kb_id_int,
        db_session=db
    )

    # Save user message
    try:
        msg = Message(chat_id=int(chat_id), role="user", content=user_input)
        db.add(msg)
        db.commit()
    except Exception:
        db.rollback()

    def generate():
        full_response = ""
        try:
            for chunk in result["stream"]:
                content = chunk.content
                full_response += content
                yield f'data: {_json.dumps({"content": content}, ensure_ascii=False)}\n\n'

            rag_model.save_bot_response(conversation_id, full_response)

            db2 = SessionLocal()
            try:
                bot_msg = Message(chat_id=int(chat_id), role="bot", content=full_response)
                db2.add(bot_msg)
                chat = db2.get(Chat, int(chat_id))
                if chat:
                    chat.updated_at = datetime.now(timezone.utc)
                db2.commit()
            finally:
                db2.close()

            yield 'data: {"done": true}\n\n'
        except Exception as e:
            error_msg = f"抱歉，生成回复时出现错误: {str(e)}"
            rag_model.save_bot_response(conversation_id, error_msg)
            yield f'data: {_json.dumps({"error": error_msg}, ensure_ascii=False)}\n\n'
            yield 'data: {"done": true}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.on_event("startup")
async def startup_event():
    from redis_utils import is_available
    if is_available():
        print("Redis 连接成功")
    else:
        print("WARNING: Redis 不可用，使用本地缓存和数据库回退")


@app.on_event("shutdown")
async def shutdown_event():
    from redis_utils import close_pool
    close_pool()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
