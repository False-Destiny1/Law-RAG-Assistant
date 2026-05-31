import hashlib
import hmac
import json as _json
import logging
import os
import re
import secrets
import urllib.parse
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy import text as _text
from sqlalchemy.orm import Session, joinedload, sessionmaker, subqueryload

from law_assistant.models import (
    Base, Chat, InterventionRequest, KnowledgeBase, KnowledgeGap,
    Message, MessageFeedback, UploadedDocument, User,
)
from law_assistant.rag import DeepSeekApiRag
from law_assistant.security import check_injection

load_dotenv()

# 配置结构化日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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


# ── App ──────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app_instance):
    from law_assistant.redis_utils import close_pool, is_available

    if is_available():
        logger.info("Redis 连接成功")
    else:
        logger.warning("Redis 不可用，使用本地缓存和数据库回退")
    yield
    close_pool()


app = FastAPI(title="智能法律助手", lifespan=lifespan)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        "font-src 'self' https://cdn.jsdelivr.net; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    )
    return response


def _generate_csrf_token(session_token: str = "") -> str:
    """Generate a CSRF token from session + secret."""
    payload = f"{session_token}:{SESSION_SECRET}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


@app.middleware("http")
async def csrf_middleware(request: Request, call_next):
    """Double-submit cookie CSRF protection."""
    # Always compute and store CSRF token for template context
    session_token = request.cookies.get("session_token", "")
    csrf_token = _generate_csrf_token(session_token)
    request.state.csrf_token = csrf_token

    # Skip safe methods and API/static endpoints
    if request.method in ("GET", "HEAD", "OPTIONS"):
        response = await call_next(request)
        # Always set CSRF cookie to ensure it matches request.state.csrf_token
        response.set_cookie("csrf_token", csrf_token, httponly=False, samesite="lax")
        return response

    # For state-changing methods, validate CSRF token
    # Skip CSRF for /ask_stream (SSE endpoint, protected by session cookie + samesite)
    if request.url.path not in ("/ask_stream",):
        cookie_token = request.cookies.get("csrf_token", "")
        if not cookie_token:
            return JSONResponse({"error": "CSRF token missing"}, status_code=403)

        # Read token from header first (works for JSON, DELETE, and multipart via JS)
        form_token = request.headers.get("X-CSRF-Token", "")

        # Fallback: parse csrf_token from form body
        if not form_token:
            content_type = request.headers.get("content-type", "")
            try:
                if "application/x-www-form-urlencoded" in content_type:
                    body = await request.body()
                    parsed = urllib.parse.parse_qs(body.decode("utf-8", errors="ignore"))
                    tokens = parsed.get("csrf_token", [])
                    form_token = tokens[0] if tokens else ""
                elif "multipart/form-data" in content_type:
                    # Don't parse body (consumes it, breaking File() uploads).
                    # samesite=lax cookie already prevents cross-site; just use cookie token.
                    if cookie_token:
                        form_token = cookie_token
            except Exception as e:
                logger.debug(f"CSRF token 解析失败: {e}")

        if not form_token or not hmac.compare_digest(cookie_token, form_token):
            # For form submissions, redirect back instead of returning JSON
            content_type = request.headers.get("content-type", "")
            if "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
                referer = request.headers.get("referer", "/")
                return RedirectResponse(url=referer, status_code=303)
            return JSONResponse({"error": "CSRF token invalid"}, status_code=403)

    return await call_next(request)


def _get_rate_limit_id(request: Request) -> str:
    """从 session cookie 提取 user_id（验证签名），无 cookie 时 fallback 到 IP"""
    token = request.cookies.get("session_token")
    if token:
        try:
            parts = token.split(":")
            if len(parts) == 4:
                user_id_str, nonce, timestamp_str, sig = parts
                expected = hmac.new(
                    SESSION_SECRET.encode(), f"{user_id_str}:{nonce}:{timestamp_str}".encode(), hashlib.sha256
                ).hexdigest()[:32]
                if hmac.compare_digest(sig, expected):
                    return f"user:{user_id_str}"
            elif len(parts) == 3:
                user_id_str, nonce, sig = parts
                expected = hmac.new(
                    SESSION_SECRET.encode(), f"{user_id_str}:{nonce}".encode(), hashlib.sha256
                ).hexdigest()[:32]
                if hmac.compare_digest(sig, expected):
                    return f"user:{user_id_str}"
        except Exception as e:
            logger.debug(f"Session token 解析失败: {e}")
    # Only trust X-Forwarded-For if TRUSTED_PROXY is configured
    if os.getenv("TRUSTED_PROXY"):
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
            return f"ip:{ip}"
    ip = request.client.host if request.client else "unknown"
    return f"ip:{ip}"


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/static"):
        return await call_next(request)
    try:
        from law_assistant.redis_utils import rate_limit_check
    except ImportError:
        # redis_utils 模块缺失（极端情况），跳过限流
        return await call_next(request)

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
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response


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
    logger.warning("SESSION_SECRET not set, generated a random one (saved to .env)")
SESSION_EXPIRE_HOURS = 24

# ── Database ─────────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 环境变量未设置，请在 .env 中配置数据库连接")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

Base.metadata.create_all(engine)
# 确保索引存在（create_all 不会更新已存在的表）

_index_sqls = [
    "CREATE INDEX IF NOT EXISTS ix_chat_user_updated ON chat (user_id, updated_at)",
    "CREATE INDEX IF NOT EXISTS ix_message_chat ON message (chat_id)",
    "CREATE INDEX IF NOT EXISTS ix_kb_user ON knowledge_base (user_id)",
    "CREATE INDEX IF NOT EXISTS ix_doc_user_kb ON uploaded_document (user_id, knowledge_base_id)",
    "ALTER TABLE uploaded_document ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'completed'",
    "CREATE INDEX IF NOT EXISTS ix_intervention_user ON intervention_request (user_id, status)",
    "CREATE INDEX IF NOT EXISTS ix_intervention_status ON intervention_request (status, created_at)",
    "CREATE INDEX IF NOT EXISTS ix_gap_status ON knowledge_gap (status, frequency)",
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
    timestamp = str(int(datetime.now(timezone.utc).timestamp()))
    payload = f"{user_id}:{secrets.token_hex(16)}:{timestamp}"
    signature = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    return f"{payload}:{signature}"


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get("session_token")
    if not token:
        return None
    # Check token blacklist (logout invalidation)
    try:
        from law_assistant.redis_utils import is_token_blacklisted

        if is_token_blacklisted(token):
            return None
    except Exception:
        pass
    try:
        parts = token.split(":")
        if len(parts) != 4:
            return None
        else:
            user_id_str, nonce, timestamp_str, sig = parts
            expected = hmac.new(
                SESSION_SECRET.encode(), f"{user_id_str}:{nonce}:{timestamp_str}".encode(), hashlib.sha256
            ).hexdigest()[:32]
            if not hmac.compare_digest(sig, expected):
                return None
            # Server-side expiry check
            token_age_hours = (datetime.now(timezone.utc).timestamp() - int(timestamp_str)) / 3600
            if token_age_hours > SESSION_EXPIRE_HOURS:
                return None

        user_id = int(user_id_str)

        # Try Redis cache first
        try:
            from law_assistant.redis_utils import cache_get_json

            cached = cache_get_json(f"user:{user_id}")
            if cached:
                user = User(
                    id=cached["id"],
                    phone=cached["phone"],
                    username=cached["username"],
                    role=cached["role"],
                )
                return user
        except Exception as e:
            logger.debug(f"Redis 用户缓存读取失败: {e}")

        # Fallback: DB query
        user = db.get(User, user_id)
        if user:
            try:
                from law_assistant.redis_utils import cache_set_json

                cache_set_json(
                    f"user:{user_id}",
                    {
                        "id": user.id,
                        "phone": user.phone,
                        "username": user.username,
                        "role": user.role,
                    },
                    ttl=300,
                )
            except Exception as e:
                logger.debug(f"Redis 用户缓存写入失败: {e}")
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
    # 注入 ORM 依赖（避免 law_assistant 包循环导入 app.py）
    rag_model.set_knowledge_base_model(KnowledgeBase)
    rag_model.set_memory_db_factory(SessionLocal, Message)

    need_faiss = not os.path.exists(db_path)
    need_bm25 = not os.path.exists("bm25_index.pkl")

    if need_faiss or need_bm25:
        # 收集所有文档文本（只处理一次）
        all_texts = []
        if os.path.exists(KNOWLEDGE_BASE_FOLDER) and os.listdir(KNOWLEDGE_BASE_FOLDER):
            logger.info(f"正在处理 knowledge_base 文件夹: {KNOWLEDGE_BASE_FOLDER}")
            for filename in os.listdir(KNOWLEDGE_BASE_FOLDER):
                file_path = os.path.join(KNOWLEDGE_BASE_FOLDER, filename)
                if os.path.isfile(file_path):
                    try:
                        pages = rag_model.document_processor._load_documents(file_path)
                        documents = rag_model.general_splitter.split_documents(pages)
                        all_texts.extend([d.page_content for d in documents])
                    except Exception as e:
                        logger.warning(f"处理文档 {filename} 失败: {e}")

        db = SessionLocal()
        try:
            for doc in db.query(UploadedDocument).all():
                if os.path.exists(doc.file_path):
                    try:
                        pages = rag_model.document_processor._load_documents(doc.file_path)
                        documents = rag_model.general_splitter.split_documents(pages)
                        all_texts.extend([d.page_content for d in documents])
                    except Exception as e:
                        logger.warning(f"处理文档 {doc.filename} 失败: {e}")
        finally:
            db.close()

        if all_texts:
            if need_faiss:
                logger.info("向量数据库不存在，开始构建...")
                rag_model.add_documents(all_texts, save_to_disk=True)
                logger.info("向量数据库构建完成")
            if need_bm25:
                logger.info("BM25 索引不存在，开始构建...")
                rag_model.bm25_retriever.build_index(all_texts)
                rag_model.bm25_retriever.save_index()
                logger.info(f"BM25 索引构建完成，文档数量: {len(all_texts)}")
        else:
            logger.info("无文档可构建索引")
    else:
        logger.info("向量数据库和 BM25 索引均已存在，跳过初始化构建")
        # 从 BM25 已有文档填充文档注册表（确保删除后重建一致性）
        if rag_model.bm25_retriever.documents:
            with rag_model._registry_lock:
                rag_model._document_registry = list(rag_model.bm25_retriever.documents)
            logger.info(f"已从 BM25 加载 {len(rag_model._document_registry)} 个文档到注册表")


initialize_vector_database()


def ensure_admin_exists():
    """启动时确保默认管理员账号存在"""
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == "admin").first()
        if not admin:
            admin_password = os.getenv("ADMIN_PASSWORD")
            if not admin_password:
                admin_password = secrets.token_urlsafe(16)
                logger.warning(f"ADMIN_PASSWORD 未设置，已生成随机密码: {admin_password}")
                _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
                if os.path.exists(_env_path):
                    with open(_env_path, "a", encoding="utf-8") as _f:
                        _f.write(f"\nADMIN_PASSWORD={admin_password}\n")
            admin_user = User(phone="admin", username="管理员", role="admin")
            admin_user.set_password(admin_password)
            db.add(admin_user)
            db.commit()
            logger.info("已创建默认管理员账号 (密码已写入 .env)")
        else:
            logger.info(f"管理员账号已存在: {admin.phone}")
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
    remember: str | None = Form(None),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.phone == identifier).first()
    if not user or not user.check_password(password):
        return templates.TemplateResponse(request, "login.html", {"error": "手机号或密码错误"})

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


@app.post("/logout")
def logout(request: Request):
    token = request.cookies.get("session_token", "")
    if token:
        try:
            from law_assistant.redis_utils import blacklist_token

            blacklist_token(token, ttl_seconds=SESSION_EXPIRE_HOURS * 3600)
        except Exception as e:
            logger.warning(f"Token 黑名单写入失败: {e}")
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("session_token")
    response.delete_cookie("csrf_token")
    return response


@app.post("/api/change-password")
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


# ── Page routes ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def home(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse(request, "index.html", {"user": user})


@app.get("/knowledge-bases", response_class=HTMLResponse)
def knowledge_bases_page(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    kb_list = (
        db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user.id).order_by(KnowledgeBase.updated_at.desc()).all()
    )
    return templates.TemplateResponse(request, "knowledge_base.html", {"user": user, "knowledge_bases": kb_list})


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
    return templates.TemplateResponse(request, "edit_knowledge_base.html", {"user": user, "kb": kb})


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


@app.delete("/knowledge-base/{kb_id}")
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


@app.get("/upload", response_class=HTMLResponse)
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


def _process_document_async(file_path: str, doc_id: int, knowledge_base_id: int = None):
    """后台异步处理文档索引（FAISS + BM25 + 知识图谱）"""
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


@app.post("/upload")
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


@app.delete("/upload/{doc_id}")
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


@app.post("/upload/{doc_id}/reprocess")
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
        db.rollback()


# ── API routes ───────────────────────────────────────────────────────
@app.get("/api/chats")
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


@app.get("/api/chats/{chat_id}")
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


@app.post("/api/chats")
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


@app.put("/api/chats/{chat_id}")
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


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    rag_model.clear_conversation_memory(f"chat_{chat_id}")
    db.delete(chat)
    db.commit()
    return {"success": True}


@app.get("/api/chats/{chat_id}/export")
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


@app.post("/api/chats/{chat_id}/clear_memory")
def clear_chat_memory(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.query(Chat).filter(Chat.id == chat_id, Chat.user_id == user.id).first()
    if not chat:
        return JSONResponse({"error": "对话不存在"}, status_code=404)
    rag_model.clear_conversation_memory(f"chat_{chat_id}")
    return {"success": True, "message": "对话记忆已清空"}


@app.get("/api/knowledge-bases")
def get_knowledge_bases_api(user: User = Depends(require_user), db: Session = Depends(get_db)):
    kbs = (
        db.query(KnowledgeBase)
        .options(subqueryload(KnowledgeBase.documents))
        .filter(KnowledgeBase.user_id == user.id)
        .order_by(KnowledgeBase.updated_at.desc())
        .all()
    )
    return [{"id": k.id, "name": k.name, "description": k.description, "document_count": len(k.documents)} for k in kbs]


@app.post("/api/retrieval-weights")
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


@app.post("/api/feedback")
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
    from law_assistant.online_eval import online_eval
    online_eval.record_feedback(rating, confidence_level="high")

    # Adaptive weight learning (thumbs up/down adjusts retrieval weights)
    from law_assistant.weight_adaptation import weight_adapter
    weight_adapter.update_from_feedback(["vector", "bm25", "graph"], rating)

    return {"success": True, "rating": rating}


@app.get("/api/admin/ab-tests")
def list_ab_tests(user: User = Depends(require_user)):
    """管理员查看 A/B 测试结果"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    from law_assistant.ab_test import ab_manager
    results = {}
    for name in ab_manager._experiments:
        results[name] = ab_manager.get_results(name)
    return results


@app.get("/api/admin/online-eval")
def get_online_eval(user: User = Depends(require_user)):
    """管理员查看线上满意度指标"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    from law_assistant.online_eval import online_eval
    return online_eval.get_all_metrics()


@app.get("/api/admin/adaptive-weights")
def get_adaptive_weights(user: User = Depends(require_user)):
    """管理员查看自适应学习后的检索权重"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    from law_assistant.weight_adaptation import weight_adapter
    return {"weights": weight_adapter.get_weights(), "update_count": weight_adapter._update_count}


@app.post("/api/admin/adaptive-weights/reset")
def reset_adaptive_weights(user: User = Depends(require_user)):
    """重置自适应权重为默认值"""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    from law_assistant.weight_adaptation import weight_adapter
    with weight_adapter._lock:
        weight_adapter._weights = {"vector": 0.4, "bm25": 0.3, "graph": 0.3}
        weight_adapter._update_count = 0
    return {"success": True, "weights": weight_adapter.get_weights()}


@app.post("/api/intervention")
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


@app.get("/api/interventions")
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


@app.get("/api/admin/knowledge-gaps")
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


@app.put("/api/admin/knowledge-gaps/{gap_id}")
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


# ── Streaming chat ──────────────────────────────────────────────────
@app.api_route("/ask_stream", methods=["GET", "POST"])
async def ask_stream(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    from law_assistant.metrics import metrics
    import time as _time
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


@app.get("/health")
def health_check():
    from law_assistant.redis_utils import is_available

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


@app.get("/metrics")
def metrics_endpoint():
    from law_assistant.metrics import metrics
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(metrics.render(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
