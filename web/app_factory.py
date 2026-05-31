"""FastAPI application factory — creates and configures the app."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Module-level state shared with routes
rag_model = None
templates = None


def _init_rag():
    """Initialize the RAG model and wire up dependencies."""
    global rag_model
    from law_assistant.rag import DeepSeekApiRag
    from web.db import SessionLocal
    from web.models import KnowledgeBase, Message

    api_key = os.getenv("MIMO_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    db_path = os.getenv("VECTOR_DB_PATH", "law_faiss")
    rag_model = DeepSeekApiRag(api_key, db_path)
    rag_model.set_knowledge_base_model(KnowledgeBase)
    rag_model.set_memory_db_factory(SessionLocal, Message)

    # Build indexes if needed
    need_faiss = not os.path.exists(db_path)
    need_bm25 = not os.path.exists("bm25_index.pkl")

    if need_faiss or need_bm25:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        knowledge_base_folder = os.path.join(project_root, "knowledge_base")
        all_texts = []
        if os.path.exists(knowledge_base_folder) and os.listdir(knowledge_base_folder):
            logger.info(f"正在处理 knowledge_base 文件夹: {knowledge_base_folder}")
            for filename in os.listdir(knowledge_base_folder):
                file_path = os.path.join(knowledge_base_folder, filename)
                if os.path.isfile(file_path):
                    try:
                        pages = rag_model.document_processor._load_documents(file_path)
                        documents = rag_model.general_splitter.split_documents(pages)
                        all_texts.extend([d.page_content for d in documents])
                    except Exception as e:
                        logger.warning(f"处理文档 {filename} 失败: {e}")

        db = SessionLocal()
        try:
            from web.models import UploadedDocument
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
            if need_bm25:
                logger.info("BM25 索引不存在，开始构建...")
                rag_model.bm25_retriever.build_index(all_texts)
                rag_model.bm25_retriever.save_index()
        else:
            logger.info("无文档可构建索引")
    else:
        logger.info("向量数据库和 BM25 索引均已存在，跳过初始化构建")
        if rag_model.bm25_retriever.documents:
            with rag_model._registry_lock:
                rag_model._document_registry = list(rag_model.bm25_retriever.documents)

    # Inject rag_model into routes
    import web.routes as _routes
    _routes.rag_model = rag_model


def _init_admin():
    """Ensure default admin account exists."""
    from web.db import SessionLocal
    from web.models import User

    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.role == "admin").first()
        if not admin:
            import secrets
            admin_password = os.getenv("ADMIN_PASSWORD")
            if not admin_password:
                admin_password = secrets.token_urlsafe(16)
                logger.warning(f"ADMIN_PASSWORD 未设置，已生成随机密码: {admin_password}")
                env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
                if os.path.exists(env_path):
                    with open(env_path, "a", encoding="utf-8") as f:
                        f.write(f"\nADMIN_PASSWORD={admin_password}\n")
            admin_user = User(phone="admin", username="管理员", role="admin")
            admin_user.set_password(admin_password)
            db.add(admin_user)
            db.commit()
            logger.info("已创建默认管理员账号")
        else:
            logger.info(f"管理员账号已存在: {admin.phone}")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app_instance):
    from law_assistant.redis_utils import close_pool, is_available
    if is_available():
        logger.info("Redis 连接成功")
    else:
        logger.warning("Redis 不可用，使用本地缓存和数据库回退")
    yield
    close_pool()


def create_app() -> FastAPI:
    global templates

    app = FastAPI(title="智能法律助手", lifespan=lifespan)

    # Register middleware (order matters: last registered = first executed)
    from web.middleware import security_headers, csrf_middleware, rate_limit_middleware
    app.middleware("http")(security_headers)
    app.middleware("http")(csrf_middleware)
    app.middleware("http")(rate_limit_middleware)

    # Mount static files and templates
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "static")
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "templates")
    if not os.path.isdir(static_dir):
        # Fallback to legacy location
        static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
        template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    templates = Jinja2Templates(directory=template_dir)

    # Inject templates into routes
    import web.routes as _routes
    _routes.templates = templates

    # Register routes
    from web.routes import router
    app.include_router(router)

    # Auth exception handler
    from web.auth import auth_redirect_handler
    app.exception_handler(401)(auth_redirect_handler)

    # Create directories
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(project_root, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "knowledge_base"), exist_ok=True)

    # Initialize DB, admin, and RAG model
    _init_admin()
    _init_rag()

    return app
