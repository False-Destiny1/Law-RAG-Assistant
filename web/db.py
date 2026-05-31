import logging
import os
import secrets

from sqlalchemy import create_engine, text as _text
from sqlalchemy.orm import sessionmaker

from web.models import Base, User

logger = logging.getLogger(__name__)

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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
