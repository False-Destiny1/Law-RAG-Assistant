import hashlib
import hmac
import logging
import os
import secrets
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from web.db import get_db
from web.models import User

logger = logging.getLogger(__name__)

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


def auth_redirect_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/") or request.url.path.startswith("/ask_stream"):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return RedirectResponse(url="/login", status_code=303)
