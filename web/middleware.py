import hashlib
import hmac
import logging
import os
import urllib.parse
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse
from web.auth import SESSION_SECRET

logger = logging.getLogger(__name__)


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
