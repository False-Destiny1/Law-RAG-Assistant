import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime
from functools import wraps
from typing import Any

import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Connection Pool (lazy singleton, thread-safe) ---
_pool: redis.ConnectionPool | None = None
_redis_client: redis.Redis | None = None
_redis_lock = threading.Lock()
_last_connect_failure: float = 0.0  # timestamp of last connection failure
_RECONNECT_COOLDOWN = 30  # seconds to wait before retrying connection

# --- Local rate limiting fallback (when Redis is unavailable) ---
_local_rate_limits: dict = defaultdict(int)
_local_rate_lock = threading.Lock()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
KEY_PREFIX = "law_assistant"


def _get_client() -> redis.Redis | None:
    """Get or create the Redis client (thread-safe, with reconnect cooldown)."""
    global _pool, _redis_client, _last_connect_failure
    if _redis_client is not None:
        return _redis_client

    # Cooldown: don't retry connection too frequently after failure
    now = datetime.now().timestamp()
    if _last_connect_failure and (now - _last_connect_failure) < _RECONNECT_COOLDOWN:
        return None

    with _redis_lock:
        if _redis_client is not None:
            return _redis_client
        try:
            _pool = redis.ConnectionPool.from_url(
                REDIS_URL,
                max_connections=20,
                decode_responses=False,
                socket_timeout=2,
                socket_connect_timeout=2,
                retry_on_timeout=True,
            )
            _redis_client = redis.Redis(connection_pool=_pool)
            _redis_client.ping()
            _last_connect_failure = 0.0
            logger.info(f"Redis connected: {REDIS_URL}")
            return _redis_client
        except Exception as e:
            _last_connect_failure = datetime.now().timestamp()
            logger.warning(f"Redis unavailable, falling back to local behavior: {e}")
            _redis_client = None
            _pool = None
            return None


def _key(suffix: str) -> str:
    return f"{KEY_PREFIX}:{suffix}"


def redis_fallback(default=None):
    """Decorator: if Redis call raises, log warning and return default."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Redis error in {func.__name__}: {e}")
                return default

        return wrapper

    return decorator


# --- Generic JSON helpers ---


@redis_fallback(None)
def cache_get_json(key: str) -> Any | None:
    client = _get_client()
    if client is None:
        return None
    data = client.get(_key(key))
    if data is None:
        return None
    return json.loads(data)


@redis_fallback()
def cache_set_json(key: str, value: Any, ttl: int = 3600):
    client = _get_client()
    if client is None:
        return
    client.setex(_key(key), ttl, json.dumps(value, ensure_ascii=False, default=str))


@redis_fallback()
def cache_delete(key: str):
    client = _get_client()
    if client is None:
        return
    client.delete(_key(key))


@redis_fallback()
def cache_delete_pattern(pattern: str):
    client = _get_client()
    if client is None:
        return
    full_pattern = _key(pattern)
    cursor = 0
    while True:
        cursor, keys = client.scan(cursor=cursor, match=full_pattern, count=100)
        if keys:
            client.delete(*keys)
        if cursor == 0:
            break


# --- JSON helpers for complex objects (replaces pickle for security) ---


@redis_fallback(None)
def cache_get_set(key: str) -> set | None:
    """Load a set stored as a JSON array. Returns None on miss."""
    client = _get_client()
    if client is None:
        return None
    data = client.get(_key(key))
    if data is None:
        return None
    return set(json.loads(data))


@redis_fallback()
def cache_set_set(key: str, value: set, ttl: int = 3600):
    """Store a set as a JSON array."""
    client = _get_client()
    if client is None:
        return
    client.setex(_key(key), ttl, json.dumps(list(value), ensure_ascii=False))


# --- Rate limiting ---


def rate_limit_check(identifier: str, limit: int, window: int = 60):
    """
    Fixed-window rate limit check.
    Returns: (allowed: bool, current_count: int, remaining: int)
    Falls back to local in-memory rate limiting when Redis is unavailable.
    """
    client = _get_client()
    if client is None:
        # Local in-memory fallback (fixed-window counter)
        now = datetime.now().timestamp()
        window_key = int(now) // window
        key = f"{identifier}:{window_key}"
        with _local_rate_lock:
            # Periodic cleanup: remove entries older than 2 windows
            if len(_local_rate_limits) > 1000:
                cutoff = int(now) // window - 2
                stale = [k for k in _local_rate_limits if int(k.rsplit(":", 1)[-1]) < cutoff]
                for k in stale:
                    del _local_rate_limits[k]
            _local_rate_limits[key] += 1
            count = _local_rate_limits[key]
        remaining = max(0, limit - count)
        return (count <= limit, count, remaining)
    try:
        key = _key(f"ratelimit:{identifier}:{int(datetime.now().timestamp()) // window}")
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, window)
        result = pipe.execute()
        count = result[0]
        remaining = max(0, limit - count)
        return (count <= limit, count, remaining)
    except Exception as e:
        logger.warning(f"Redis rate limit error: {e}")
        # Same local fallback on Redis error
        now = datetime.now().timestamp()
        window_key = int(now) // window
        key = f"{identifier}:{window_key}"
        with _local_rate_lock:
            _local_rate_limits[key] += 1
            count = _local_rate_limits[key]
        remaining = max(0, limit - count)
        return (count <= limit, count, remaining)


# --- Session token blacklist ---


def blacklist_token(token: str, ttl_seconds: int = 86400) -> None:
    """Add a session token to the blacklist (TTL matches session expiry)."""
    client = _get_client()
    if client is None:
        return
    try:
        client.setex(_key(f"blacklist:{token}"), ttl_seconds, "1")
    except Exception as e:
        logger.warning(f"Failed to blacklist token: {e}")


def is_token_blacklisted(token: str) -> bool:
    """Check if a session token has been blacklisted."""
    client = _get_client()
    if client is None:
        return False
    try:
        return client.exists(_key(f"blacklist:{token}")) > 0
    except Exception as e:
        logger.warning(f"Failed to check token blacklist: {e}")
        return False


# --- Health check ---


def is_available() -> bool:
    try:
        client = _get_client()
        return client is not None and client.ping()
    except Exception:
        return False


def close_pool():
    global _pool, _redis_client
    with _redis_lock:
        if _pool:
            _pool.disconnect()
        _redis_client = None
        _pool = None
