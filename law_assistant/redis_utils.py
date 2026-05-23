import os
import json
import pickle
import logging
from datetime import datetime
from typing import Optional, Any
from functools import wraps

import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# --- Connection Pool (lazy singleton) ---
_pool: Optional[redis.ConnectionPool] = None
_redis_client: Optional[redis.Redis] = None

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
KEY_PREFIX = "law_assistant"


def _get_client() -> Optional[redis.Redis]:
    """Get or create the Redis client. Returns None if Redis is unavailable."""
    global _pool, _redis_client
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
        logger.info(f"Redis connected: {REDIS_URL}")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable, falling back to local behavior: {e}")
        _redis_client = None
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
def cache_get_json(key: str) -> Optional[Any]:
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


# --- Pickle helpers (for complex objects like sets) ---

@redis_fallback(None)
def cache_get_pickle(key: str) -> Optional[Any]:
    client = _get_client()
    if client is None:
        return None
    data = client.get(_key(key))
    if data is None:
        return None
    return pickle.loads(data)


@redis_fallback()
def cache_set_pickle(key: str, value: Any, ttl: int = 3600):
    client = _get_client()
    if client is None:
        return
    client.setex(_key(key), ttl, pickle.dumps(value))


# --- Rate limiting ---

@redis_fallback((True, 0, 999))
def rate_limit_check(identifier: str, limit: int, window: int = 60):
    """
    Fixed-window rate limit check.
    Returns: (allowed: bool, current_count: int, remaining: int)
    """
    client = _get_client()
    if client is None:
        return (True, 0, limit)
    key = _key(f"ratelimit:{identifier}:{int(datetime.now().timestamp()) // window}")
    pipe = client.pipeline()
    pipe.incr(key)
    pipe.expire(key, window)
    result = pipe.execute()
    count = result[0]
    remaining = max(0, limit - count)
    return (count <= limit, count, remaining)


# --- Health check ---

def is_available() -> bool:
    try:
        client = _get_client()
        return client is not None and client.ping()
    except Exception:
        return False


def close_pool():
    global _pool, _redis_client
    if _pool:
        _pool.disconnect()
    _redis_client = None
    _pool = None
