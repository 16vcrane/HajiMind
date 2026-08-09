"""Optional Redis cache helpers."""
import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_client = None
_disabled = False


def get_redis_client():
    global _client, _disabled
    if _disabled:
        return None
    if _client is not None:
        return _client
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    try:
        import redis

        _client = redis.Redis.from_url(redis_url, decode_responses=True)
        _client.ping()
        return _client
    except Exception:
        _disabled = True
        return None


def cache_get_json(key: str) -> Any | None:
    client = get_redis_client()
    if not client:
        return None
    try:
        raw = client.get(key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


def cache_set_json(key: str, value: Any, ttl: int = 300) -> None:
    client = get_redis_client()
    if not client:
        return
    try:
        client.setex(key, ttl, json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        return


def cache_delete(*keys: str) -> None:
    client = get_redis_client()
    if not client or not keys:
        return
    try:
        client.delete(*keys)
    except Exception:
        return


def cache_delete_pattern(pattern: str) -> None:
    client = get_redis_client()
    if not client:
        return
    try:
        keys = list(client.scan_iter(match=pattern, count=100))
        if keys:
            client.delete(*keys)
    except Exception:
        return
