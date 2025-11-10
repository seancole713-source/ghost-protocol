"""
Shared helpers for worker tasks: Redis JSON set/get, and news sentiment bridging.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

try:
    import redis.asyncio as redis  # type: ignore
except Exception:
    redis = None  # type: ignore

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_PREFIX = os.environ.get("REDIS_PREFIX", "ghost")


async def get_redis():
    if redis is None:
        return None
    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        # tiny ping to verify
        try:
            await asyncio.wait_for(client.ping(), timeout=0.4)
        except Exception:
            return None
        return client
    except Exception:
        return None


def _key(name: str) -> str:
    return f"{REDIS_PREFIX}:{name}"


async def redis_set_json(name: str, obj: Any) -> bool:
    client = await get_redis()
    if client is None:
        return False
    try:
        await client.set(_key(name), json.dumps(obj, separators=(",", ":")))
        return True
    except Exception:
        return False


async def redis_get_json(name: str):
    client = await get_redis()
    if client is None:
        return None
    try:
        s = await client.get(_key(name))
        return json.loads(s) if s else None
    except Exception:
        return None


async def get_news_sentiment() -> float | None:
    # Bridge: read the latest news_signal score from existing mechanisms via Redis cache if present
    # Expecting the app to push 'news:signal:latest' occasionally; otherwise returns None.
    obj = await redis_get_json("news:signal:latest")
    try:
        return float((obj or {}).get("score"))
    except Exception:
        return None
