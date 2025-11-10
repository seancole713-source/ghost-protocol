"""
Lightweight async state layer with Redis (if available) and in-memory fallback.

Contracts:
- save_state(**kwargs): merge into a persistent state dict
- get_universe() -> (stocks: list[str], crypto: list[str])
- set_universe(stocks, crypto)
- push_error(msg: str): append a timestamped error entry

Behavior:
- If redis is installed and reachable via REDIS_URL, data is persisted in Redis.
- Otherwise, operations use a module-level in-memory dict so app still works.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

_STATE: dict[str, Any] = {}
_UNIVERSE: dict[str, list[str]] = {"stocks": [], "crypto": []}
_ERRORS: list[dict[str, Any]] = []  # capped to last 500

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_PREFIX = os.environ.get("REDIS_PREFIX", "ghost")
REDIS_ENABLED = os.environ.get("ENABLE_REDIS", "1").lower() in ("1", "true", "yes", "on")

_redis = None  # lazy-initialized redis.asyncio client


def _cap_errors_in_memory() -> None:
    # Keep last 500
    if len(_ERRORS) > 500:
        del _ERRORS[500:]


async def _get_redis():
    """Return an async redis client or None if unavailable."""
    global _redis
    if not REDIS_ENABLED:
        return None
    if _redis is not None:
        return _redis
    try:
        # Prefer redis.asyncio for async usage
        import redis.asyncio as redis  # type: ignore

        # Use short socket timeouts so requests never block the API for long
        _redis = redis.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )
        # quick ping with bounded timeout to verify connection
        try:
            import asyncio as _aio

            async def _ping():
                try:
                    return await _redis.ping()  # type: ignore[attr-defined]
                except Exception:
                    return False

            ok = await _aio.wait_for(_ping(), timeout=0.5)
            if not ok:
                _redis = None
                return None
        except Exception:
            _redis = None
            return None
        return _redis
    except Exception:
        _redis = None
        return None


def _key(name: str) -> str:
    return f"{REDIS_PREFIX}:{name}"


async def save_state(**kwargs: Any) -> bool:
    """Merge kwargs into state; persist to Redis if available."""
    _STATE.update(kwargs)
    client = await _get_redis()
    if client is None:
        return True
    try:
        data = json.dumps(_STATE, separators=(",", ":"))
        await client.set(_key("state"), data)
        return True
    except Exception:
        return False


async def get_universe() -> tuple[list[str], list[str]]:
    client = await _get_redis()
    if client is None:
        return (_UNIVERSE.get("stocks", []), _UNIVERSE.get("crypto", []))
    try:
        s = await client.get(_key("universe:stocks"))
        c = await client.get(_key("universe:crypto"))
        stocks = json.loads(s) if s else []
        crypto = json.loads(c) if c else []
        return (stocks, crypto)
    except Exception:
        return (_UNIVERSE.get("stocks", []), _UNIVERSE.get("crypto", []))


async def set_universe(stocks: list[str], crypto: list[str]) -> bool:
    _UNIVERSE["stocks"] = list(stocks or [])
    _UNIVERSE["crypto"] = list(crypto or [])
    client = await _get_redis()
    if client is None:
        return True
    try:
        await client.set(_key("universe:stocks"), json.dumps(_UNIVERSE["stocks"]))
        await client.set(_key("universe:crypto"), json.dumps(_UNIVERSE["crypto"]))
        return True
    except Exception:
        return False


async def push_error(msg: str) -> bool:
    entry = {"ts": int(time.time()), "message": str(msg)}
    _ERRORS.insert(0, entry)
    _cap_errors_in_memory()
    client = await _get_redis()
    if client is None:
        return True
    try:
        await client.lpush(_key("errors"), json.dumps(entry))
        # keep only the last 500
        await client.ltrim(_key("errors"), 0, 499)
        return True
    except Exception:
        return False
