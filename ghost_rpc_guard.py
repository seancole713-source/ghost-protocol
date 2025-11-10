# ghost_rpc_guard.py
# RPC efficiency + provider rotation + QPS guard + TTL cache + batch ERC20 balances.
# No external deps beyond 'requests'. Thread-safe.

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import deque
from typing import Any

import requests

# --- Config from env ---
PRIMARY = os.getenv("RPC_URL", "").strip()
ROTATION = [u.strip() for u in os.getenv("RPC_URLS", "").split(",") if u.strip()]
PROVIDERS: list[str] = [u for u in [PRIMARY] + ROTATION if u]
if not PROVIDERS:
    # Fallback local, but you should set RPC_URL in Secrets
    PROVIDERS = ["http://127.0.0.1:8545"]

RPC_BUDGET_DAILY = int(os.getenv("RPC_BUDGET_DAILY", "300000"))
RPC_QPS_MAX = int(os.getenv("RPC_QPS_MAX", "5"))
RPC_TTL_SEC = int(os.getenv("RPC_TTL_SEC", "15"))

# TTL map per method
TTL_MAP: dict[str, int] = {
    "eth_blockNumber": max(1, min(RPC_TTL_SEC, 15)),
    "eth_getBalance": max(1, min(RPC_TTL_SEC, 30)),
    "eth_call": max(1, min(RPC_TTL_SEC, 15)),
}

# --- State ---
_session = requests.Session()
_session.headers.update({"Content-Type": "application/json"})
_timeout = 20

_provider_ix = 0
_provider_lock = threading.Lock()

# QPS guard
_qps_lock = threading.Lock()
_qps_window = deque()  # epoch seconds of recent calls

# Budget tracking (resets when date changes)
_usage_lock = threading.Lock()
_usage = {
    "date": time.strftime("%Y-%m-%d", time.gmtime()),
    "total_requests": 0,
    "by_provider": {p: 0 for p in PROVIDERS},
    "warnings": [],
    "started_at": int(time.time()),
}

# Simple TTL cache: key -> (expires_at, value)
_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, Any]] = {}


# --- Helpers ---
def _now() -> float:
    return time.time()


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def _rotate_provider():
    global _provider_ix
    with _provider_lock:
        _provider_ix = (_provider_ix + 1) % len(PROVIDERS)
        return PROVIDERS[_provider_ix]


def _current_provider():
    with _provider_lock:
        return PROVIDERS[_provider_ix]


def _budget_ok(n: int) -> bool:
    with _usage_lock:
        # reset per-day
        if _usage["date"] != _today():
            _usage["date"] = _today()
            _usage["total_requests"] = 0
            _usage["by_provider"] = {p: 0 for p in PROVIDERS}
            _usage["warnings"] = []
        will = _usage["total_requests"] + n
        if will > RPC_BUDGET_DAILY:
            return False
        if will > int(RPC_BUDGET_DAILY * 0.8) and "80" not in _usage["warnings"]:
            _usage["warnings"].append("80")
        return True


def _budget_add(provider: str, n: int):
    with _usage_lock:
        _usage["total_requests"] += n
        _usage["by_provider"][provider] = _usage["by_provider"].get(provider, 0) + n


def _qps_wait(units: int):
    """Ensure <= RPC_QPS_MAX requests per second (units = number of RPC calls we are about to do)."""
    if units <= 0:
        return
    with _qps_lock:
        now = _now()
        # purge older than 1s
        while _qps_window and now - _qps_window[0] > 1.0:
            _qps_window.popleft()
        # If adding 'units' would exceed limit, sleep until safe
        while len(_qps_window) + units > RPC_QPS_MAX:
            to_sleep = 1.0 - (now - _qps_window[0]) if _qps_window else 0.2
            if to_sleep > 0:
                time.sleep(min(0.5, max(0.01, to_sleep)))
            now = _now()
            while _qps_window and now - _qps_window[0] > 1.0:
                _qps_window.popleft()
        # Reserve slots
        for _ in range(units):
            _qps_window.append(_now())


def _cache_key(method: str, params: Any, provider: str) -> str:
    m = hashlib.sha256()
    m.update(method.encode())
    m.update(json.dumps(params, separators=(",", ":"), sort_keys=True).encode())
    # do not pin cache to provider to allow reuse across rotations
    return m.hexdigest()


def _cache_get(key: str):
    with _cache_lock:
        ent = _cache.get(key)
        if not ent:
            return None
        exp, val = ent
        if _now() < exp:
            return val
        # expired
        _cache.pop(key, None)
        return None


def _cache_set(key: str, ttl: int, value: Any):
    if ttl <= 0:
        return
    with _cache_lock:
        _cache[key] = (_now() + ttl, value)


def _post(provider: str, payload: Any) -> requests.Response:
    return _session.post(provider, data=json.dumps(payload), timeout=_timeout)


# --- Public API ---
def rpc_call(
    method: str,
    params: list,
    ttl_override: int | None = None,
    preferred_provider: str | None = None,
) -> tuple[Any, str]:
    """
    Call a single JSON-RPC method with rotation, QPS, budget, and caching.
    Returns: (result, provider_url)
    Raises: RuntimeError on failure or budget exceeded.
    """
    ttl = TTL_MAP.get(method, 0)
    if ttl_override is not None:
        ttl = max(0, int(ttl_override))

    key = _cache_key(method, params, "any")
    if ttl > 0:
        val = _cache_get(key)
        if val is not None:
            return val, _current_provider()

    # Budget check (1 call)
    if not _budget_ok(1):
        raise RuntimeError("rpc budget exceeded")

    # QPS guard
    _qps_wait(1)

    # choose provider
    provider = preferred_provider or _current_provider()

    # Try each provider at most once
    tried = set()
    for _ in range(len(PROVIDERS)):
        tried.add(provider)
        try:
            resp = _post(provider, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
            if resp.status_code >= 500:
                raise RuntimeError(f"{provider} HTTP {resp.status_code}")
            data = resp.json()
            if "error" in data:
                # Rotate on provider-side errors
                raise RuntimeError(f"{provider} RPC error: {data['error']}")
            res = data.get("result")
            _budget_add(provider, 1)
            if ttl > 0:
                _cache_set(key, ttl, res)
            return res, provider
        except Exception:
            # rotate
            provider = _rotate_provider()
            if provider in tried:
                continue
    raise RuntimeError("all providers failed")


def rpc_batch(
    payloads: list[dict[str, Any]], units_cost: int | None = None
) -> tuple[list[Any], str]:
    """
    Send a JSON-RPC batch. 'payloads' is a list of individual RPC objects.
    Returns (results_list, provider).
    """
    n = len(payloads)
    cost = int(units_cost) if units_cost is not None else n
    if n == 0:
        return [], _current_provider()

    if not _budget_ok(cost):
        raise RuntimeError("rpc budget exceeded")

    _qps_wait(cost)
    provider = _current_provider()
    tried = set()
    for _ in range(len(PROVIDERS)):
        tried.add(provider)
        try:
            resp = _post(provider, payloads)
            if resp.status_code >= 500:
                raise RuntimeError(f"{provider} HTTP {resp.status_code}")
            arr = resp.json()
            if not isinstance(arr, list):
                raise RuntimeError(f"{provider} invalid batch response")
            # map id -> result
            id2 = {e["id"]: e for e in arr if isinstance(e, dict)}
            out = []
            for obj in payloads:
                rid = obj.get("id")
                r = id2.get(rid, {})
                if "error" in r:
                    out.append({"error": r["error"]})
                else:
                    out.append(r.get("result"))
            _budget_add(provider, cost)
            return out, provider
        except Exception:
            provider = _rotate_provider()
            if provider in tried:
                continue
    raise RuntimeError("all providers failed (batch)")


def _pad_hex_address(addr: str) -> str:
    a = addr.lower()
    if a.startswith("0x"):
        a = a[2:]
    return "0" * (64 - len(a)) + a


def erc20_balances_multicall(
    wallet: str, tokens: list[str], block: str = "latest", ttl_sec: int = 30
) -> dict[str, int]:
    """
    Batch balanceOf(wallet) for many ERC20 tokens using JSON-RPC batch eth_call.
    Returns {token_address_lower: balance_int}
    """
    if not tokens:
        return {}
    # Prepare batch payloads
    w = wallet
    if w.startswith("0x"):
        w = w[2:]
    arg = _pad_hex_address(wallet)
    selector = "0x70a08231"  # balanceOf(address)
    payloads = []
    # Cache key across the set
    ck = _cache_key("erc20_balances_multicall", {"w": wallet, "t": tokens, "b": block}, "any")
    if ttl_sec > 0:
        val = _cache_get(ck)
        if val is not None:
            return val

    req_id = 1
    for t in tokens:
        to = t
        if not to.startswith("0x"):
            to = "0x" + to
        data = selector + arg
        payloads.append(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "eth_call",
                "params": [{"to": to, "data": data}, block],
            }
        )
        req_id += 1

    results, provider = rpc_batch(payloads)  # counts cost ~ len(tokens)
    out: dict[str, int] = {}
    for tok, res in zip(tokens, results, strict=False):
        if isinstance(res, dict) and "error" in res:
            out[tok.lower()] = 0
            continue
        if isinstance(res, str) and res.startswith("0x"):
            try:
                out[tok.lower()] = int(res, 16)
            except Exception:
                out[tok.lower()] = 0
        else:
            out[tok.lower()] = 0

    if ttl_sec > 0:
        _cache_set(ck, ttl_sec, out)
    return out


def rpc_usage_totals() -> dict[str, Any]:
    with _usage_lock, _qps_lock, _cache_lock, _provider_lock:
        return {
            "date": _usage["date"],
            "started_at": _usage["started_at"],
            "total_requests": _usage["total_requests"],
            "by_provider": dict(_usage["by_provider"]),
            "providers": list(PROVIDERS),
            "current_provider": PROVIDERS[_provider_ix],
            "qps_window_size": len(_qps_window),
            "budget_daily": RPC_BUDGET_DAILY,
            "qps_max": RPC_QPS_MAX,
            "cache_items": len(_cache),
            "ttl_map": TTL_MAP,
            "warnings": list(_usage["warnings"]),
        }
