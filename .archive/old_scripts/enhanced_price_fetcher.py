import asyncio
import os
import time

import requests

from core.cache_manager import PRICE_CACHE, async_cached

COINGECKO_API_KEY = os.getenv("COINGECKO_API_KEY", "")


async def _http_get(url: str, headers: dict[str, str] | None = None) -> requests.Response:
    return await asyncio.to_thread(requests.get, url, {"timeout": 8, "headers": headers or {}})


@async_cached(cache=PRICE_CACHE, ttl=60.0)  # Cache for 1 minute
async def median_crypto(ids: list[str]) -> dict[str, dict]:
    """
    Fetch live crypto prices for given CoinGecko ids.
    Returns: { id: { price: float, timestamp: float, source: "coingecko" } }

    Cached for 60 seconds to reduce API calls.
    """
    ids = [i.strip().lower() for i in ids if i and i.strip()]
    if not ids:
        return {}

    joined = ",".join(sorted(set(ids)))
    base = "https://api.coingecko.com/api/v3/simple/price"
    url = f"{base}?ids={joined}&vs_currencies=usd&include_last_updated_at=true"
    headers = {"Accept": "application/json"}
    # CoinGecko Pro header (if provided)
    if COINGECKO_API_KEY:
        # Pro uses this header; keep both for compatibility
        headers["x-cg-pro-api-key"] = COINGECKO_API_KEY
        headers["x-cg-api-key"] = COINGECKO_API_KEY

    # Perform request in a thread to avoid blocking the event loop
    resp = await asyncio.to_thread(requests.get, url, timeout=8, headers=headers)
    resp.raise_for_status()
    data = resp.json() or {}

    out: dict[str, dict] = {}
    now = time.time()
    for cid in ids:
        item = data.get(cid)
        if not item:
            continue
        price = float(item.get("usd") or 0)
        ts = float(item.get("last_updated_at") or now)
        if price > 0:
            out[cid] = {"price": price, "timestamp": ts, "source": "coingecko"}
    return out
