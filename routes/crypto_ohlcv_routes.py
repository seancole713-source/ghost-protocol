import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

crypto_ohlcv_router = APIRouter()


def _http_get_simple(url: str, headers: dict[str, str] | None = None, timeout: float = 15.0):
    if requests is None:
        # Minimal urllib fallback if requests is unavailable
        import urllib.request

        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            data = resp.read()

            class R:
                status_code = resp.getcode()
                text = data.decode("utf-8", errors="ignore")

                def json(self):
                    try:
                        return json.loads(self.text)
                    except Exception:
                        return {}

            return R()
    else:
        return requests.get(url, headers=headers or {}, timeout=timeout)


@crypto_ohlcv_router.get("/api/crypto/ohlcv/{symbol}")
async def api_crypto_ohlcv(symbol: str, days: int = 30, interval: str = "1h") -> dict[str, Any]:
    """
    Historical OHLCV for crypto using CoinGecko market_chart.
    Public, read-only endpoint.
    """
    if os.getenv("CRYPTO_ENABLED", "0") != "1":
        raise HTTPException(503, "Crypto module not enabled")

    # Resolve CoinGecko ID if possible via simple map; fallback to lower symbol
    sym = (symbol or "").strip().upper()
    symbol_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "DOGE": "dogecoin",
        "SHIB": "shiba-inu",
        "PEPE": "pepe",
        "XRP": "ripple",
        "ADA": "cardano",
        "BNB": "binancecoin",
    }
    gecko_id = symbol_map.get(sym, (symbol or "").strip().lower())

    base = os.getenv("COINGECKO_BASE", "https://api.coingecko.com/api/v3").rstrip("/")
    key = os.getenv("COINGECKO_API_KEY", "").strip()
    granularity = "hourly" if interval.lower() in ("1h", "hour", "hourly") else "daily"
    url = f"{base}/coins/{gecko_id}/market_chart?vs_currency=usd&days={days}&interval={granularity}"

    headers: dict[str, str] = {}
    if key:
        headers["x-cg-pro-api-key"] = key
        headers["x-cg-api-key"] = key

    try:
        resp = _http_get_simple(
            url, headers=headers, timeout=float(os.getenv("HTTP_TIMEOUT_S", "15"))
        )
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"coingecko_error: {resp.text[:200]}")
        data = resp.json() or {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"OHLCV fetch failed: {e}") from e

    prices = data.get("prices") or []
    totals = data.get("total_volumes") or []
    candles = []
    for i, row in enumerate(prices):
        try:
            ts_ms, close = row
            prev = prices[i - 1][1] if i > 0 else close
            high = max(prev, close)
            low = min(prev, close)
            vol = 0.0
            if i > 0 and i < len(totals):
                vol = max(0.0, float(totals[i][1]) - float(totals[i - 1][1]))
            candles.append(
                {
                    "t": int(ts_ms // 1000),
                    "o": float(prev),
                    "h": float(high),
                    "l": float(low),
                    "c": float(close),
                    "v": float(vol),
                }
            )
        except Exception:
            continue

    return {
        "symbol": sym,
        "gecko_id": gecko_id,
        "interval": granularity,
        "count": len(candles),
        "candles": candles,
        "source": "coingecko",
    }
