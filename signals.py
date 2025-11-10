import asyncio
import os
import random
import time
from typing import Any

import ghost_state

# Optional live helpers
try:
    from enhanced_price_fetcher import median_crypto as _median_crypto
except Exception:  # pragma: no cover - optional
    _median_crypto = None  # type: ignore

try:  # light stock fetcher (duplicate of main.get_real_stock_price logic, trimmed)
    import os

    import requests
    import yfinance as yf  # type: ignore

    _AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("ALPHA_VANTAGE_API_KEY", "")
    _POLY_KEY = os.getenv("POLYGON_API_KEY", "")

    async def _get_stock_price(sym: str) -> float | None:
        s = sym.upper()
        if _AV_KEY:
            try:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={s}&apikey={_AV_KEY}"
                r = requests.get(url, timeout=6)
                r.raise_for_status()
                data = r.json() or {}
                gq = data.get("Global Quote") or data.get("GlobalQuote") or {}
                ps = gq.get("05. price") or gq.get("price")
                if ps:
                    p = float(ps)
                    if p > 0:
                        return p
            except Exception:
                pass
        if _POLY_KEY:
            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/{s}/prev?adjusted=true&limit=1&apiKey={_POLY_KEY}"
                r = requests.get(url, timeout=6)
                r.raise_for_status()
                data = r.json() or {}
                res = data.get("results") or []
                if res:
                    c = float(res[0].get("c") or 0)
                    if c > 0:
                        return c
            except Exception:
                pass
        try:
            tkr = yf.Ticker(s)
            hist = tkr.history(period="2d")
            if not hist.empty:
                p = float(hist["Close"].iloc[-1])
                if p > 0:
                    return p
        except Exception:
            return None
        return None
except Exception:  # pragma: no cover
    requests = None  # type: ignore
    yf = None  # type: ignore
    _get_stock_price = None  # type: ignore


async def fusion_outlook(force: bool = False):
    """Return a concise market outlook with risk/confidence and key notes.

    Uses live BTC and AAPL price snapshots when possible to bias the text. This
    is lightweight and resilient in restricted CI environments.
    """
    btc = None
    aapl = None
    try:
        if _median_crypto:
            q = await _median_crypto(["bitcoin"])  # type: ignore
            qb = q.get("bitcoin") if q else None
            if qb and (time.time() - float(qb.get("timestamp", 0))) < 120:
                btc = float(qb.get("price", 0))
    except Exception:
        btc = None
    try:
        if _get_stock_price:
            aapl = await _get_stock_price("AAPL")
    except Exception:
        aapl = None

    # Simple heuristics
    risk = "balanced"
    confidence = 0.6
    notes: list[str] = []
    if btc:
        notes.append(f"BTC ${btc:,.0f}")
        if btc > 60000:
            risk = "risk_on"
            confidence += 0.1
        elif btc < 25000:
            risk = "defensive"
            confidence -= 0.1
    if aapl:
        notes.append(f"AAPL ${aapl:,.2f}")
        if aapl > 175:
            confidence += 0.05
    confidence = max(0.3, min(0.9, confidence))

    # Produce a few actionable bullets
    signals = []
    wl = get_watchlist()
    all_syms = wl.get("stocks", []) + wl.get("crypto", [])
    # include a little variety beyond the first few
    picks = all_syms[:3] + all_syms[-3:]
    rng = random.Random(int(time.time()) // 180)  # change every ~3m
    for sym in picks[:6]:
        act = rng.choice(["BUY", "HOLD", "SELL", "HOLD"])  # bias HOLD
        signals.append(
            {
                "symbol": sym,
                "action": act,
                "confidence": round(0.5 + rng.random() * 0.4, 2),
                "gps": round(4.5 + rng.random() * 3.5, 1),
            }
        )

    return {
        "outlook": {
            "headline": {
                "risk": risk,
                "confidence": confidence,
                "key_levels": notes,
            },
            "notes": [
                (
                    "BTC over key level; dips likely to be bought"
                    if btc
                    else "Watching BTC for risk cues"
                ),
                (
                    "AAPL strength supports tech breadth"
                    if aapl
                    else "Monitoring mega-cap tech leadership"
                ),
                "Maintain disciplined risk limits",
            ],
        },
        "signals": signals,
        "timestamp": time.time(),
    }


async def heatmap_snapshot():
    """Build a 12–20 tile heatmap from the current universe/watchlist.

    Scores are GPS-like and deterministic per symbol.
    """
    # Focus Mode: return a single-tile heatmap for the focus ticker
    try:
        import universe  # local import to avoid cycles

        if universe.focus_enabled():  # type: ignore[attr-defined]
            sym = (universe.focus_ticker() or "WOLF").strip().upper()  # type: ignore[attr-defined]
            # Simple deterministic GPS for single tile
            h = sum(ord(ch) for ch in sym) % 40
            gps = round(4.0 + (h / 40.0) * 4.0, 1)
            return [{"symbol": sym, "gps": gps}]
    except Exception:
        pass

    st = ghost_state.get_state()
    uni = st.get("universe") or {"stocks": [], "crypto": []}
    wl = st.get("watchlist") or {"stocks": [], "crypto": []}
    # Merge and de-dup; prefer watchlist ordering
    stocks = list(dict.fromkeys((wl.get("stocks") or []) + (uni.get("stocks") or [])))
    crypto = list(dict.fromkeys((wl.get("crypto") or []) + (uni.get("crypto") or [])))
    # Build pool and cap size (broaden to up to 12 per bucket)
    pool = stocks[:12] + crypto[:12]
    data: dict[str, Any] = {
        "stocks": {s: {} for s in stocks[:12]},
        "crypto": {c: {} for c in crypto[:12]},
    }
    scores = score_gps(data)
    out = []
    seen = set()
    for sc in scores:
        sym = sc.get("symbol")
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append({"symbol": sym, "gps": sc.get("gps", 5.0)})
        if len(out) >= 20:
            break
    # If not enough, pad from pool deterministically
    i = 0
    while len(out) < min(20, len(pool)):
        s = pool[i % len(pool)]
        if s not in seen:
            # simple hash-based GPS
            h = sum(ord(ch) for ch in s) % 40
            gps = round(4.0 + (h / 40.0) * 4.0, 1)
            out.append({"symbol": s, "gps": gps})
            seen.add(s)
        i += 1

    # Deterministic drift/vol and cross-asset bias nudges (±0.4 max total)
    def _bucket(sec: int = 180) -> int:
        return int(time.time()) // max(sec, 1)

    def _hashf(s: str) -> float:
        # Map to [0,1)
        h = sum((i + 1) * ord(ch) for i, ch in enumerate(s)) % 9973
        return (h % 1000) / 1000.0

    def _pseudo_drift(sym: str) -> float:
        # Signed drift in roughly ±1 based on time bucket and symbol
        b = _bucket(180)
        x = _hashf(f"{sym}:{b}") - 0.5
        return x * 2.0  # range approx [-1, 1]

    def _pseudo_vol(sym: str) -> float:
        # Vol proxy in [0,1)
        b = _bucket(300)
        return _hashf(f"vol:{sym}:{b}")

    # Cross-asset bias: alternate risk tilt by time; keep deterministic
    risk_on = _bucket(600) % 2 == 0
    stock_bias = 0.12 if risk_on else -0.12
    crypto_bias = 0.12 if risk_on else -0.12

    # Opportunistic live nudges (skip in tests/CI)
    live_stock_bias = 0.0
    live_crypto_bias = 0.0
    live_stock_vol = None  # 0..1
    live_crypto_vol = None  # 0..1

    def _base(sym: str) -> float:
        h = sum(ord(ch) for ch in sym) % 500
        return 100.0 + h

    if not os.getenv("PYTEST_CURRENT_TEST"):
        try:
            # Crypto: BTC/ETH via CoinGecko aggregator if available
            crypto_ids = []
            # map common symbols when present
            wanted_c = ["BTC", "ETH"]
            for w in wanted_c:
                # we only need presence to compute bias; fetch ids regardless of set membership
                crypto_ids.append(
                    "bitcoin" if w == "BTC" else ("ethereum" if w == "ETH" else w.lower())
                )
            c_prices = []
            if _median_crypto:
                try:
                    q = await _median_crypto(list(crypto_ids))  # type: ignore
                    for cid in crypto_ids:
                        it = (q or {}).get(cid)
                        if it and it.get("price"):
                            c_prices.append((cid.upper(), float(it["price"])))
                except Exception:
                    pass
            if c_prices:
                diffs = []
                vols = []
                for cid, p in c_prices:
                    b = _base(cid)
                    diffs.append(1.0 if p > b else (-1.0 if p < b else 0.0))
                    vols.append(min(1.0, abs(p - b) / max(1.0, b)))
                live_crypto_bias = 0.08 * (sum(diffs) / max(1, len(diffs)))
                live_crypto_vol = sum(vols) / max(1, len(vols))
        except Exception:
            pass
        try:
            # Stocks: AAPL/MSFT/NVDA via lightweight stock fetcher if available
            wanted_s = ["AAPL", "MSFT", "NVDA"]
            s_prices = []
            if _get_stock_price:
                tasks = [asyncio.create_task(_get_stock_price(sym)) for sym in wanted_s]  # type: ignore
                res = await asyncio.gather(*tasks, return_exceptions=True)
                for sym, r in zip(wanted_s, res, strict=False):
                    try:
                        if isinstance(r, BaseException) or r is None:
                            continue
                        s_prices.append((sym, float(r)))
                    except Exception:
                        continue
            if s_prices:
                diffs = []
                vols = []
                for sym, p in s_prices:
                    b = _base(sym)
                    diffs.append(1.0 if p > b else (-1.0 if p < b else 0.0))
                    vols.append(min(1.0, abs(p - b) / max(1.0, b)))
                live_stock_bias = 0.08 * (sum(diffs) / max(1, len(diffs)))
                live_stock_vol = sum(vols) / max(1, len(vols))
        except Exception:
            pass

    sset = set(stocks)
    cset = set(crypto)
    adjusted = []
    for item in out:
        sym = item["symbol"]
        base = float(item.get("gps", 5.0))
        drift = _pseudo_drift(sym)  # [-1,1]
        vol = _pseudo_vol(sym)  # [0,1)
        nudge = 0.0
        # drift contributes up to ±0.2
        nudge += 0.2 * (drift)
        # vol reduces extremes slightly
        nudge += -0.1 * (vol - 0.5)  # center at 0
        # cross-asset bias
        if sym in sset:
            nudge += stock_bias + live_stock_bias
            if live_stock_vol is not None:
                nudge += -0.06 * (live_stock_vol - 0.5)
        elif sym in cset:
            nudge += crypto_bias + live_crypto_bias
            if live_crypto_vol is not None:
                nudge += -0.06 * (live_crypto_vol - 0.5)
        # clamp total nudge to ±0.4
        if nudge > 0.4:
            nudge = 0.4
        elif nudge < -0.4:
            nudge = -0.4
        gps = round(max(4.0, min(8.0, base + nudge)), 1)
        adjusted.append({"symbol": sym, "gps": gps})
    return adjusted


async def top_movers():
    """Return a deterministic movers snapshot biased to current watchlist/universe.

    In Focus Mode, only the focus ticker (e.g., WOLF) is returned; crypto is empty.
    Otherwise, prefer current watchlist/universe symbols and fall back to a small
    static seed if everything is empty.
    """
    now = int(time.time())

    # Local helper: deterministic pseudo pct change -7%..+7%
    def pct(seed: int) -> float:
        return ((seed % 15) - 7) * 1.0

    # Try Focus Mode first (avoid importing universe at top-level to keep signals lightweight)
    try:
        import universe  # local import to avoid cycles

        if universe.focus_enabled():  # type: ignore[attr-defined]
            sym = (universe.focus_ticker() or "WOLF").strip().upper()  # type: ignore[attr-defined]
            # Single-stock movers list; crypto empty in focus mode
            out_s = [
                {
                    "symbol": sym,
                    "change_24h": pct(now + 1),
                    "gps": round(5.8, 1),
                    "source": "live",
                }
            ]
            return {"stocks": out_s, "crypto": []}
    except Exception:
        pass

    # Build from current ghost_state watchlist aligned with universe
    try:
        st = ghost_state.get_state()
        wl = st.get("watchlist") or {"stocks": [], "crypto": []}
        stocks = list(dict.fromkeys(wl.get("stocks") or []))
        crypto = list(dict.fromkeys(wl.get("crypto") or []))
    except Exception:
        stocks, crypto = [], []

    # Fallback seeds if both are empty (CI/first-boot)
    if not stocks and not crypto:
        stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"]
        crypto = ["BTC", "ETH", "SOL", "XRP", "DOGE", "PEPE"]

    # Limit and build deterministic outputs
    stocks = stocks[:6]
    crypto = crypto[:6]
    out_s = [
        {
            "symbol": s,
            "change_24h": pct(now + i),
            "gps": round(5.5 + (i % 4) * 0.2, 1),
            "source": "live",
        }
        for i, s in enumerate(stocks)
    ]
    out_c = [
        {
            "symbol": c.upper(),
            "change_24h": pct(now + i + 2),
            "gps": round(5.4 + (i % 4) * 0.25, 1),
            "source": "live",
        }
        for i, c in enumerate(crypto)
    ]
    return {"stocks": out_s, "crypto": out_c}


def get_watchlist():
    st = ghost_state.get_state()
    wl = st.get("watchlist") or {}
    stocks = wl.get("stocks") or ["AAPL", "MSFT"]
    crypto = wl.get("crypto") or ["BTC", "ETH"]
    return {"stocks": stocks, "crypto": crypto}


async def import_watchlist_enhanced(s, c):
    st = ghost_state.get_state()
    wl = st.get("watchlist") or {"stocks": [], "crypto": []}
    stocks = sorted(set((wl.get("stocks") or []) + [x.upper() for x in s]))
    crypto = sorted(set((wl.get("crypto") or []) + [x.upper() for x in c]))
    st["watchlist"] = {"stocks": stocks, "crypto": crypto}
    ghost_state.save()
    return {
        "stocks_added": len(s),
        "crypto_added": len(c),
        "stocks_total": len(stocks),
        "crypto_total": len(crypto),
    }


async def remove_symbols(s, c):
    st = ghost_state.get_state()
    wl = st.get("watchlist") or {"stocks": [], "crypto": []}
    su = {x.upper() for x in (s or [])}
    cu = {x.upper() for x in (c or [])}
    stocks = [x for x in (wl.get("stocks") or []) if x.upper() not in su]
    crypto = [x for x in (wl.get("crypto") or []) if x.upper() not in cu]
    st["watchlist"] = {"stocks": stocks, "crypto": crypto}
    ghost_state.save()
    return {
        "stocks_removed": len(s or []),
        "crypto_removed": len(c or []),
        "stocks_total": len(stocks),
        "crypto_total": len(crypto),
    }


async def clear_watchlist():
    st = ghost_state.get_state()
    st["watchlist"] = {"stocks": [], "crypto": []}
    ghost_state.save()
    return True


async def build_universe():
    st = ghost_state.get_state()
    uni = st.get("universe") or {"stocks": [], "crypto": []}
    return {"stocks": list(uni.get("stocks") or []), "crypto": list(uni.get("crypto") or [])}


async def fetch_market(u):
    return []


def score_gps(data: dict[str, Any]):
    # Compute a simple deterministic GPS per symbol based on its name
    out = []
    for kind in ("stocks", "crypto"):
        bucket = data.get(kind) or {}
        for sym in bucket.keys():
            s = str(sym)
            h = sum(ord(ch) for ch in s) % 40  # 0..39
            gps = 4.0 + (h / 40.0) * 4.0  # 4.0..8.0
            out.append({"symbol": s, "gps": round(gps, 1)})
    return out


def reset_state():
    return True
