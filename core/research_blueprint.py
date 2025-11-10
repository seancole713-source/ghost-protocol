"""
Ghost Research Blueprint

Build a structured research snapshot across 12 categories for a symbol (stock or crypto),
using available local providers (yfinance, Polygon news via wolf_app helper for WOLF when applicable,
EDGAR filings via existing EDGARClient), and compute an aggregate research impact and confidence.

The snapshot is designed to be consumed by the forecast engine and the UI.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import numpy as np
import requests

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

try:
    from core.edgar_integration import EDGARClient  # type: ignore
except Exception:  # pragma: no cover
    EDGARClient = None  # type: ignore


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _compute_technicals(hist_df) -> dict[str, Any]:
    out: dict[str, Any] = {"have_history": False}
    try:
        if hist_df is None or len(hist_df) < 60:
            return out
        out["have_history"] = True
        close = hist_df["Close"].astype(float).to_numpy()

        # Moving averages
        def sma(arr, n):
            if len(arr) < n:
                return None
            return float(np.mean(arr[-n:]))

        out["ma20"] = sma(close, 20)
        out["ma50"] = sma(close, 50)
        out["ma200"] = sma(close, 200) if len(close) >= 200 else None
        # RSI (14)
        try:
            delta = np.diff(close)
            gain = np.maximum(delta, 0)
            loss = -np.minimum(delta, 0)
            n = 14
            if len(delta) >= n:
                avg_gain = np.mean(gain[-n:])
                avg_loss = np.mean(loss[-n:])
                rs = (avg_gain / avg_loss) if avg_loss > 0 else 999.0
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = None
        except Exception:
            rsi = None
        out["rsi14"] = None if rsi is None else float(rsi)
        # Bollinger (20, 2)
        try:
            n = 20
            if len(close) >= n:
                m = float(np.mean(close[-n:]))
                s = float(np.std(close[-n:], ddof=0))
                out["bb_mid"] = m
                out["bb_lo"] = m - 2.0 * s
                out["bb_hi"] = m + 2.0 * s
            else:
                out["bb_mid"] = out["bb_lo"] = out["bb_hi"] = None
        except Exception:
            out["bb_mid"] = out["bb_lo"] = out["bb_hi"] = None
    except Exception:
        pass
    return out


def _gather_yf_info(symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {"source": "yfinance", "ok": False}
    if yf is None:
        return out
    try:
        tkr = yf.Ticker(symbol)
        info = tkr.info or {}
        out["ok"] = True
        out["info"] = {
            "longName": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "marketCap": info.get("marketCap"),
            "enterpriseValue": info.get("enterpriseValue"),
            "grossMargins": info.get("grossMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "profitMargins": info.get("profitMargins"),
            "debtToEquity": info.get("debtToEquity"),
            "currentRatio": info.get("currentRatio"),
            "freeCashflow": info.get("freeCashflow"),
            "dividendYield": info.get("dividendYield"),
            "payoutRatio": info.get("payoutRatio"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToSalesTrailing12Months": info.get("priceToSalesTrailing12Months"),
            "priceToBook": info.get("priceToBook"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "beta": info.get("beta"),
            "country": info.get("country"),
            "fullTimeEmployees": info.get("fullTimeEmployees"),
        }
        # Recent history for technicals
        try:
            hist = tkr.history(period="1y", interval="1d")
        except Exception:
            hist = None
        out["technicals"] = _compute_technicals(hist)
        # Calendar (earnings dates)
        try:
            cal = tkr.calendar
            if hasattr(cal, "to_dict"):
                out["calendar"] = cal.to_dict()
        except Exception:
            pass
    except Exception:
        pass
    return out


def _gather_sec(symbol: str) -> dict[str, Any]:
    out: dict[str, Any] = {"source": "edgar", "ok": False}
    try:
        if EDGARClient is None:
            return out
        client = EDGARClient()
        filings = client.get_company_filings(symbol, limit=20)
        out["ok"] = True
        # Summaries
        recent_types = {}
        for f in filings:
            ft = getattr(f, "filing_type", "?")
            recent_types[ft] = recent_types.get(ft, 0) + 1
        out["recent_counts"] = recent_types
        out["last8k_urgency"] = None
        out["has_bankruptcy"] = False
        out["has_delisting"] = False
        out["has_product"] = False
        for f in filings:
            try:
                if getattr(f, "filing_type", "") == "8-K":
                    out["last8k_urgency"] = getattr(f, "urgency", None)
                txt = (getattr(f, "description", "") or "").lower()
                if any(k in txt for k in ["bankruptcy", "chapter 11", "chapter 7"]):
                    out["has_bankruptcy"] = True
                if "3.01" in set(getattr(f, "items", []) or []):
                    out["has_delisting"] = True
                if any(k in txt for k in ["launch", "launched", "introduc", "product"]):
                    out["has_product"] = True
            except Exception:
                pass
        return out
    except Exception:
        return out


def _aggregate_research_signal(parts: dict[str, Any]) -> dict[str, Any]:
    """Compute an aggregate impact in [-1,1] and confidence in [0,100]."""
    # Default neutral
    impact = 0.0
    conf = 60.0
    # Technicals tilt
    tech = _safe_get(parts, "yfinance", "technicals", default={}) or {}
    rsi = tech.get("rsi14")
    if isinstance(rsi, (int, float)):
        # Overbought/oversold bias
        if rsi >= 70:
            impact -= 0.15
            conf += 5
        elif rsi <= 30:
            impact += 0.15
            conf += 5
    # Leverage / liquidity tilt
    d2e = _safe_get(parts, "yfinance", "info", "debtToEquity")
    if isinstance(d2e, (int, float)):
        if d2e > 250:
            impact -= 0.1
        elif d2e < 50:
            impact += 0.05
    # SEC events tilt
    if parts.get("edgar", {}).get("has_bankruptcy"):
        impact -= 0.6
        conf += 10
    if parts.get("edgar", {}).get("has_delisting"):
        impact -= 0.5
        conf += 10
    if parts.get("edgar", {}).get("has_product"):
        impact += 0.2
    urg = parts.get("edgar", {}).get("last8k_urgency")
    if isinstance(urg, str):
        u = urg.lower()
        if "critical" in u:
            conf += 10
        elif "high" in u:
            conf += 5
    # Clamp and map
    impact = max(-1.0, min(1.0, impact))
    conf = max(30.0, min(95.0, conf))
    return {"impact": impact, "confidence": int(round(conf))}


def build_research_snapshot(symbol: str, asset_type: str = "stock") -> dict[str, Any]:
    """Return structured research snapshot and aggregate research signal.

    asset_type: 'stock' | 'crypto' (crypto gracefully limits to technicals/price)
    """
    parts: dict[str, Any] = {}
    # Company core and financials via yfinance when available (stocks)
    if asset_type == "stock" and yf is not None:
        parts["yfinance"] = _gather_yf_info(symbol)
    else:
        parts["yfinance"] = {"ok": False, "reason": "unsupported_asset_or_yf_missing"}
    # SEC filings (stocks)
    if asset_type == "stock" and EDGARClient is not None:
        parts["edgar"] = _gather_sec(symbol)
    else:
        parts["edgar"] = {"ok": False}
    # News sentiment is handled in wolf_app get_wolf_news for WOLF; generic news integration can be added per provider keys
    # Aggregate baseline (deterministic)
    agg = _aggregate_research_signal(parts)
    snapshot: dict[str, Any] = {
        "symbol": symbol,
        "as_of": int(time.time()),
        "parts": parts,
        "aggregate": agg,
    }
    # Optional: LLM enrichment to fill qualitative categories (executives, ownership, competitors, guidance)
    # Guarded by env RESEARCH_LLM_ON=1 and presence of OPENAI_API_KEY.
    try:
        if os.getenv("RESEARCH_LLM_ON", "0").lower() in ("1", "true", "yes"):
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if api_key:
                # Use canonical AGENT_MODEL (AI_MODEL remains an alias elsewhere)
                model = os.getenv(
                    "RESEARCH_LLM_MODEL", os.getenv("AGENT_MODEL", "gpt-4o-mini")
                ).strip()
                base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
                # Build a compact context from available parts to reduce token usage
                ctx_bits: list[str] = []
                yf_info = parts.get("yfinance", {}).get("info", {}) or {}
                tech = parts.get("yfinance", {}).get("technicals", {}) or {}
                edgar = parts.get("edgar", {}) or {}
                if yf_info:
                    ctx_bits.append(
                        f"Sector: {yf_info.get('sector')} | Industry: {yf_info.get('industry')} | MarketCap: {yf_info.get('marketCap')}"
                    )
                if tech:
                    ctx_bits.append(
                        f"RSI14: {tech.get('rsi14')} | MA20/50/200: {tech.get('ma20')}/{tech.get('ma50')}/{tech.get('ma200')}"
                    )
                if edgar:
                    ctx_bits.append(
                        f"EDGAR: bankruptcy={edgar.get('has_bankruptcy')} delisting={edgar.get('has_delisting')} last8k_urgency={edgar.get('last8k_urgency')}"
                    )
                context_blob = " \n".join([b for b in ctx_bits if b])[:1200]
                prompt = (
                    "You are a precise equity research assistant. Given the symbol and sparse context, "
                    "return a strict JSON object with keys: executives (top 3 names/titles), ownership (top holders or insider %), "
                    "guidance (most recent guidance change short), competitors (3 names), sector (1 sentence structure), "
                    "risks (bulleted 2-3), opportunities (bulleted 2-3), sources (array of URLs, if none use []), and a numeric confidence 0-100.\n"
                    "Only output JSON, no prose."
                )
                body = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You output only valid JSON strictly matching the requested keys.",
                        },
                        {
                            "role": "user",
                            "content": f"Symbol: {symbol} (asset: {asset_type})\nContext:\n{context_blob}",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.2,
                }
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                try:
                    resp = requests.post(
                        f"{base_url}/chat/completions", headers=headers, json=body, timeout=12
                    )
                    data = resp.json() if resp.ok else {}
                    content = (
                        (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                        if isinstance(data, dict)
                        else None
                    )
                    llm_json: dict[str, Any] = {}
                    if content:
                        # Try to extract JSON block
                        m = re.search(r"\{[\s\S]*\}", content)
                        raw = m.group(0) if m else content
                        try:
                            llm_json = json.loads(raw)
                        except Exception:
                            llm_json = {"raw": content}
                    snapshot["llm"] = {"ok": True, "model": model, "data": llm_json}
                except Exception as _e:
                    snapshot["llm"] = {"ok": False, "error": str(_e)[:200]}
            else:
                snapshot["llm"] = {"ok": False, "error": "no_api_key"}
    except Exception as e:
        snapshot["llm"] = {"ok": False, "error": str(e)[:200]}
    return snapshot
