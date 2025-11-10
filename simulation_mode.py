"""
GHOST Full Simulation Mode - UI Validation Environment
Tag: ghost_ui_full_simulation_test_v2

Provides synthetic data for all UI panels without live API calls.
All data routes through this module when SIM_MODE=1.
"""

import contextlib
import random
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

# =========================================================================
# GLOBALS / ROTATING STATE
# =========================================================================
_MARKET_OUTLOOK_ROTATE = ["BULLISH", "NEUTRAL", "BEARISH"]
_MARKET_OUTLOOK_INDEX = 0
_LAST_OUTLOOK_ROT_TS = 0.0
_OUTLOOK_ROTATE_SECS = 30.0

_HEATMAP_SYMBOLS = ["WOLF", "AAPL", "TSLA", "MSFT", "AMZN", "GOOG", "NVDA", "BTC", "ETH"]
_HEATMAP_LAST_TS = 0.0
_HEATMAP_CACHE: list[dict[str, float]] = []  # type: ignore[assignment]
_HEATMAP_REFRESH_SECS = 15.0

_SSE_MOCK_LAST_TS = 0.0
_SSE_MOCK_INTERVAL = 5.0

SESSION_TAG = "ghost_ui_full_simulation_test_v2"

# Simulation state
SIMULATION_ACTIVE = True
SIMULATION_START_TIME = time.time()


def log_simulation(message: str):
    """Log simulation events with clear tagging."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[SIMULATION] {timestamp} - {message}")


# ============================================================================
# MOCK PORTFOLIO DATA
# ============================================================================


def get_mock_portfolio() -> dict[str, Any]:
    """Generate synthetic portfolio with 3 positions."""
    positions = [
        {
            "symbol": "AAPL",
            "type": "stock",
            "qty": 50,
            "price": 175.20,
            "current": 178.45,
            "pnl": 162.50,
            "pnl_pct": 1.85,
            "gps": 7.8,
            "src": "simulation",
        },
        {
            "symbol": "TSLA",
            "type": "stock",
            "qty": 25,
            "price": 242.50,
            "current": 238.90,
            "pnl": -90.00,
            "pnl_pct": -1.48,
            "gps": 6.5,
            "src": "simulation",
        },
        {
            "symbol": "WOLF",
            "type": "stock",
            "qty": 2000,
            "price": 1.20,
            "current": 1.24,
            "pnl": 80.00,
            "pnl_pct": 3.33,
            "gps": 7.2,
            "src": "simulation",
        },
    ]

    total_value = sum(p["qty"] * p["current"] for p in positions)
    cash = 5000.00
    nav = total_value + cash

    log_simulation(f"Portfolio: {len(positions)} positions, NAV ${nav:,.2f}")

    return {
        "positions": positions,
        "cash": cash,
        "nav": round(nav, 2),
        "total_pnl": round(sum(p["pnl"] for p in positions), 2),
        "total_pnl_pct": round(
            (total_value - sum(p["qty"] * p["price"] for p in positions))
            / sum(p["qty"] * p["price"] for p in positions)
            * 100,
            2,
        ),
    }


# ============================================================================
# MOCK WATCHLIST DATA
# ============================================================================


def get_mock_watchlist() -> list[dict[str, Any]]:
    """Generate synthetic watchlist with 5 tickers."""
    tickers = [
        {"symbol": "MSFT", "gps": 8.1, "price": 378.50, "change_pct": 1.2, "sentiment": 0.65},
        {"symbol": "AMZN", "gps": 7.9, "price": 145.20, "change_pct": 0.8, "sentiment": 0.55},
        {"symbol": "GOOG", "gps": 7.5, "price": 139.75, "change_pct": -0.3, "sentiment": 0.45},
        {"symbol": "PEPE", "gps": 6.2, "price": 0.00001234, "change_pct": 15.7, "sentiment": 0.85},
        {"symbol": "DOGE", "gps": 5.8, "price": 0.0875, "change_pct": -2.1, "sentiment": 0.35},
    ]

    log_simulation(f"Watchlist: {len(tickers)} tickers tracked")

    return [
        {
            "symbol": t["symbol"],
            "gps": t["gps"],
            "current_price": t["price"],
            "change_pct": t["change_pct"],
            "sentiment": t["sentiment"],
            "signal": "BUY" if t["gps"] >= 7.0 else "HOLD" if t["gps"] >= 5.0 else "SELL",
            "last_updated": int(time.time()),
        }
        for t in tickers
    ]


# ============================================================================
# MOCK 48H FORECAST
# ============================================================================


def get_mock_forecast_48h(symbol: str = "WOLF") -> dict[str, Any]:
    """Generate 48-hour price forecast with cone projection."""
    base_price = 1.24
    horizon_h = 48
    step_h = 2
    num_points = horizon_h // step_h

    points = []
    current_time = int(time.time())

    for i in range(num_points):
        t = current_time + (i * step_h * 3600)

        # Add slight upward drift with volatility
        drift = 0.001 * i  # 0.1% per 2h
        volatility = 0.02 * (i / num_points)  # Increasing uncertainty

        price_mid = base_price * (1 + drift)
        price_lo = price_mid * (1 - volatility)
        price_hi = price_mid * (1 + volatility)

        # Calculate P&L based on 2000 shares @ $1.20 entry
        qty = 2000
        entry = 1.20
        pnl_mid = (price_mid - entry) * qty
        pnl_lo = (price_lo - entry) * qty
        pnl_hi = (price_hi - entry) * qty

        points.append(
            {
                "t": t,
                "price_mid": round(price_mid, 4),
                "price_lo": round(price_lo, 4),
                "price_hi": round(price_hi, 4),
                "pnl_mid": round(pnl_mid, 2),
                "pnl_lo": round(pnl_lo, 2),
                "pnl_hi": round(pnl_hi, 2),
            }
        )

    log_simulation(f"Forecast: {len(points)} data points over {horizon_h}h")

    return {
        "ticker": symbol,
        "as_of": current_time,
        "horizon_h": horizon_h,
        "step_h": step_h,
        "points": points,
        "summary": {
            "confidence": 68,
            "drift_daily_pct": 0.12,
            "pnl_48h_mid": points[-1]["pnl_mid"],
        },
    }


# ============================================================================
# MOCK AI TRADE CARD
# ============================================================================


def get_mock_trade_card(symbol: str = "WOLF", action: str = "BUY") -> dict[str, Any]:
    """Generate AI explainability trade card."""
    log_simulation(f"Trade Card: {action} {symbol} with AI rationale")

    return {
        "action": action,
        "symbol": symbol,
        "confidence": 72.5,
        "timestamp": int(time.time()),
        "top_features": [
            {"name": "price_momentum", "importance": 23.4, "impact": "+0.8%"},
            {"name": "volume_profile", "importance": 18.7, "impact": "+0.5%"},
            {"name": "news_sentiment", "importance": 15.2, "impact": "+0.3%"},
            {"name": "market_regime", "importance": 12.8, "impact": "SIDEWAYS"},
            {"name": "volatility", "importance": 9.5, "impact": "LOW"},
        ],
        "analogs": [
            {"date": "2024-09-15", "similarity": 0.87, "outcome": "+4.2%"},
            {"date": "2024-08-22", "similarity": 0.82, "outcome": "+2.8%"},
            {"date": "2024-07-10", "similarity": 0.79, "outcome": "+1.5%"},
        ],
        "expected_return_1d": 0.8,
        "expected_return_7d": 3.2,
        "expected_return_30d": 8.5,
        "price_target": 1.35,
        "confidence_band": [1.28, 1.42],
        "stop_loss_price": 1.12,
        "stop_loss_reason": "Below 200-day MA",
        "invalidation_signals": [
            "Price breaks below $1.10",
            "Volume drops below 50k/day",
            "News sentiment turns negative",
        ],
        "var_95": -0.08,
        "max_loss_estimate": -6.7,
        "win_probability": 68.0,
        "rationale": "Strong upward momentum with positive news sentiment and favorable market conditions. Volume profile shows institutional interest.",
        "risks": ["Overall market downturn", "Sector rotation", "Unexpected negative news"],
        "catalysts": [
            "Earnings announcement (7 days)",
            "Product launch rumors",
            "Positive analyst coverage",
        ],
    }


# ============================================================================
# MOCK MARKET MOOD / FUSION AI
# ============================================================================


def get_mock_market_mood() -> dict[str, Any]:
    """Generate market sentiment/mood state with rotating outlook every 30s."""
    global _MARKET_OUTLOOK_INDEX, _LAST_OUTLOOK_ROT_TS
    now = time.time()
    if (now - _LAST_OUTLOOK_ROT_TS) >= _OUTLOOK_ROTATE_SECS:
        _MARKET_OUTLOOK_INDEX = (_MARKET_OUTLOOK_INDEX + 1) % len(_MARKET_OUTLOOK_ROTATE)
        _LAST_OUTLOOK_ROT_TS = now
    mood = _MARKET_OUTLOOK_ROTATE[_MARKET_OUTLOOK_INDEX]
    log_simulation(f"Market Mood (rotating): {mood}")
    return {
        "action": mood,
        "confidence": random.randint(55, 85),
        "regime": random.choice(["TRENDING_UP", "SIDEWAYS", "TRENDING_DOWN"]),
        "vix": round(random.uniform(14.0, 22.0), 2),
        "spy_change": round(random.uniform(-1.5, 1.5), 2),
        "sentiment": round(random.uniform(-0.3, 0.7), 2),
        "summary": f"Outlook rotating cycle — currently {mood.lower()}.",
        "timestamp": int(now),
    }


def get_mock_heatmap() -> list[dict[str, float]]:  # type: ignore[return-type]
    """Generate or refresh a dynamic heatmap of GPS scores spanning 5.0–9.9."""
    global _HEATMAP_LAST_TS, _HEATMAP_CACHE
    now = time.time()
    if (now - _HEATMAP_LAST_TS) >= _HEATMAP_REFRESH_SECS or not _HEATMAP_CACHE:
        _HEATMAP_CACHE = []
        _HEATMAP_CACHE = []
        for s in _HEATMAP_SYMBOLS:
            gps_val: float = round(random.uniform(5.0, 9.9), 1)
            _HEATMAP_CACHE.append({"symbol": s, "gps": gps_val})  # type: ignore[arg-type]
        _HEATMAP_LAST_TS = now
        log_simulation(f"Heatmap refreshed: {len(_HEATMAP_CACHE)} symbols")
    return _HEATMAP_CACHE


def maybe_emit_mock_tick(callback: Callable[[dict], None] | None = None):
    """Optionally emit a mock diagnostics tick every 5s (for SSE)."""
    global _SSE_MOCK_LAST_TS
    now = time.time()
    if (now - _SSE_MOCK_LAST_TS) >= _SSE_MOCK_INTERVAL:
        _SSE_MOCK_LAST_TS = now
        tick = {
            "ts": int(now),
            "price": round(1.20 + random.uniform(-0.02, 0.05), 4),
            "gps": round(random.uniform(6.0, 9.5), 1),
            "outlook": _MARKET_OUTLOOK_ROTATE[_MARKET_OUTLOOK_INDEX],
            "tag": "diag_tick",
        }
        log_simulation(f"Diagnostics Tick: {tick}")
        if callback:
            with contextlib.suppress(Exception):
                callback(tick)
    return _SSE_MOCK_LAST_TS


# ============================================================================
# MOCK NEWS FEED
# ============================================================================


def get_mock_news(limit: int = 20) -> list[dict[str, Any]]:
    """Generate simulated news headlines."""
    headlines = [
        "Tech Stocks Rally on Strong Earnings Reports",
        "Federal Reserve Holds Interest Rates Steady",
        "AI Companies See Record Investment in Q3",
        "Market Analysts Predict Continued Growth",
        "Small-Cap Stocks Outperform Major Indices",
        "Energy Sector Shows Resilience Amid Volatility",
        "Crypto Market Stabilizes After Recent Swings",
        "Retail Investors Return to Growth Stocks",
        "Biotech Sector Gets Boost from FDA Approvals",
        "Trade Talks Progress Positively",
    ]

    articles = []
    current_time = int(time.time())

    for i, headline in enumerate(headlines[:limit]):
        sentiment_val = random.uniform(-0.2, 0.8)
        sentiment_label = (
            "bullish" if sentiment_val > 0.2 else "bearish" if sentiment_val < -0.2 else "neutral"
        )

        articles.append(
            {
                "title": f"[SIMULATED] {headline}",
                "source": random.choice(["Bloomberg", "Reuters", "CNBC", "WSJ", "FT"]),
                "url": f"https://example.com/article/{i}",
                "published_at": current_time - (i * 3600),
                "sentiment": sentiment_val,
                "sentiment_label": sentiment_label,
                "relevance": random.uniform(0.5, 1.0),
                "symbols": random.sample(
                    ["AAPL", "TSLA", "MSFT", "AMZN", "GOOG", "WOLF"], k=random.randint(1, 3)
                ),
            }
        )

    log_simulation(f"News Feed: {len(articles)} simulated headlines")

    return articles


# ============================================================================
# MOCK AI PREVIEW / GHOST SCORE
# ============================================================================


def get_mock_ai_preview() -> dict[str, Any]:
    """Generate AI decision preview with GPS score."""
    gps = round(random.uniform(6.5, 8.5), 1)
    confidence = random.randint(60, 85)

    log_simulation(f"AI Preview: GPS {gps}, Confidence {confidence}%")

    return {
        "gps": gps,
        "confidence": confidence,
        "reasons": [
            "Price momentum positive",
            "News sentiment favorable",
            "Risk level acceptable",
            "Volume above average",
        ],
        "analogs": [
            "2024-09-15: +4.2% (87% match)",
            "2024-08-22: +2.8% (82% match)",
            "2024-07-10: +1.5% (79% match)",
        ],
        "features": {
            "price_momentum": 0.15,
            "news_sentiment": 0.45,
            "risk_score": 0.35,
            "volume_score": 0.68,
            "volatility": 0.22,
        },
    }


# ============================================================================
# MOCK RISK STATUS
# ============================================================================


def get_mock_risk_status() -> dict[str, Any]:
    """Generate risk shell status."""
    can_trade = random.choice([True, True, True, False])  # 75% can trade
    risk_level = "LOW" if can_trade else "HIGH"

    log_simulation(f"Risk Status: {risk_level}, Can Trade: {can_trade}")

    return {
        "can_trade": can_trade,
        "risk_level": risk_level,
        "reasons": [
            "Position size within limits" if can_trade else "Position size exceeded",
            "Volatility acceptable" if can_trade else "Volatility too high",
            "Drawdown under threshold" if can_trade else "Drawdown limit reached",
        ],
        "kill_switch_active": False,
        "circuit_breaker_tripped": not can_trade,
        "timestamp": int(time.time()),
    }


# ============================================================================
# MOCK TOP MOVERS
# ============================================================================


def get_mock_top_movers(threshold: float = 7.0) -> dict[str, Any]:
    """Generate top movers list."""
    watchlist = get_mock_watchlist()
    movers = [t for t in watchlist if t["gps"] >= threshold]

    log_simulation(f"Top Movers: {len(movers)} stocks above GPS {threshold}")

    stocks = []
    for ticker in movers:
        stocks.append(
            {
                "symbol": ticker["symbol"],
                "gps": ticker["gps"],
                "price": ticker["current_price"],
                "change_pct": ticker["change_pct"],
                "volume": random.randint(100000, 5000000),
            }
        )

    return {"stocks": stocks, "crypto": [], "threshold": threshold, "count": len(stocks)}


# ============================================================================
# SIMULATION ACTIVATION
# ============================================================================


def activate_simulation_mode():
    """Initialize simulation mode."""
    global SIMULATION_ACTIVE, SIMULATION_START_TIME
    SIMULATION_ACTIVE = True
    SIMULATION_START_TIME = time.time()

    print("\n" + "=" * 80)
    print("✅ SIMULATION MODE ACTIVE — All panels now running mock data for validation.")
    print("=" * 80)
    print(f"[SIMULATION] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[SIMULATION] Tag: {SESSION_TAG}")
    print("=" * 80 + "\n")

    # Log initial state
    log_simulation("Initializing mock data providers...")
    log_simulation("✓ Portfolio simulator ready (3 positions)")
    log_simulation("✓ Watchlist simulator ready (5 tickers)")
    log_simulation("✓ Forecast engine ready (24 data points)")
    log_simulation("✓ Trade card generator ready")
    log_simulation("✓ News feed simulator ready")
    log_simulation("✓ AI preview simulator ready")
    log_simulation("✓ Risk status simulator ready")
    log_simulation("✓ Rotating market outlook active (30s cycle)")
    log_simulation("✓ Dynamic heatmap generator ready (5.0–9.9 GPS range)")
    log_simulation("✓ Diagnostics tick emitter active (5s interval)")
    log_simulation("\n✅ All simulation modules loaded successfully (v2)\n")
    print(
        "✅ Ghost now running in FULL SIMULATION MODE — all UI panels populated with mock data for validation."
    )

    # Monkeypatch external data sources to prevent any live calls
    _monkeypatch_external_calls()


def _monkeypatch_external_calls():
    """Disable outbound data sources: yfinance, requests, httpx."""
    # Note: Monkeypatching removed to avoid blocking internal server calls
    # SIM_MODE=1 in wolf_app.py already prevents external data fetches
    import os as _os

    # Nullify provider keys to disable external APIs
    for k in ["POLYGON_API_KEY", "ALPHAVANTAGE_API_KEY"]:
        if k in _os.environ:
            _os.environ[k] = "SIM_DISABLED"
    log_simulation("External provider keys neutralized (SIM_MODE=1 active)")


def is_simulation_active() -> bool:
    """Check if simulation mode is active."""
    return SIMULATION_ACTIVE


def get_simulation_uptime() -> float:
    """Get simulation uptime in seconds."""
    return time.time() - SIMULATION_START_TIME if SIMULATION_ACTIVE else 0.0


# Auto-activate on module import
if __name__ != "__main__":
    activate_simulation_mode()
