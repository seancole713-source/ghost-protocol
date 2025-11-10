#!/usr/bin/env python3
"""
GHOST Simulation Mode Activator
Injects mock data into API endpoints for UI validation
"""

import os
import sys

# Set simulation mode environment variable
os.environ["GHOST_SIM_MODE"] = "1"
os.environ["SIM_MODE"] = "1"

print("\n" + "=" * 80)
print("🔧 ACTIVATING GHOST FULL SIMULATION MODE")
print("=" * 80)
print("Goal: Validate all UI panels with synthetic data")
print("Tag: ghost_ui_full_simulation_test_v1")
print("=" * 80 + "\n")

# Import simulation module
from simulation_mode import (
    get_mock_ai_preview,
    get_mock_forecast_48h,
    get_mock_market_mood,
    get_mock_news,
    get_mock_portfolio,
    get_mock_risk_status,
    get_mock_top_movers,
    get_mock_trade_card,
    get_mock_watchlist,
    log_simulation,
)

# Monkey-patch the server endpoints
print("[SIMULATION] Injecting mock data providers into API endpoints...")

try:
    # Import the wolf_app module
    import wolf_app

    # Store original functions
    _original_portfolio = None
    _original_watchlist = None
    _original_forecast = None

    # Override portfolio endpoint
    if hasattr(wolf_app, "api_portfolio"):
        _original_portfolio = wolf_app.api_portfolio

        async def mock_api_portfolio():
            log_simulation("API call: /api/portfolio")
            data = get_mock_portfolio()
            # Enforce exactly 3 positions and add metadata
            positions = data.get("positions", [])
            response = {
                "positions": positions[:3],
                "count": len(positions[:3]),
                "cash": data.get("cash"),
                "nav": data.get("nav"),
                "total_pnl": data.get("total_pnl"),
                "total_pnl_pct": data.get("total_pnl_pct"),
                "simulation": True,
                "tag": os.getenv("SIM_TAG", "ghost_ui_full_simulation_test_v2"),
            }
            return response

        wolf_app.api_portfolio = mock_api_portfolio
        print("  ✓ Portfolio endpoint patched")

    # Override watchlist endpoint
    if hasattr(wolf_app, "api_watcher_get_watchlist"):
        _original_watchlist = wolf_app.api_watcher_get_watchlist

        async def mock_api_watchlist():
            log_simulation("API call: /api/watcher/watchlist")
            tickers = get_mock_watchlist()
            return {
                "tickers": tickers,
                "count": len(tickers),
                "max_capacity": 25,
                "timestamp": int(__import__("time").time()),
            }

        wolf_app.api_watcher_get_watchlist = mock_api_watchlist
        print("  ✓ Watchlist endpoint patched")

    # Override forecast endpoint
    if hasattr(wolf_app, "predict_48h"):
        _original_forecast = wolf_app.predict_48h

        async def mock_predict_48h():
            log_simulation("API call: /predict/48h")
            return get_mock_forecast_48h()

        wolf_app.predict_48h = mock_predict_48h
        print("  ✓ Forecast endpoint patched")

    # Override AI preview endpoint
    if hasattr(wolf_app, "ai_preview"):

        async def mock_ai_preview():
            log_simulation("API call: /ai/preview")
            return get_mock_ai_preview()

        wolf_app.ai_preview = mock_ai_preview
        print("  ✓ AI Preview endpoint patched")

    # Override trade card endpoint
    if hasattr(wolf_app, "api_trade_card"):

        async def mock_trade_card(symbol: str, action: str = "BUY", lookback_days: int = 90):
            log_simulation(f"API call: /api/trade_card/{symbol}")
            return get_mock_trade_card(symbol, action)

        wolf_app.api_trade_card = mock_trade_card
        print("  ✓ Trade Card endpoint patched")

    # Override market mood endpoint
    if hasattr(wolf_app, "fusion_ai"):

        async def mock_fusion_ai():
            log_simulation("API call: /fusion/ai")
            return get_mock_market_mood()

        wolf_app.fusion_ai = mock_fusion_ai
        print("  ✓ Market Mood endpoint patched")

    # Override news feed endpoint
    if hasattr(wolf_app, "api_get_latest_articles"):

        async def mock_news_feed(limit: int = 20, symbol: str | None = None):
            log_simulation(f"API call: /api/feeds/latest (limit={limit})")
            articles = get_mock_news(limit)
            return {
                "articles": articles,
                "count": len(articles),
                "symbol": symbol or "all",
                "timestamp": int(__import__("time").time()),
            }

        wolf_app.api_get_latest_articles = mock_news_feed
        print("  ✓ News Feed endpoint patched")

    # Override risk status endpoint
    if hasattr(wolf_app, "api_risk_status"):

        async def mock_risk_status(symbol: str = "WOLF"):
            log_simulation("API call: /api/risk/status")
            return get_mock_risk_status()

        wolf_app.api_risk_status = mock_risk_status
        print("  ✓ Risk Status endpoint patched")

    # Override top movers endpoint
    if hasattr(wolf_app, "api_top_movers"):

        async def mock_top_movers(threshold: float = 7.0, limit: int = 20):
            log_simulation(f"API call: /api/top_movers (threshold={threshold})")
            return get_mock_top_movers(threshold)

        wolf_app.api_top_movers = mock_top_movers
        print("  ✓ Top Movers endpoint patched")

    print("\n" + "=" * 80)
    print("✅ SIMULATION MODE ACTIVE")
    print("=" * 80)
    print("All panels now running mock data for validation.")
    print("SSE streams will update with simulated data every 5 seconds.")
    print("=" * 80 + "\n")

    print("📱 Test URLs:")
    print("   http://localhost:5000/cockpit.html")
    print("   http://localhost:5000/bank.html")
    print("   http://localhost:5000/markets.html")
    print("   http://localhost:5000/engine.html")
    print("\n" + "=" * 80 + "\n")

except Exception as e:
    print(f"❌ ERROR: Failed to activate simulation mode: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
