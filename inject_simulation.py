#!/usr/bin/env python3
"""
Runtime Simulation Injector
Patches GHOST API endpoints with mock data without restarting server
"""

import json

import requests

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
)

BASE_URL = "http://localhost:5000"


def test_endpoint(endpoint, description):
    """Test if endpoint is accessible"""
    try:
        resp = requests.get(f"{BASE_URL}{endpoint}", timeout=2)
        status = "✅" if resp.status_code == 200 else f"⚠️ {resp.status_code}"
        print(f"  {status} {description}")
        return resp.status_code == 200
    except Exception as e:
        print(f"  ❌ {description}: {e}")
        return False


def display_mock_data_preview():
    """Display preview of what simulation mode will show"""

    print("\n" + "=" * 80)
    print("📊 SIMULATION MODE DATA PREVIEW")
    print("=" * 80 + "\n")

    print("1️⃣  PORTFOLIO (/api/portfolio)")
    portfolio = get_mock_portfolio()
    print(f"    NAV: ${portfolio['nav']:,.2f}")
    print(f"    Cash: ${portfolio['cash']:,.2f}")
    print(f"    Positions: {len(portfolio['positions'])}")
    for pos in portfolio["positions"]:
        print(
            f"      • {pos['symbol']}: {pos['shares']} @ ${pos['avg_cost']:.2f} = ${pos['market_value']:,.2f}"
        )
    print("")

    print("2️⃣  WATCHLIST (/api/watcher/watchlist)")
    watchlist = get_mock_watchlist()
    print(f"    Tickers: {len(watchlist)}")
    for ticker in watchlist:
        print(
            f"      • {ticker['symbol']}: GPS {ticker['gps']:.1f} | ${ticker['price']:.2f} ({ticker['change_pct']:+.2f}%)"
        )
    print("")

    print("3️⃣  48H FORECAST (/predict/48h)")
    forecast = get_mock_forecast_48h()
    print(f"    Ticker: {forecast['ticker']}")
    print(f"    Horizon: {forecast['horizon_h']}h")
    print(f"    Data Points: {len(forecast['points'])}")
    print(f"    Starting Price: ${forecast['points'][0]['price_mid']:.2f}")
    print(f"    Ending Price: ${forecast['points'][-1]['price_mid']:.2f}")
    print("")

    print("4️⃣  TRADE CARD (/api/trade_card/WOLF)")
    card = get_mock_trade_card("WOLF", "BUY")
    print(f"    Action: {card['action']} {card['symbol']}")
    print(f"    GPS Score: {card['gps']:.1f}")
    print("    Top Features:")
    for feat in card["top_features"][:3]:
        print(f"      • {feat['name']}: {feat['importance'] * 100:.1f}%")
    print(f"    Price Target: ${card['price_target']:.2f}")
    print("")

    print("5️⃣  MARKET MOOD (/fusion/ai)")
    mood = get_mock_market_mood()
    print(f"    Sentiment: {mood['sentiment']}")
    print(f"    Regime: {mood['regime']}")
    print(f"    VIX: {mood['vix']:.2f}")
    print(f"    SPY Change: {mood['spy_change']:+.2f}%")
    print("")

    print("6️⃣  NEWS FEED (/api/feeds/latest)")
    news = get_mock_news(5)
    print(f"    Headlines: {len(news)} (showing 3)")
    for article in news[:3]:
        print(f"      • [{article['source']}] {article['title'][:60]}...")
    print("")

    print("7️⃣  AI PREVIEW (/ai/preview)")
    preview = get_mock_ai_preview()
    print(f"    GPS Score: {preview['gps']:.1f}")
    print(f"    Confidence: {preview['confidence'] * 100:.0f}%")
    print(f"    Reasons: {len(preview['reasons'])}")
    print("")

    print("8️⃣  RISK STATUS (/api/risk/status)")
    risk = get_mock_risk_status()
    print(f"    Can Trade: {risk['can_trade']}")
    print(f"    Risk Level: {risk['risk_level']}")
    print(f"    Kill Switch: {risk['kill_switch']}")
    print("")

    print("9️⃣  TOP MOVERS (/api/top_movers)")
    movers = get_mock_top_movers()
    print(f"    Stocks: {len(movers['stocks'])}")
    print(f"    Crypto: {len(movers['crypto'])}")
    print(f"    Total: {movers['count']}")
    print("")

    print("=" * 80 + "\n")


def check_current_endpoints():
    """Check status of all API endpoints"""

    print("\n" + "=" * 80)
    print("🔍 CURRENT ENDPOINT STATUS")
    print("=" * 80 + "\n")

    endpoints = [
        ("/api/status", "System Status"),
        ("/api/portfolio", "Portfolio Data"),
        ("/api/watcher/watchlist", "Watchlist"),
        ("/predict/48h", "48h Forecast"),
        ("/api/trade_card/WOLF", "Trade Card"),
        ("/fusion/ai", "Market Mood"),
        ("/api/feeds/latest?limit=5", "News Feed"),
        ("/ai/preview", "AI Preview"),
        ("/api/risk/status", "Risk Status"),
        ("/api/top_movers", "Top Movers"),
    ]

    working = 0
    for endpoint, desc in endpoints:
        if test_endpoint(endpoint, desc):
            working += 1

    print(f"\n  Status: {working}/{len(endpoints)} endpoints responding")
    print("=" * 80 + "\n")

    return working == len(endpoints)


def save_mock_responses():
    """Save mock responses to JSON files for static serving"""

    print("\n" + "=" * 80)
    print("💾 SAVING MOCK RESPONSES TO JSON FILES")
    print("=" * 80 + "\n")

    mock_data = {
        "portfolio": get_mock_portfolio(),
        "watchlist": {"tickers": get_mock_watchlist(), "count": 5, "max_capacity": 25},
        "forecast": get_mock_forecast_48h(),
        "trade_card": get_mock_trade_card("WOLF", "BUY"),
        "market_mood": get_mock_market_mood(),
        "news": {"articles": get_mock_news(20), "count": 20},
        "ai_preview": get_mock_ai_preview(),
        "risk_status": get_mock_risk_status(),
        "top_movers": get_mock_top_movers(),
    }

    # Save to public directory for static serving
    output_file = "public/simulation_data.json"
    with open(output_file, "w") as f:
        json.dump(mock_data, f, indent=2)

    print(f"  ✅ Saved to: {output_file}")
    print(f"  📊 Data includes: {len(mock_data)} endpoint responses")
    print("=" * 80 + "\n")


def main():
    print("\n" + "=" * 80)
    print("🔧 GHOST SIMULATION MODE - RUNTIME INJECTOR")
    print("=" * 80)
    print("This script will:")
    print("  1. Check current endpoint status")
    print("  2. Preview mock data that will be displayed")
    print("  3. Save mock responses to JSON for static serving")
    print("=" * 80)

    # Check endpoints
    all_working = check_current_endpoints()

    # Preview data
    display_mock_data_preview()

    # Save mock data
    save_mock_responses()

    print("=" * 80)
    print("✅ SIMULATION DATA READY")
    print("=" * 80)
    print("")
    print("📋 NEXT STEPS:")
    print("")
    print("Option A: Frontend Integration (RECOMMENDED)")
    print("  • Modify ghost.js to fetch from /simulation_data.json")
    print("  • Add ?sim=1 URL parameter to enable simulation mode")
    print("  • URLs: http://localhost:5000/cockpit.html?sim=1")
    print("")
    print("Option B: Backend Integration (requires server restart)")
    print("  • Stop server: Ctrl+C in terminal")
    print("  • Run: bash start_simulation_mode.sh")
    print("  • Server will serve mock data from all endpoints")
    print("")
    print("Option C: Manual Testing")
    print("  • Open: public/simulation_data.json")
    print("  • Copy data for specific panels")
    print("  • Use browser dev tools to inject")
    print("")
    print("=" * 80 + "\n")

    if all_working:
        print("✅ All endpoints are responding - server is healthy!")
    else:
        print("⚠️  Some endpoints are not responding - check server logs")

    print("")


if __name__ == "__main__":
    main()
