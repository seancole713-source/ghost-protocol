#!/usr/bin/env python3
"""
Simulation Data Generator
Creates mock JSON responses for all UI panels
"""

import json
import os
import sys

# Add current directory to path
sys.path.insert(0, "/workspaces/GHOST")

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


def display_preview():
    """Display preview of simulation data"""

    print("\n" + "=" * 80)
    print("📊 SIMULATION MODE - DATA PREVIEW")
    print("=" * 80 + "\n")

    print("1️⃣  PORTFOLIO (/api/portfolio)")
    portfolio = get_mock_portfolio()
    print(f"    NAV: ${portfolio['nav']:,.2f}")
    print(f"    Cash: ${portfolio['cash']:,.2f}")
    print(f"    Total P&L: ${portfolio['total_pnl']:,.2f} ({portfolio['total_pnl_pct']:+.2f}%)")
    print(f"    Positions: {len(portfolio['positions'])}")
    for pos in portfolio["positions"]:
        pnl_pct = pos["pnl_pct"]
        pnl_color = "🟢" if pnl_pct > 0 else "🔴"
        value = pos["qty"] * pos["current"]
        print(f"      {pnl_color} {pos['symbol']}: {pos['qty']} shares @ ${pos['price']:.2f}")
        print(
            f"         Current: ${pos['current']:.2f} | Value: ${value:,.2f} | P&L: ${pos['pnl']:+,.2f} ({pnl_pct:+.2f}%)"
        )
    print("")

    print("2️⃣  WATCHLIST (/api/watcher/watchlist)")
    watchlist = get_mock_watchlist()
    print(f"    Tickers: {len(watchlist)}/25")
    for ticker in watchlist:
        # Map sentiment score to label
        sent = ticker["sentiment"]
        if sent > 0.6:
            sent_label = "BULLISH"
            sent_icon = "🟢"
        elif sent < 0.4:
            sent_label = "BEARISH"
            sent_icon = "🔴"
        else:
            sent_label = "NEUTRAL"
            sent_icon = "⚪"
        print(
            f"      {sent_icon} {ticker['symbol']}: GPS {ticker['gps']:.1f} | ${ticker['current_price']:.2f} ({ticker['change_pct']:+.2f}%)"
        )
        print(f"         Signal: {ticker['signal']} | Sentiment: {sent_label}")
    print("")

    print("3️⃣  48H FORECAST (/predict/48h)")
    forecast = get_mock_forecast_48h()
    print(f"    Ticker: {forecast['ticker']}")
    print(f"    Horizon: {forecast['horizon_h']} hours")
    print(f"    Step: {forecast['step_h']} hours")
    print(f"    Data Points: {len(forecast['points'])}")
    first = forecast["points"][0]
    last = forecast["points"][-1]
    price_change = ((last["price_mid"] - first["price_mid"]) / first["price_mid"]) * 100
    print("    Price Trajectory:")
    print(
        f"      Start: ${first['price_mid']:.4f} (range: ${first['price_lo']:.4f} - ${first['price_hi']:.4f})"
    )
    print(
        f"      End:   ${last['price_mid']:.4f} (range: ${last['price_lo']:.4f} - ${last['price_hi']:.4f})"
    )
    print(f"      Change: {price_change:+.2f}%")
    print("")

    print("4️⃣  TRADE CARD (/api/trade_card/WOLF)")
    card = get_mock_trade_card("WOLF", "BUY")
    print(f"    Action: {card['action']} {card['symbol']}")
    print(f"    Confidence: {card['confidence']:.1f}%")
    print(f"    Win Probability: {card['win_probability']:.1f}%")
    print("    Expected Returns:")
    print(f"      1 Day:  {card['expected_return_1d']:+.2f}%")
    print(f"      7 Days: {card['expected_return_7d']:+.2f}%")
    print(f"      30 Days: {card['expected_return_30d']:+.2f}%")
    print("    Price Targets:")
    print(f"      Target: ${card['price_target']:.2f}")
    print(f"      Stop Loss: ${card['stop_loss_price']:.2f}")
    print(
        f"      Confidence Band: ${card['confidence_band'][0]:.2f} - ${card['confidence_band'][1]:.2f}"
    )
    print("    Top Features:")
    for feat in card["top_features"][:3]:
        print(f"      • {feat['name']}: {feat['importance']:.1f}% impact → {feat['impact']}")
    print(f"    Historical Analogs: {len(card['analogs'])}")
    for analog in card["analogs"]:
        # Parse outcome string (e.g. "+4.2%")
        outcome_str = analog["outcome"]
        outcome_color = "🟢" if outcome_str.startswith("+") else "🔴"
        print(
            f"      {outcome_color} {analog['date']}: {outcome_str} (match: {analog['similarity'] * 100:.0f}%)"
        )
    print("")

    print("5️⃣  MARKET MOOD (/fusion/ai)")
    mood = get_mock_market_mood()
    mood_icons = {"BULLISH": "🟢📈", "BEARISH": "🔴📉", "NEUTRAL": "⚪➡️"}
    print(f"    {mood_icons.get(mood['sentiment'], '⚪')} Sentiment: {mood['sentiment']}")
    print(f"    Regime: {mood['regime']}")
    print(f"    Confidence: {mood['confidence'] * 100:.0f}%")
    print("    Market Indicators:")
    print(f"      VIX: {mood['vix']:.2f}")
    print(f"      SPY: {mood['spy_change']:+.2f}%")
    print("")

    print("6️⃣  NEWS FEED (/api/feeds/latest)")
    news = get_mock_news(5)
    print(f"    Latest Headlines: {len(news)}")
    for i, article in enumerate(news[:3], 1):
        sentiment_icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
            article["sentiment"], "⚪"
        )
        print(f"      {i}. {sentiment_icon} [{article['source']}] {article['title']}")
        if article.get("symbols"):
            print(f"         Symbols: {', '.join(article['symbols'])}")
    print("")

    print("7️⃣  AI PREVIEW (/ai/preview)")
    preview = get_mock_ai_preview()
    print(f"    GPS Score: {preview['gps']:.1f}/10")
    print(f"    Confidence: {preview['confidence'] * 100:.0f}%")
    print("    Reasons:")
    for reason in preview["reasons"][:3]:
        print(f"      • {reason}")
    print("    Top Features:")
    for feat, val in list(preview["features"].items())[:3]:
        print(f"      • {feat}: {val}")
    print("")

    print("8️⃣  RISK STATUS (/api/risk/status)")
    risk = get_mock_risk_status()
    trade_icon = "✅" if risk["can_trade"] else "🚫"
    print(f"    {trade_icon} Can Trade: {risk['can_trade']}")
    print(f"    Risk Level: {risk['risk_level']}")
    print(f"    Kill Switch: {'🔴 ACTIVE' if risk.get('kill_switch_active', False) else '🟢 OFF'}")
    print(
        f"    Circuit Breaker: {'🔴 TRIPPED' if risk.get('circuit_breaker_tripped', False) else '🟢 OK'}"
    )
    if risk.get("reasons"):
        print("    Reasons:")
        for reason in risk["reasons"]:
            print(f"      • {reason}")
    print("")

    print("9️⃣  TOP MOVERS (/api/top_movers)")
    movers = get_mock_top_movers(threshold=7.0)
    print(f"    Threshold: GPS ≥ {movers['threshold']}")
    print(f"    Total Count: {movers['count']}")
    print(f"    Stocks: {len(movers['stocks'])}")
    for stock in movers["stocks"][:3]:
        print(
            f"      🟢 {stock['symbol']}: GPS {stock['gps']:.1f} | ${stock['price']:.2f} ({stock['change_pct']:+.2f}%)"
        )
    if movers["crypto"]:
        print(f"    Crypto: {len(movers['crypto'])}")
        for crypto in movers["crypto"][:2]:
            print(
                f"      🟡 {crypto['symbol']}: GPS {crypto['gps']:.1f} | ${crypto['price']:.4f} ({crypto['change_pct']:+.2f}%)"
            )
    print("")

    print("=" * 80 + "\n")


def save_to_json():
    """Save all mock data to JSON file"""

    print("💾 Saving mock data to JSON...")

    mock_data = {
        "portfolio": get_mock_portfolio(),
        "watchlist": {
            "tickers": get_mock_watchlist(),
            "count": 5,
            "max_capacity": 25,
            "timestamp": int(__import__("time").time()),
        },
        "forecast": get_mock_forecast_48h(),
        "trade_card": get_mock_trade_card("WOLF", "BUY"),
        "market_mood": get_mock_market_mood(),
        "news": {
            "articles": get_mock_news(20),
            "count": 20,
            "timestamp": int(__import__("time").time()),
        },
        "ai_preview": get_mock_ai_preview(),
        "risk_status": get_mock_risk_status(),
        "top_movers": get_mock_top_movers(threshold=7.0),
    }

    # Save to public directory
    os.makedirs("public", exist_ok=True)
    output_file = "public/simulation_data.json"

    with open(output_file, "w") as f:
        json.dump(mock_data, f, indent=2)

    print(f"  ✅ Saved to: {output_file}")
    print(f"  📊 Total endpoints: {len(mock_data)}")
    print(f"  📏 File size: {os.path.getsize(output_file):,} bytes")
    print("")


def main():
    print("\n" + "=" * 80)
    print("🔧 GHOST SIMULATION MODE - DATA GENERATOR")
    print("=" * 80)
    print("Tag: ghost_ui_full_simulation_test_v1")
    print("=" * 80 + "\n")

    # Display preview
    display_preview()

    # Save to JSON
    save_to_json()

    print("=" * 80)
    print("✅ SIMULATION DATA GENERATED")
    print("=" * 80)
    print("")
    print("📋 NEXT STEPS - Choose Integration Method:")
    print("")
    print("🅰️  OPTION A: Frontend JavaScript Integration (EASIEST)")
    print("   1. Open static/ghost.js")
    print("   2. Add at top of initCockpit():")
    print("      const urlParams = new URLSearchParams(window.location.search);")
    print("      if (urlParams.get('sim') === '1') {")
    print("        return loadSimulationData();  // Load from simulation_data.json")
    print("      }")
    print("   3. Test URL: http://localhost:5000/cockpit.html?sim=1")
    print("")
    print("🅱️  OPTION B: Backend API Integration (REQUIRES RESTART)")
    print("   1. Stop server: Ctrl+C in terminal")
    print("   2. Run: bash start_simulation_mode.sh")
    print("   3. Server will route all API calls to mock data")
    print("   4. Test URL: http://localhost:5000/cockpit.html")
    print("")
    print("©️  OPTION C: Manual Browser Injection (FOR TESTING)")
    print("   1. Open: http://localhost:5000/cockpit.html")
    print("   2. Open DevTools (F12) → Console")
    print("   3. Paste:")
    print("      fetch('/simulation_data.json')")
    print("        .then(r => r.json())")
    print("        .then(data => window.GHOST_SIM_DATA = data);")
    print("   4. Refresh panels to see mock data")
    print("")
    print("=" * 80)
    print("")
    print("📊 SIMULATION DATA FILE:")
    print("   Location: public/simulation_data.json")
    print("   URL: http://localhost:5000/simulation_data.json")
    print("")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
