#!/usr/bin/env python3
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


async def test_pepe():
    from core.crypto.crypto_providers import get_crypto_price_quorum
    from wolf_app import _get_crypto_engine

    print("\n" + "=" * 70)
    print("🤖 GHOST REAL INTELLIGENCE TEST - PEPE $1000 Investment")
    print("=" * 70 + "\n")

    # STEP 1: Get real-time price
    print("📊 Fetching PEPE real-time data...")
    price_data = await get_crypto_price_quorum("PEPE", use_cache=False)
    price = price_data["price"]
    change_24h = price_data.get("change_24h_pct", 0)
    volume = price_data.get("volume_24h", 0)

    print(f"  Price: ${price:.8f}")
    print(f"  24h Change: {change_24h:+.2f}%")
    print(f"  Volume: ${volume:,.0f}\n")

    # STEP 2: Run prediction engine
    print("🎯 Running Ghost's prediction engine...")
    engine = _get_crypto_engine()
    prediction = await engine.generate_prediction("PEPE")

    print(f"  Method: {prediction.get('method')}")
    print(f"  Direction: {prediction.get('direction')}")
    print(f"  Confidence: {prediction.get('confidence', 0) * 100:.1f}%")
    print(f"  Volatility: {prediction.get('volatility')}\n")

    # STEP 3: Calculate Ghost's confidence
    pred_conf = prediction.get("confidence", 0.5)
    momentum = abs(change_24h) / 10
    volume_bonus = 0.1 if volume > 1_000_000_000 else 0.05
    regime_bonus = 0.1 if change_24h > 5 else 0

    total_conf = min(pred_conf + momentum + volume_bonus + regime_bonus, 1.0)

    print("🧠 Ghost's Confidence Calculation:")
    print(f"  Base: {pred_conf * 100:.1f}%")
    print(f"  Momentum: +{momentum * 100:.1f}%")
    print(f"  Volume: +{volume_bonus * 100:.1f}%")
    print(f"  Regime: +{regime_bonus * 100:.1f}%")
    print(f"  TOTAL: {total_conf * 100:.1f}%\n")

    # STEP 4: Investment calculation
    investment = 1000
    coins = investment / price

    print("💰 Investment Analysis:")
    print("  Investment: $1,000")
    print(f"  PEPE Coins: {coins:,.0f}\n")

    # STEP 5: 30-day scenarios
    if prediction.get("direction") == "UP":
        conservative = total_conf * 0.15
        moderate = total_conf * 0.25
        optimistic = total_conf * 0.40
    else:
        conservative = -0.05
        moderate = -0.10
        optimistic = -0.15

    print("📈 30-DAY PROFIT SCENARIOS:")
    for name, gain in [
        ("Conservative", conservative),
        ("Moderate", moderate),
        ("Optimistic", optimistic),
    ]:
        new_price = price * (1 + gain)
        value = coins * new_price
        profit = value - investment
        print(f"  {name} ({gain * 100:+.1f}%):")
        print(f"    New Price: ${new_price:.8f}")
        print(f"    Your Value: ${value:,.2f}")
        print(f"    Profit: ${profit:+,.2f}\n")

    # STEP 6: Recommendation
    if total_conf >= 0.70:
        action = "🟢 BUY"
        size = "3% of portfolio"
    elif total_conf >= 0.60:
        action = "🟡 CONSIDER"
        size = "2% of portfolio"
    else:
        action = "🔴 WAIT"
        size = "Hold off"

    print("🎯 GHOST'S RECOMMENDATION:")
    print(f"  Action: {action}")
    print(f"  Position Size: {size}")
    print(f"  Entry: ${price:.8f}")
    print(f"  Stop Loss: ${price * 0.92:.8f} (-8%)")
    print(f"  Target: ${price * (1 + moderate):.8f} ({moderate * 100:+.1f}%)\n")

    print("⚠️  RISKS:")
    print("  - Meme coin (high speculation)")
    print(f"  - Volatility: {prediction.get('volatility')}")
    print(f"  - 24h swing: {change_24h:+.2f}%")

    print("\n" + "=" * 70)
    print("THIS IS REAL INTELLIGENCE - NOT CANNED RESPONSES!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_pepe())
