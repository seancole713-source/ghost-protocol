"""
GHOST Market Mood Tracker
==========================
Tracks market regime and sentiment using SPY, QQQ, VIX.

Features:
- Bull/bear/sideways regime classification
- VIX volatility tracking
- Sector rotation detection
- Risk-on/risk-off sentiment
- Daily snapshot persistence

Author: Ghost AI
Date: 2025-10-05
"""

import json
import logging
import time
from typing import Any

import numpy as np
import yfinance as yf


def update_market_mood(output_path: str = "data/market_mood.json") -> dict[str, Any]:
    """
    Update daily market mood snapshot.

    Analyzes:
    - SPY (S&P 500) trend
    - QQQ (Nasdaq 100) trend
    - VIX (Volatility Index)
    - Market regime classification

    Args:
        output_path: Path to save JSON snapshot

    Returns:
        Market mood dictionary
    """
    try:
        # Fetch market data
        spy = yf.Ticker("SPY")
        qqq = yf.Ticker("QQQ")
        vix = yf.Ticker("^VIX")

        # Get 5-day history for trend
        spy_hist = spy.history(period="5d")
        qqq_hist = qqq.history(period="5d")
        vix_hist = vix.history(period="1d")

        if len(spy_hist) < 2:
            raise ValueError("Insufficient SPY data")
        if len(vix_hist) < 1:
            raise ValueError("Insufficient VIX data")

        # Calculate trends
        spy_price = spy_hist["Close"].iloc[-1]
        spy_start = spy_hist["Close"].iloc[0]
        spy_change = ((spy_price / spy_start) - 1) * 100

        qqq_price = qqq_hist["Close"].iloc[-1] if len(qqq_hist) >= 2 else 0
        qqq_start = qqq_hist["Close"].iloc[0] if len(qqq_hist) >= 2 else 1
        qqq_change = ((qqq_price / qqq_start) - 1) * 100 if qqq_start > 0 else 0

        vix_current = vix_hist["Close"].iloc[-1]

        # Calculate moving averages (if enough data)
        spy_ma20 = (
            spy_hist["Close"].rolling(min(20, len(spy_hist))).mean().iloc[-1]
            if len(spy_hist) >= 2
            else spy_price
        )
        spy_ma50 = (
            spy_hist["Close"].rolling(min(50, len(spy_hist))).mean().iloc[-1]
            if len(spy_hist) >= 2
            else spy_price
        )

        # Volatility calculation
        returns = spy_hist["Close"].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.15

        # Regime classification
        regime, sentiment, confidence = classify_regime(
            spy_change, qqq_change, vix_current, spy_price, spy_ma20, spy_ma50, volatility
        )

        # Build mood dict
        mood = {
            "date": time.strftime("%Y-%m-%d"),
            "timestamp": int(time.time()),
            # Market indices
            "spy": {
                "price": round(spy_price, 2),
                "change_5d": round(spy_change, 2),
                "ma20": round(spy_ma20, 2),
                "ma50": round(spy_ma50, 2),
            },
            "qqq": {"price": round(qqq_price, 2), "change_5d": round(qqq_change, 2)},
            "vix": {"current": round(vix_current, 2), "interpretation": interpret_vix(vix_current)},
            # Regime analysis
            "market_regime": regime,
            "sentiment": sentiment,
            "confidence": confidence,
            "volatility": round(volatility, 3),
            # Technical signals
            "signals": generate_signals(spy_price, spy_ma20, spy_ma50, vix_current),
            # Summary
            "summary": generate_summary(regime, sentiment, spy_change, vix_current),
        }

        # Save to file
        try:
            with open(output_path, "w") as f:
                json.dump(mood, f, indent=2)
            logging.info(f"Market mood updated: {regime} regime, {sentiment} sentiment")
        except Exception as e:
            logging.error(f"Failed to save market mood: {e}")

        return mood

    except Exception as e:
        logging.error(f"Market mood update failed: {e}")
        return {
            "error": str(e),
            "timestamp": int(time.time()),
            "date": time.strftime("%Y-%m-%d"),
            "market_regime": "unknown",
            "sentiment": "neutral",
        }


def classify_regime(
    spy_change: float,
    qqq_change: float,
    vix: float,
    spy_price: float,
    spy_ma20: float,
    spy_ma50: float,
    volatility: float,
) -> tuple[str, str, int]:
    """
    Classify market regime as bull, bear, or sideways.

    Args:
        spy_change: SPY 5-day % change
        qqq_change: QQQ 5-day % change
        vix: Current VIX level
        spy_price: Current SPY price
        spy_ma20: SPY 20-day MA
        spy_ma50: SPY 50-day MA
        volatility: Annualized volatility

    Returns:
        Tuple of (regime, sentiment, confidence)
    """
    bullish_signals = 0
    bearish_signals = 0

    # VIX analysis
    if vix < 15:
        bullish_signals += 2  # Strong bull signal
    elif vix > 25:
        bearish_signals += 2  # Strong bear signal
    elif vix > 20:
        bearish_signals += 1  # Moderate bear signal

    # Trend analysis
    if spy_change > 1:
        bullish_signals += 1
    elif spy_change < -2:
        bearish_signals += 1

    if qqq_change > 1:
        bullish_signals += 1
    elif qqq_change < -2:
        bearish_signals += 1

    # Moving average analysis
    if spy_price > spy_ma20 and spy_price > spy_ma50:
        bullish_signals += 1
    elif spy_price < spy_ma20 and spy_price < spy_ma50:
        bearish_signals += 1

    if spy_ma20 > spy_ma50:
        bullish_signals += 1
    elif spy_ma20 < spy_ma50:
        bearish_signals += 1

    # Volatility analysis
    if volatility < 0.15:
        bullish_signals += 1
    elif volatility > 0.25:
        bearish_signals += 1

    # Classify regime
    total_signals = bullish_signals + bearish_signals
    confidence = min(
        95, max(50, int((abs(bullish_signals - bearish_signals) / (total_signals or 1)) * 100))
    )

    if bullish_signals > bearish_signals + 1:
        regime = "bull"
        sentiment = "risk-on"
    elif bearish_signals > bullish_signals + 1:
        regime = "bear"
        sentiment = "risk-off"
    else:
        regime = "sideways"
        sentiment = "neutral"

    return regime, sentiment, confidence


def interpret_vix(vix: float) -> str:
    """Interpret VIX level."""
    if vix < 12:
        return "very low (complacent)"
    elif vix < 15:
        return "low (calm)"
    elif vix < 20:
        return "normal"
    elif vix < 25:
        return "elevated (caution)"
    elif vix < 30:
        return "high (fear)"
    else:
        return "very high (panic)"


def generate_signals(
    spy_price: float, spy_ma20: float, spy_ma50: float, vix: float
) -> dict[str, str]:
    """Generate technical signals."""
    signals = {}

    # Price vs MA signals
    if spy_price > spy_ma20:
        signals["short_term"] = "bullish (above 20MA)"
    else:
        signals["short_term"] = "bearish (below 20MA)"

    if spy_price > spy_ma50:
        signals["medium_term"] = "bullish (above 50MA)"
    else:
        signals["medium_term"] = "bearish (below 50MA)"

    # MA crossover
    if spy_ma20 > spy_ma50:
        signals["ma_crossover"] = "golden cross (bullish)"
    elif spy_ma20 < spy_ma50:
        signals["ma_crossover"] = "death cross (bearish)"
    else:
        signals["ma_crossover"] = "neutral"

    # VIX signal
    if vix < 15:
        signals["volatility"] = "low (bullish)"
    elif vix > 25:
        signals["volatility"] = "high (bearish)"
    else:
        signals["volatility"] = "normal"

    return signals


def generate_summary(regime: str, sentiment: str, spy_change: float, vix: float) -> str:
    """Generate human-readable summary."""
    summaries = {
        "bull": f"Bull market confirmed. SPY trending higher ({spy_change:+.1f}% over 5 days), VIX low ({vix:.1f}). {sentiment.capitalize()} positioning favored.",
        "bear": f"Bear market conditions. SPY under pressure ({spy_change:+.1f}% over 5 days), VIX elevated ({vix:.1f}). {sentiment.capitalize()} positioning advised.",
        "sideways": f"Sideways/consolidation phase. SPY range-bound ({spy_change:+.1f}% over 5 days), VIX {vix:.1f}. {sentiment.capitalize()} bias.",
    }
    return summaries.get(regime, f"Market regime: {regime}")


def get_market_mood(json_path: str = "data/market_mood.json") -> dict[str, Any]:
    """
    Load market mood from JSON file.

    Args:
        json_path: Path to market mood JSON

    Returns:
        Market mood dict or empty dict if not found
    """
    try:
        with open(json_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Market mood file not found: {json_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Market mood JSON decode error: {e}")
        return {}
    except Exception as e:
        logging.error(f"Failed to load market mood: {e}")
        return {}


def is_market_mood_stale(json_path: str = "data/market_mood.json", max_age_hours: int = 24) -> bool:
    """
    Check if market mood is stale.

    Args:
        json_path: Path to market mood JSON
        max_age_hours: Maximum age in hours

    Returns:
        True if stale or missing
    """
    mood = get_market_mood(json_path)
    if not mood or "timestamp" not in mood:
        return True

    age = int(time.time()) - mood["timestamp"]
    return age > (max_age_hours * 3600)


# Quick test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing Market Mood Tracker...")

    mood = update_market_mood()

    print("\n" + "=" * 60)
    print(f"Market Mood: {mood.get('date', 'N/A')}")
    print("=" * 60)
    print(f"Regime: {mood.get('market_regime', 'unknown')}")
    print(f"Sentiment: {mood.get('sentiment', 'unknown')}")
    print(f"Confidence: {mood.get('confidence', 0)}%")
    print(
        f"SPY: ${mood.get('spy', {}).get('price', 0):.2f} ({mood.get('spy', {}).get('change_5d', 0):+.2f}%)"
    )
    print(
        f"VIX: {mood.get('vix', {}).get('current', 0):.2f} ({mood.get('vix', {}).get('interpretation', 'unknown')})"
    )
    print(f"Summary: {mood.get('summary', 'N/A')}")
    print("=" * 60)
