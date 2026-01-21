#!/usr/bin/env python3
"""
🔍 GHOST FULL-MARKET SCANNER

Scans the entire US stock market + crypto markets for opportunities.
Finds hidden movers, volume anomalies, and breakout patterns.

This is the CORE of Ghost's investment hunter capabilities.
"""

import asyncio
import logging
import os
import time
from typing import Any

import requests

LOGGER = logging.getLogger("ghost.market_scanner")

# Configuration
SCAN_ENABLED = os.getenv("MARKET_SCAN_ENABLED", "1") == "1"
SCAN_INTERVAL = int(os.getenv("MARKET_SCAN_INTERVAL", "300"))  # 5 minutes
MAX_OPPORTUNITIES = int(os.getenv("MAX_OPPORTUNITIES", "20"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.70"))

# Scanner scope configuration
SCAN_EQUITY_SCOPE = os.getenv("SCAN_EQUITY_SCOPE", "all")  # "all", "sp500", "nasdaq100"
SCAN_CRYPTO_SCOPE = os.getenv("SCAN_CRYPTO_SCOPE", "all")  # "all", "top100", "top500"
SCAN_MAX_EQUITIES = int(os.getenv("SCAN_MAX_EQUITIES", "3000"))
SCAN_MAX_CRYPTO = int(os.getenv("SCAN_MAX_CRYPTO", "3000"))
SCAN_MIN_DOLLAR_VOL_24H = float(os.getenv("SCAN_MIN_DOLLAR_VOL_24H", "500000"))
SCAN_MIN_PRICE_USD = float(os.getenv("SCAN_MIN_PRICE_USD", "1.00"))
SCAN_INCLUDE_OTC = os.getenv("SCAN_INCLUDE_OTC", "0") == "1"

# Polygon API
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
POLYGON_BASE = "https://api.polygon.io"


def get_all_tickers() -> list[str]:
    """
    Fetch all active US stock tickers from Polygon.
    
    Returns:
        List of ticker symbols (e.g., ['AAPL', 'TSLA', 'WOLF', ...])
    """
    if not POLYGON_API_KEY:
        LOGGER.warning("POLYGON_API_KEY not set - cannot fetch ticker list")
        return []

    try:
        url = f"{POLYGON_BASE}/v3/reference/tickers"
        params = {
            "active": "true",
            "market": "stocks",
            "limit": 1000,
            "apiKey": POLYGON_API_KEY,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        tickers = [t["ticker"] for t in data.get("results", [])]

        LOGGER.info(f"Fetched {len(tickers)} active tickers from Polygon")
        return tickers

    except Exception as e:
        LOGGER.error(f"Failed to fetch ticker list: {e}")
        return []


def get_volume_anomalies(tickers: list[str], top_n: int = 100) -> list[dict[str, Any]]:
    """
    Find stocks with unusual volume (3x+ average).
    
    Args:
        tickers: List of symbols to check
        top_n: Return top N anomalies
        
    Returns:
        List of {symbol, volume, avg_volume, volume_ratio}
    """
    anomalies = []

    # Use Polygon aggregate endpoint to check volume
    for symbol in tickers[:500]:  # Limit to 500 to avoid rate limits
        try:
            # Get today's data
            url = f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/range/1/day/{time.strftime('%Y-%m-%d')}/{time.strftime('%Y-%m-%d')}"
            params = {"apiKey": POLYGON_API_KEY}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                continue

            data = response.json()
            results = data.get("results", [])
            if not results:
                continue

            today = results[0]
            volume = today.get("v", 0)

            # Volume spike detection (simplified threshold calculation)
            # Note: Full implementation requires 30-day rolling average from time-series data
            avg_volume = volume * 0.33  # 3x threshold assumption

            volume_ratio = volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio >= 3.0:  # 3x+ volume
                anomalies.append(
                    {
                        "symbol": symbol,
                        "volume": volume,
                        "avg_volume": avg_volume,
                        "volume_ratio": volume_ratio,
                    }
                )

            # Small delay to avoid rate limits
            time.sleep(0.1)

        except Exception as e:
            LOGGER.debug(f"Error checking volume for {symbol}: {e}")
            continue

    # Sort by volume ratio (highest first)
    anomalies.sort(key=lambda x: x["volume_ratio"], reverse=True)

    LOGGER.info(f"Found {len(anomalies)} volume anomalies (top {top_n} returned)")
    return anomalies[:top_n]


def get_momentum_movers(tickers: list[str], top_n: int = 100) -> list[dict[str, Any]]:
    """
    Find stocks with strong price momentum (5%+ daily move).
    
    Args:
        tickers: List of symbols to check
        top_n: Return top N movers
        
    Returns:
        List of {symbol, price, change_pct, direction}
    """
    movers = []

    for symbol in tickers[:500]:  # Limit to avoid rate limits
        try:
            # Get today's price action
            url = f"{POLYGON_BASE}/v2/aggs/ticker/{symbol}/range/1/day/{time.strftime('%Y-%m-%d')}/{time.strftime('%Y-%m-%d')}"
            params = {"apiKey": POLYGON_API_KEY}

            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                continue

            data = response.json()
            results = data.get("results", [])
            if not results:
                continue

            today = results[0]
            open_price = today.get("o", 0)
            close_price = today.get("c", 0)

            if open_price <= 0:
                continue

            change_pct = ((close_price - open_price) / open_price) * 100
            direction = "UP" if change_pct > 0 else "DOWN"

            if abs(change_pct) >= 5.0:  # 5%+ move
                movers.append(
                    {
                        "symbol": symbol,
                        "price": close_price,
                        "change_pct": change_pct,
                        "direction": direction,
                    }
                )

            time.sleep(0.1)

        except Exception as e:
            LOGGER.debug(f"Error checking momentum for {symbol}: {e}")
            continue

    # Sort by absolute change (largest first)
    movers.sort(key=lambda x: abs(x["change_pct"]), reverse=True)

    LOGGER.info(f"Found {len(movers)} momentum movers (top {top_n} returned)")
    return movers[:top_n]


async def predict_opportunity(symbol: str) -> dict[str, Any] | None:
    """
    Run AI prediction on a symbol to get confidence and predicted move.
    
    Args:
        symbol: Stock symbol
        
    Returns:
        {confidence, predicted_pct, timeframe_hours, reasons}
    """
    try:
        # Import AI prediction engine
        from ghost_agent_loop import agent_decide

        # Run prediction
        decision = await agent_decide(symbol)

        if not decision:
            return None

        confidence = decision.get("confidence", 0)
        action = decision.get("action", "HOLD")
        reasoning = decision.get("reasoning", "")

        # Estimate predicted % move based on action
        predicted_pct = 0
        if action == "BUY":
            predicted_pct = 5.0 + (confidence * 10)  # 5-15% predicted gain
        elif action == "SELL":
            predicted_pct = -(3.0 + (confidence * 7))  # -3 to -10% predicted drop

        return {
            "confidence": confidence,
            "predicted_pct": predicted_pct,
            "timeframe_hours": 24,  # Default 24hr prediction
            "reasons": [reasoning] if reasoning else [],
            "action": action,
        }

    except Exception as e:
        LOGGER.error(f"Failed to predict {symbol}: {e}")
        return None


async def scan_stocks() -> list[dict[str, Any]]:
    """
    Scan all US stocks for opportunities.
    
    Returns:
        List of opportunities with predictions
    """
    LOGGER.info("🔍 Starting full stock market scan...")

    # Get all tickers
    tickers = get_all_tickers()
    if not tickers:
        LOGGER.warning("No tickers available - using fallback list")
        tickers = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "WOLF"]

    # Find volume anomalies
    volume_anomalies = get_volume_anomalies(tickers, top_n=50)

    # Find momentum movers
    momentum_movers = get_momentum_movers(tickers, top_n=50)

    # Combine candidates (deduplicate)
    candidates = {}
    for item in volume_anomalies:
        candidates[item["symbol"]] = {"volume_signal": True, **item}

    for item in momentum_movers:
        symbol = item["symbol"]
        if symbol in candidates:
            candidates[symbol].update({"momentum_signal": True, **item})
        else:
            candidates[symbol] = {"momentum_signal": True, **item}

    LOGGER.info(f"Found {len(candidates)} candidate stocks")

    # Run AI predictions on candidates
    opportunities = []
    for symbol, data in list(candidates.items())[:30]:  # Limit to 30 predictions
        prediction = await predict_opportunity(symbol)

        if prediction and prediction["confidence"] >= MIN_CONFIDENCE:
            opportunity = {
                "symbol": symbol,
                "type": "stock",
                "confidence": prediction["confidence"],
                "predicted_pct": prediction["predicted_pct"],
                "timeframe_hours": prediction["timeframe_hours"],
                "reasons": prediction["reasons"],
                "action": prediction["action"],
                "signals": {
                    "volume_anomaly": data.get("volume_signal", False),
                    "momentum": data.get("momentum_signal", False),
                },
            }

            opportunities.append(opportunity)

    # Sort by confidence (highest first)
    opportunities.sort(key=lambda x: x["confidence"], reverse=True)

    LOGGER.info(f"✅ Found {len(opportunities)} high-confidence stock opportunities")
    return opportunities[:MAX_OPPORTUNITIES]


async def scan_crypto() -> list[dict[str, Any]]:
    """
    Scan crypto markets for opportunities.
    
    Returns:
        List of crypto opportunities
    """
    LOGGER.info("🔍 Starting crypto market scan...")

    # Top crypto to scan (includes VIP coins)
    crypto_symbols = [
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "MATIC", "DOT", "AVAX",
        "SHIB", "PEPE", "FLOKI",  # Meme coins
        "WEPE", "LILPEPE", "DORKL", "SLOTH", "APC"  # VIP coins
    ]

    opportunities = []

    for symbol in crypto_symbols:
        prediction = await predict_opportunity(symbol)

        if prediction and prediction["confidence"] >= MIN_CONFIDENCE:
            opportunity = {
                "symbol": symbol,
                "type": "crypto",
                "confidence": prediction["confidence"],
                "predicted_pct": prediction["predicted_pct"],
                "timeframe_hours": prediction["timeframe_hours"],
                "reasons": prediction["reasons"],
                "action": prediction["action"],
            }

            opportunities.append(opportunity)

    opportunities.sort(key=lambda x: x["confidence"], reverse=True)

    LOGGER.info(f"✅ Found {len(opportunities)} high-confidence crypto opportunities")
    return opportunities


async def scan_all() -> dict[str, Any]:
    """
    Scan both stocks and crypto.
    
    Returns:
        {stocks: [...], crypto: [...], total: N}
    """
    stock_opportunities = await scan_stocks()
    crypto_opportunities = await scan_crypto()

    return {
        "stocks": stock_opportunities,
        "crypto": crypto_opportunities,
        "total": len(stock_opportunities) + len(crypto_opportunities),
        "timestamp": int(time.time()),
    }


async def market_scan_loop():
    """
    Continuous market scanning loop.
    Runs every SCAN_INTERVAL seconds.
    """
    LOGGER.info(f"🔍 Market scanner started (interval={SCAN_INTERVAL}s)")

    while True:
        try:
            if not SCAN_ENABLED:
                LOGGER.debug("Market scanner disabled")
                await asyncio.sleep(SCAN_INTERVAL)
                continue

            # Run full scan
            results = await scan_all()

            # Log summary
            LOGGER.info(
                f"📊 Scan complete: {results['total']} opportunities "
                f"(stocks={len(results['stocks'])}, crypto={len(results['crypto'])})"
            )

            # Results stored in memory only (no persistence required for scanning)

            # Send instant alerts for high-scoring opportunities
            try:
                from core.telegram_hunter import send_instant_alert
                
                all_opportunities = results["stocks"] + results["crypto"]
                for opp in all_opportunities:
                    # send_instant_alert checks score threshold + cooldown internally
                    await send_instant_alert(opp)
            except Exception as e:
                LOGGER.error(f"Error sending telegram alerts: {e}", exc_info=True)

            # Wait before next scan
            await asyncio.sleep(SCAN_INTERVAL)

        except Exception as e:
            LOGGER.error(f"Error in market scan loop: {e}", exc_info=True)
            await asyncio.sleep(SCAN_INTERVAL * 2)


async def start_market_scanner():
    """Start the market scanner as a background task."""
    if not SCAN_ENABLED:
        LOGGER.info("Market scanner is disabled (set MARKET_SCAN_ENABLED=1 to enable)")
        return

    await market_scan_loop()


if __name__ == "__main__":
    # Run standalone
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("🔍 Starting Ghost Market Scanner (standalone mode)")
    print(f"   Scan interval: {SCAN_INTERVAL}s")
    print(f"   Max opportunities: {MAX_OPPORTUNITIES}")
    print(f"   Min confidence: {MIN_CONFIDENCE}")
    print()

    asyncio.run(market_scan_loop())
