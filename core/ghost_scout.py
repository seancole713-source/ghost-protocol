#!/usr/bin/env python3
"""
🔍 GHOST SCOUT - Find the NEXT BIG DEAL

Think like a video game scout finding new talent:
- Scan ALL assets for bullish potential
- Track which ones are MAKING MONEY
- The goal: Find the next #1 money maker

EVERY asset is competing to prove they can MAKE MONEY.
Losses are BAD. Profits are WINS.

This is survival of the fittest - only the TOP 10 money makers
get to be in Ghost's predictions.

NEW FEATURES:
- Dynamic mover detection (catches 10%+ daily gainers)
- News sentiment integration (real ✅ indicator)
- Flexible hold periods (not just 48hr)
"""

import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random

# ── #111-116: Centralized Symbol Registry (replaces 4 duplicate lists) ──
from core.symbol_registry import (
    ALL_STOCKS as _REGISTRY_STOCKS,
    ALL_CRYPTO as _REGISTRY_CRYPTO,
    get_coingecko_id,
    get_all_coingecko_ids,
)

LOGGER = logging.getLogger("ghost.scout")


# ALL ASSETS IN THE GAME - Everyone competes!
# Source of truth: core/symbol_registry.py — NO duplicate lists here
ALL_STOCKS = list(_REGISTRY_STOCKS)
ALL_CRYPTO = list(_REGISTRY_CRYPTO)


def fetch_daily_movers(min_gain_pct: float = 5.0) -> List[Dict]:
    """
    🚀 DYNAMIC MOVER DETECTION
    
    Fetch today's biggest gainers that Ghost might be missing.
    This catches stocks like Nextpower +16%, Seagate +15% that
    aren't in our static list.
    
    Uses Yahoo Finance screener API.
    
    Returns: List of {symbol, name, change_pct, price}
    """
    movers = []
    
    try:
        # Yahoo Finance day gainers
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            "scrIds": "day_gainers",
            "count": 25
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
            
            for q in quotes:
                change_pct = q.get("regularMarketChangePercent", 0)
                if change_pct >= min_gain_pct:
                    movers.append({
                        "symbol": q.get("symbol", ""),
                        "name": q.get("shortName", q.get("symbol", "")),
                        "change_pct": round(change_pct, 2),
                        "price": q.get("regularMarketPrice", 0),
                        "volume": q.get("regularMarketVolume", 0)
                    })
            
            LOGGER.info(f"🚀 [MOVERS] Found {len(movers)} stocks up {min_gain_pct}%+ today")
    except Exception as e:
        LOGGER.error(f"🚀 [MOVERS] Fetch error: {e}")
    
    return movers


def get_news_sentiment_for_symbol(symbol: str) -> Dict:
    """
    📰 Get news sentiment for a symbol.
    
    Returns sentiment data that can influence predictions.
    Used to determine if ✅ indicator should show.
    """
    try:
        from core.news_sentiment import fetch_news_sentiment
        news_data = fetch_news_sentiment(symbol, limit=5)
        
        return {
            "has_news": news_data.get("article_count", 0) > 0,
            "sentiment_score": news_data.get("sentiment_score", 0),
            "sentiment_label": news_data.get("sentiment_label", "NEUTRAL"),
            "article_count": news_data.get("article_count", 0),
            "news_influenced": abs(news_data.get("sentiment_score", 0)) > 0.2  # Strong sentiment
        }
    except Exception as e:
        LOGGER.debug(f"News fetch failed for {symbol}: {e}")
        return {
            "has_news": False,
            "sentiment_score": 0,
            "sentiment_label": "NEUTRAL",
            "article_count": 0,
            "news_influenced": False
        }


class GhostScout:
    """
    🔍 The Scout finds MONEY MAKERS
    
    Every day, the scout:
    1. Looks at ALL assets (static + dynamic movers!)
    2. Evaluates bullish potential with NEWS SENTIMENT
    3. Records predictions for EVERYONE
    4. Later: See who MADE MONEY
    
    The ones who make money = TOP 10
    The ones who lose money = Stay benched
    
    NO BLACKLIST. Everyone gets a fair shot.
    Prove yourself through PROFITS.
    
    NEW: Dynamic mover detection catches 10%+ gainers!
    NEW: News sentiment integration for ✅ indicator!
    """
    
    def __init__(self, include_dynamic_movers: bool = True):
        self.stocks = ALL_STOCKS[:]
        self.crypto = ALL_CRYPTO[:]
        self.include_dynamic_movers = include_dynamic_movers
        self.dynamic_movers_added = []
        
        # #41: Reusable HTTP session — keeps TCP connections alive across calls
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        
        # Add dynamic movers to stock list
        if include_dynamic_movers:
            self._add_dynamic_movers()
        
        LOGGER.info(f"🔍 [SCOUT] Ready to find money makers!")
        LOGGER.info(f"   {len(self.stocks)} stocks competing")
        LOGGER.info(f"   {len(self.crypto)} crypto competing")
        if self.dynamic_movers_added:
            LOGGER.info(f"   🚀 Dynamic movers added: {self.dynamic_movers_added}")
    
    def _add_dynamic_movers(self):
        """Add today's biggest gainers to the scout list with BULLISH bias"""
        try:
            movers = fetch_daily_movers(min_gain_pct=5.0)  # 5%+ gainers
            for mover in movers[:20]:  # Max 20 dynamic adds
                symbol = mover.get("symbol", "").replace(".US", "").split(".")[0]
                if symbol and symbol not in self.stocks:
                    self.stocks.append(symbol)
                    self.dynamic_movers_added.append(f"{symbol} (+{mover['change_pct']}%)")
                    # Track that this is a BIG GAINER - should be BUY not SELL!
                    self._bullish_movers = getattr(self, '_bullish_movers', set())
                    self._bullish_movers.add(symbol)
            
            if self.dynamic_movers_added:
                LOGGER.info(f"🚀 [SCOUT] Added {len(self.dynamic_movers_added)} dynamic movers to watchlist!")
        except Exception as e:
            LOGGER.error(f"🚀 [SCOUT] Dynamic mover fetch failed: {e}")
    
    def scout_all(self, use_news: bool = True) -> Dict:
        """
        Run a full scouting cycle.
        
        This makes predictions for EVERY asset so we can
        track who's actually making money.
        
        Args:
            use_news: If True, fetch news sentiment for each symbol (slower but accurate ✅)
        """
        from core.money_game_engine import get_money_game
        
        game = get_money_game()
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "stocks_scouted": 0,
            "crypto_scouted": 0,
            "news_influenced_count": 0,
            "dynamic_movers_found": len(self.dynamic_movers_added),
            "trades_recorded": []
        }
        
        # Import _LATEST_PREDICTIONS to populate it alongside Money Game
        # This allows TOP 10 to find these predictions
        try:
            import wolf_app
            latest_predictions = wolf_app._LATEST_PREDICTIONS
        except ImportError:
            latest_predictions = {}
        
        LOGGER.info("🔍 [SCOUT] Starting full scouting run...")
        
        # Scout all stocks
        for symbol in self.stocks:
            try:
                prediction = self._make_prediction(symbol, "stock", use_news=use_news)
                if prediction:
                    trade_id = game.record_trade(
                        symbol=symbol,
                        asset_type="stock",
                        direction=prediction["direction"],
                        entry_price=prediction["entry_price"],
                        target_price=prediction["target_price"],
                        confidence=prediction["confidence"]
                    )
                    if trade_id > 0:
                        results["stocks_scouted"] += 1
                        if prediction.get("news_influenced"):
                            results["news_influenced_count"] += 1
                        results["trades_recorded"].append({
                            "trade_id": trade_id,
                            "symbol": symbol,
                            "direction": prediction["direction"],
                            "news_influenced": prediction.get("news_influenced", False)
                        })
                        # ALSO populate _LATEST_PREDICTIONS for TOP 10
                        if latest_predictions is not None:
                            latest_predictions[symbol] = {
                                "symbol": symbol,
                                "direction": prediction["direction"],
                                "confidence": prediction["confidence"],
                                "price": prediction["entry_price"],
                                "current_price": prediction["entry_price"],
                                "entry_price": prediction["entry_price"],
                                "target_price": prediction["target_price"],
                                "asset_type": "stock",
                                "run_at": time.time(),
                                "source": "money_game_scout",
                                # NEWS SENTIMENT DATA - for ✅ indicator
                                "news_influenced": prediction.get("news_influenced", False),
                                "sentiment_score": prediction.get("sentiment_score", 0),
                                "sentiment_label": prediction.get("sentiment_label", "NEUTRAL"),
                                # HOLD PERIOD - flexible, not just 48hr
                                "hold_hours": prediction.get("hold_hours", 48),
                                "hold_reason": prediction.get("hold_reason", "default")
                            }
            except Exception as e:
                LOGGER.error(f"🔍 [SCOUT] Error scouting {symbol}: {e}")
        
        # Scout all crypto
        for symbol in self.crypto:
            try:
                prediction = self._make_prediction(symbol, "crypto", use_news=use_news)
                if prediction:
                    trade_id = game.record_trade(
                        symbol=symbol,
                        asset_type="crypto",
                        direction=prediction["direction"],
                        entry_price=prediction["entry_price"],
                        target_price=prediction["target_price"],
                        confidence=prediction["confidence"]
                    )
                    if trade_id > 0:
                        results["crypto_scouted"] += 1
                        if prediction.get("news_influenced"):
                            results["news_influenced_count"] += 1
                        results["trades_recorded"].append({
                            "trade_id": trade_id,
                            "symbol": symbol,
                            "direction": prediction["direction"],
                            "news_influenced": prediction.get("news_influenced", False)
                        })
                        # ALSO populate _LATEST_PREDICTIONS for TOP 10
                        if latest_predictions is not None:
                            latest_predictions[symbol] = {
                                "symbol": symbol,
                                "direction": prediction["direction"],
                                "confidence": prediction["confidence"],
                                "price": prediction["entry_price"],
                                "current_price": prediction["entry_price"],
                                "entry_price": prediction["entry_price"],
                                "target_price": prediction["target_price"],
                                "asset_type": "crypto",
                                "run_at": time.time(),
                                "source": "money_game_scout",
                                # NEWS SENTIMENT DATA - for ✅ indicator
                                "news_influenced": prediction.get("news_influenced", False),
                                "sentiment_score": prediction.get("sentiment_score", 0),
                                "sentiment_label": prediction.get("sentiment_label", "NEUTRAL"),
                                # HOLD PERIOD - flexible
                                "hold_hours": prediction.get("hold_hours", 48),
                                "hold_reason": prediction.get("hold_reason", "default")
                            }
            except Exception as e:
                LOGGER.error(f"🔍 [SCOUT] Error scouting {symbol}: {e}")
        
        LOGGER.info(f"🔍 [SCOUT] Scouting complete!")
        LOGGER.info(f"   Stocks: {results['stocks_scouted']}")
        LOGGER.info(f"   Crypto: {results['crypto_scouted']}")
        LOGGER.info(f"   📰 News-influenced: {results['news_influenced_count']}")
        if self.dynamic_movers_added:
            LOGGER.info(f"   🚀 Dynamic movers: {len(self.dynamic_movers_added)}")
        if latest_predictions:
            LOGGER.info(f"   Predictions in memory: {len(latest_predictions)}")
        
        return results

    # ── #41-43: Concurrent scout with ThreadPoolExecutor ──────────────
    def scout_all_fast(self, use_news: bool = True, max_workers: int = 10) -> Dict:
        """
        Run a full scouting cycle with PARALLEL price fetches.

        Uses ThreadPoolExecutor to scout up to `max_workers` symbols
        simultaneously. Same logic as scout_all() but ~10x faster.

        Args:
            use_news: If True, fetch news sentiment (slower but accurate).
            max_workers: Max concurrent HTTP requests (default 10,
                         keeps CoinGecko/Polygon rate limits happy).
        """
        from core.money_game_engine import get_money_game

        game = get_money_game()
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "stocks_scouted": 0,
            "crypto_scouted": 0,
            "news_influenced_count": 0,
            "dynamic_movers_found": len(self.dynamic_movers_added),
            "trades_recorded": [],
            "mode": "fast_parallel",
        }

        try:
            import wolf_app
            latest_predictions = wolf_app._LATEST_PREDICTIONS
        except ImportError:
            latest_predictions = {}

        LOGGER.info(f"⚡ [SCOUT-FAST] Starting parallel scouting ({max_workers} workers)...")

        # Build a unified work list: (symbol, asset_type)
        work = [(s, "stock") for s in self.stocks] + [(s, "crypto") for s in self.crypto]

        def _scout_one(item):
            """Scout a single symbol — runs in a thread."""
            sym, atype = item
            try:
                pred = self._make_prediction(sym, atype, use_news=use_news)
                if pred:
                    return (sym, atype, pred, None)
            except Exception as exc:
                return (sym, atype, None, exc)
            return (sym, atype, None, None)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_scout_one, w): w for w in work}
            for future in as_completed(futures):
                sym, atype, pred, exc = future.result()
                if exc:
                    LOGGER.error(f"⚡ [SCOUT-FAST] Error scouting {sym}: {exc}")
                    continue
                if not pred:
                    continue

                trade_id = game.record_trade(
                    symbol=sym,
                    asset_type=atype,
                    direction=pred["direction"],
                    entry_price=pred["entry_price"],
                    target_price=pred["target_price"],
                    confidence=pred["confidence"],
                )
                if trade_id <= 0:
                    continue

                if atype == "stock":
                    results["stocks_scouted"] += 1
                else:
                    results["crypto_scouted"] += 1
                if pred.get("news_influenced"):
                    results["news_influenced_count"] += 1
                results["trades_recorded"].append({
                    "trade_id": trade_id,
                    "symbol": sym,
                    "direction": pred["direction"],
                    "news_influenced": pred.get("news_influenced", False),
                })

                if latest_predictions is not None:
                    latest_predictions[sym] = {
                        "symbol": sym,
                        "direction": pred["direction"],
                        "confidence": pred["confidence"],
                        "price": pred["entry_price"],
                        "current_price": pred["entry_price"],
                        "entry_price": pred["entry_price"],
                        "target_price": pred["target_price"],
                        "asset_type": atype,
                        "run_at": time.time(),
                        "source": "money_game_scout_fast",
                        "news_influenced": pred.get("news_influenced", False),
                        "sentiment_score": pred.get("sentiment_score", 0),
                        "sentiment_label": pred.get("sentiment_label", "NEUTRAL"),
                        "hold_hours": pred.get("hold_hours", 48),
                        "hold_reason": pred.get("hold_reason", "default"),
                    }

        LOGGER.info(f"⚡ [SCOUT-FAST] Scouting complete!")
        LOGGER.info(f"   Stocks: {results['stocks_scouted']}")
        LOGGER.info(f"   Crypto: {results['crypto_scouted']}")
        LOGGER.info(f"   📰 News-influenced: {results['news_influenced_count']}")
        if latest_predictions:
            LOGGER.info(f"   Predictions in memory: {len(latest_predictions)}")

        return results
    
    def _make_prediction(self, symbol: str, asset_type: str, use_news: bool = True) -> Optional[Dict]:
        """
        Make a prediction for a symbol.
        
        Integrates:
        - Technical analysis
        - News sentiment (for ✅ indicator)
        - Dynamic hold period calculation
        """
        try:
            # Get current price
            current_price = self._get_current_price(symbol, asset_type)
            if not current_price:
                return None
            
            # Get base prediction from the engine
            prediction = self._get_prediction_from_engine(symbol, asset_type, current_price)
            if not prediction:
                return None
            
            # Add news sentiment if enabled (this makes ✅ real!)
            if use_news:
                news_data = get_news_sentiment_for_symbol(symbol)
                prediction["news_influenced"] = news_data.get("news_influenced", False)
                prediction["sentiment_score"] = news_data.get("sentiment_score", 0)
                prediction["sentiment_label"] = news_data.get("sentiment_label", "NEUTRAL")
                
                # Boost confidence if news agrees with direction
                if news_data["news_influenced"]:
                    sentiment = news_data["sentiment_score"]
                    if (prediction["direction"] == "BUY" and sentiment > 0) or \
                       (prediction["direction"] == "SELL" and sentiment < 0):
                        # News confirms direction - boost confidence!
                        prediction["confidence"] = min(0.85, prediction["confidence"] * 1.15)
                        LOGGER.info(f"📰 [NEWS] {symbol}: News confirms {prediction['direction']} (sentiment: {sentiment:.2f})")
            
            # Calculate smart hold period based on volatility and momentum
            prediction["hold_hours"] = self._calculate_hold_period(symbol, asset_type, prediction)
            prediction["hold_reason"] = self._get_hold_reason(prediction["hold_hours"])
            
            return prediction
        except Exception as e:
            LOGGER.error(f"🔍 [SCOUT] Prediction error for {symbol}: {e}")
            return None
    
    def _calculate_hold_period(self, symbol: str, asset_type: str, prediction: Dict) -> int:
        """
        Calculate optimal hold period based on asset characteristics.
        
        ENHANCED: Returns 1-7 days based on:
        - Asset volatility (crypto = shorter, stocks = longer)
        - News catalyst (hot news = 1-2 days max)
        - Confidence level (high confidence = can wait longer)
        - RSI momentum exhaustion (overbought/oversold = shorter)
        - Trend strength from prediction data
        
        Returns hours to hold (24-168h = 1-7 days).
        """
        # Base hold in DAYS (1-7 scale)
        base_days = 3  # Default 3-day swing
        
        # Asset type adjustment
        if asset_type == "crypto":
            base_days = 2  # Crypto: faster moves, 1-3 day range
        else:
            base_days = 4  # Stocks: slower, 2-5 day range typical
        
        # News catalyst - shorter hold (ride the wave, don't overstay)
        if prediction.get("news_influenced"):
            base_days = min(base_days, 2)  # News plays = 1-2 days max
        
        # Confidence adjustment
        conf = prediction.get("confidence", 0.5)
        if conf >= 0.85:
            base_days += 1  # Very strong signal = can hold longer for bigger target
        elif conf >= 0.7:
            pass  # Normal confidence = keep base
        elif conf >= 0.5:
            base_days = max(1, base_days - 1)  # Weak signal = shorter hold
        else:
            base_days = 1  # Very weak = quick scalp only
        
        # RSI-based momentum exhaustion (from prediction if available)
        rsi = prediction.get("rsi", 50)
        direction = prediction.get("direction", "UP")
        
        if direction == "UP" and rsi > 75:
            # Overbought on bullish = momentum exhausting, take profits soon
            base_days = max(1, base_days - 1)
        elif direction == "DOWN" and rsi < 25:
            # Oversold on bearish = bounce coming, don't overstay short
            base_days = max(1, base_days - 1)
        
        # Volatility adjustment (if price change % is extreme)
        price_change = abs(prediction.get("price_change_pct", 0))
        if price_change > 10:
            # Big move already happened - shorter hold to lock in gains
            base_days = max(1, base_days - 1)
        elif price_change < 2:
            # Slow mover - needs more time to develop
            base_days = min(7, base_days + 1)
        
        # Clamp to 1-7 days
        final_days = max(1, min(7, base_days))
        
        # Store the day estimate for downstream use
        prediction["hold_days"] = final_days
        prediction["hold_estimate"] = self._format_hold_estimate(final_days)
        
        # Return hours for backward compatibility (existing code expects hours)
        return final_days * 24
    
    def _format_hold_estimate(self, days: int) -> str:
        """Format hold estimate like '2-3 days' or '5-7 days'"""
        if days == 1:
            return "1-2 days"
        elif days == 2:
            return "2-3 days"
        elif days == 3:
            return "3-4 days"
        elif days == 4:
            return "4-5 days"
        elif days == 5:
            return "5-6 days"
        elif days == 6:
            return "5-7 days"
        else:  # 7
            return "6-7 days"
    
    def _get_hold_reason(self, hours: int) -> str:
        """Get human-readable hold reason based on days"""
        days = hours // 24
        if days <= 1:
            return "day_trade"  # 1 day
        elif days <= 2:
            return "momentum_trade"  # 1-2 days
        elif days <= 3:
            return "swing_trade"  # 2-3 days
        elif days <= 5:
            return "position_trade"  # 4-5 days
        else:
            return "trend_trade"  # 6-7 days
    
    def _get_current_price(self, symbol: str, asset_type: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            if asset_type == "crypto":
                return self._get_crypto_price(symbol)
            else:
                return self._get_stock_price(symbol)
        except Exception as e:
            LOGGER.debug(f"Price fetch failed for {symbol}: {e}")
            return None
    
    def _get_crypto_price(self, symbol: str) -> Optional[float]:
        """Get crypto price from CoinGecko"""
        
        # Use centralized symbol registry (#84: kill duplicate CoinGecko maps)
        cg_id = get_coingecko_id(symbol)
        if not cg_id:
            # Fallback for unknown symbols
            cg_id = symbol.lower()
        
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
        
        # Retry once on 429 rate-limit (CoinGecko free tier)
        for attempt in range(2):
            try:
                resp = self._session.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if cg_id in data:
                        return data[cg_id]["usd"]
                elif resp.status_code == 429 and attempt == 0:
                    import time
                    LOGGER.debug(f"[SCOUT] CoinGecko 429 for {symbol}, retrying in 2s...")
                    time.sleep(2)
                    continue
                # Non-retryable error
                break
            except Exception:
                break
        
        return None
    
    def _get_stock_price(self, symbol: str) -> Optional[float]:
        """Get stock price from Polygon API (Yahoo blocks server requests)"""
        polygon_key = os.getenv("POLYGON_API_KEY")
        if not polygon_key:
            LOGGER.warning(f"No POLYGON_API_KEY - cannot fetch {symbol}")
            return None
        
        try:
            # Polygon prev close endpoint - most reliable
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={polygon_key}"
            resp = self._session.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    price = results[0].get("c")  # close price
                    if price:
                        return float(price)
        except Exception as e:
            LOGGER.debug(f"Polygon price fetch failed for {symbol}: {e}")
        
        return None
    
    def _get_prediction_from_engine(self, symbol: str, asset_type: str, current_price: float) -> Dict:
        """
        Get a real prediction from the Ghost engine.
        
        This integrates with the actual prediction system.
        Returns: direction, target_price, confidence
        """
        # Try to use the actual prediction engine
        try:
            if asset_type == "crypto":
                from core.multi_crypto_predictor import MultiCryptoPredictor
                predictor = MultiCryptoPredictor()
                result = predictor.predict_symbol(symbol)
                
                if result:
                    direction = "BUY" if result.get("confidence", 0) > 0 else "SELL"
                    confidence = abs(result.get("confidence", 0.5))
                    
                    # Calculate target based on confidence
                    if direction == "BUY":
                        target = current_price * (1 + (confidence * 0.1))  # Up to 10% gain
                    else:
                        target = current_price * (1 - (confidence * 0.1))  # Up to 10% drop
                    
                    return {
                        "direction": direction,
                        "entry_price": current_price,
                        "target_price": target,
                        "confidence": confidence
                    }
        except ImportError:
            pass
        except Exception as e:
            LOGGER.debug(f"Engine error for {symbol}: {e}")
        
        # Fallback: Technical analysis based prediction
        return self._technical_prediction(symbol, asset_type, current_price)
    
    def _technical_prediction(self, symbol: str, asset_type: str, current_price: float) -> Dict:
        """
        Make a technical analysis based prediction.
        
        This is a simplified version - the real system would use
        full technical analysis, sentiment, etc.
        """
        # Default to slight bullish bias (markets generally go up)
        direction = "BUY"
        confidence = 0.55
        
        # CRITICAL: Dynamic movers (up 5%+ today) should ALWAYS be BUY!
        # They have proven momentum - ride the wave!
        bullish_movers = getattr(self, '_bullish_movers', set())
        if symbol in bullish_movers:
            LOGGER.info(f"🚀 [SCOUT] {symbol} is a dynamic mover - forcing BUY direction!")
            direction = "BUY"
            confidence = 0.70  # Higher confidence for momentum plays
            target = current_price * 1.05  # 5% continuation target
            return {
                "direction": direction,
                "entry_price": current_price,
                "target_price": target,
                "confidence": confidence
            }
        
        polygon_key = os.getenv("POLYGON_API_KEY")
        
        try:
            # Get some historical data to make a better prediction
            if asset_type == "stock" and polygon_key:
                # Polygon API for historical data
                from datetime import datetime, timedelta
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}?apiKey={polygon_key}"
                resp = self._session.get(url, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    results = data.get("results", [])
                    closes = [r.get("c") for r in results if r.get("c")]
                    
                    if len(closes) >= 10:
                        # Calculate momentum
                        recent = sum(closes[-5:]) / 5
                        older = sum(closes[-10:-5]) / 5
                        momentum = (recent - older) / older
                        
                        if momentum > 0.02:  # Uptrend
                            direction = "BUY"
                            confidence = min(0.8, 0.55 + momentum)
                        elif momentum < -0.02:  # Downtrend
                            direction = "SELL"
                            confidence = min(0.8, 0.55 + abs(momentum))
            
            elif asset_type == "crypto":
                # CoinGecko historical — use centralized registry
                cg_id = get_coingecko_id(symbol) or symbol.lower()
                
                url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days=30"
                resp = self._session.get(url, timeout=5)
                
                if resp.status_code == 200:
                    data = resp.json()
                    prices = [p[1] for p in data.get("prices", [])]
                    
                    if len(prices) >= 10:
                        recent = sum(prices[-5:]) / 5
                        older = sum(prices[-10:-5]) / 5
                        momentum = (recent - older) / older
                        
                        if momentum > 0.03:
                            direction = "BUY"
                            confidence = min(0.8, 0.55 + momentum)
                        elif momentum < -0.03:
                            direction = "SELL"
                            confidence = min(0.8, 0.55 + abs(momentum))
        
        except Exception as e:
            LOGGER.debug(f"Technical analysis fallback for {symbol}: {e}")
        
        # Calculate target price
        if direction == "BUY":
            target = current_price * (1 + (confidence * 0.08))  # Up to 8% target
        else:
            target = current_price * (1 - (confidence * 0.08))  # Down to -8%
        
        return {
            "direction": direction,
            "entry_price": current_price,
            "target_price": target,
            "confidence": confidence
        }


class GameResolver:
    """
    🏆 Resolves trades and counts the MONEY
    
    After 24-48 hours, we check:
    - What was the prediction?
    - What actually happened?
    - Did they MAKE MONEY or LOSE MONEY?
    
    Winners rise. Losers fall.
    That's the game.
    """
    
    def __init__(self):
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        LOGGER.info("🏆 [RESOLVER] Ready to count the money!")
    
    def _get_connection(self):
        import psycopg2
        return psycopg2.connect(self.DATABASE_URL)
    
    def resolve_pending_trades(self, hours_old: int = 24) -> Dict:
        """
        Resolve all trades older than X hours.
        
        This is where we find out WHO MADE MONEY!
        """
        from core.money_game_engine import get_money_game
        
        if not self.DATABASE_URL:
            return {"error": "No database"}
        
        game = get_money_game()
        results = {
            "resolved": 0,
            "winners": [],
            "losers": [],
            "total_profit": 0.0,
            "total_loss": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cutoff = datetime.utcnow() - timedelta(hours=hours_old)
            
            # Get unresolved trades
            cur.execute("""
                SELECT id, symbol, asset_type, direction, entry_price, target_price
                FROM money_game_trades
                WHERE resolved_at IS NULL AND created_at < %s
                ORDER BY created_at
                LIMIT 100
            """, (cutoff,))
            
            trades = cur.fetchall()
            conn.close()
            
            LOGGER.info(f"🏆 [RESOLVER] Found {len(trades)} trades to resolve...")
            
            for trade in trades:
                trade_id, symbol, asset_type, direction, entry_price, target_price = trade
                
                # Get current price
                scout = GhostScout()
                current_price = scout._get_current_price(symbol, asset_type)
                
                if current_price is None:
                    LOGGER.warning(f"🏆 [RESOLVER] Could not get price for {symbol}")
                    continue
                
                # Resolve the trade!
                result = game.resolve_trade(trade_id, current_price)
                
                if "error" not in result:
                    results["resolved"] += 1
                    profit = result.get("profit_pct", 0)
                    
                    if profit > 0:
                        results["winners"].append({
                            "symbol": symbol,
                            "profit": f"+{profit:.1f}%"
                        })
                        results["total_profit"] += profit
                    else:
                        results["losers"].append({
                            "symbol": symbol,
                            "loss": f"{profit:.1f}%"
                        })
                        results["total_loss"] += abs(profit)
            
            # After resolving, update rankings!
            if results["resolved"] > 0:
                LOGGER.info("🏆 [RESOLVER] Updating rankings after resolution...")
                game.update_rankings()
            
            LOGGER.info(f"🏆 [RESOLVER] Resolved {results['resolved']} trades")
            LOGGER.info(f"   💰 Winners: {len(results['winners'])}, Total Profit: +{results['total_profit']:.1f}%")
            LOGGER.info(f"   💸 Losers: {len(results['losers'])}, Total Loss: -{results['total_loss']:.1f}%")
            
            return results
            
        except Exception as e:
            LOGGER.error(f"🏆 [RESOLVER] Error: {e}")
            return {"error": str(e)}


# Convenience functions
def run_scouting_cycle(fast: bool = True) -> Dict:
    """Run a full scouting cycle (fast=True uses parallel ThreadPoolExecutor)"""
    scout = GhostScout()
    if fast:
        return scout.scout_all_fast()
    return scout.scout_all()


def resolve_trades(hours: int = 24) -> Dict:
    """Resolve pending trades"""
    resolver = GameResolver()
    return resolver.resolve_pending_trades(hours)


def get_game_status() -> Dict:
    """Get current game status"""
    from core.money_game_engine import get_money_game
    return get_money_game().get_game_status()
