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
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random

LOGGER = logging.getLogger("ghost.scout")


# ALL ASSETS IN THE GAME - Everyone competes!
ALL_STOCKS = [
    # Tech Giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA",
    "NFLX", "CRM", "ORCL", "ADBE", "INTC", "CSCO", "IBM",
    
    # Storage & Hardware (STX = Seagate, NOT the crypto!)
    "STX", "WDC", "NTAP", "PSTG",
    
    # Semiconductors
    "AVGO", "QCOM", "TXN", "MU", "LRCX", "AMAT", "KLAC", "MRVL",
    "ON", "NXPI", "ADI", "MCHP",
    
    # Cloud & Software
    "SNOW", "DDOG", "NET", "ZS", "CRWD", "PANW", "FTNT",
    "NOW", "WDAY", "HUBS", "TTD", "U", "RBLX",
    
    # AI & Innovation
    "PLTR", "AI", "PATH", "UPST", "COIN", "HOOD", "SOFI",
    
    # Healthcare & Biotech
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "AMGN",
    "GILD", "BMY", "REGN", "VRTX", "MRNA", "BIIB",
    
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA",
    "PYPL", "SQ", "BLK", "SCHW",
    
    # Consumer
    "NKE", "SBUX", "MCD", "KO", "PEP", "WMT", "COST", "TGT",
    "HD", "LOW", "DIS", "CMCSA",
    
    # Energy & Industrial
    "XOM", "CVX", "COP", "SLB", "CAT", "DE", "HON", "GE",
    "BA", "RTX", "LMT", "UPS", "FDX",
    
    # Others
    "ABNB", "UBER", "LYFT", "DASH", "SPOT", "ZM", "SHOP",
    "ROKU", "SNAP", "PINS", "TWLO", "OKTA"
]

ALL_CRYPTO = [
    # Majors
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT",
    "MATIC", "LINK", "UNI", "LTC", "BCH", "ATOM", "XLM",
    
    # Layer 1
    "NEAR", "APT", "SUI", "SEI", "FTM", "ALGO", "HBAR", "VET",
    "ICP", "FIL", "THETA", "EOS", "XTZ", "EGLD",
    
    # DeFi
    "AAVE", "CRV", "MKR", "SNX", "COMP", "SUSHI", "YFI",
    "1INCH", "BAL", "LDO", "PENDLE", "GMX",
    
    # Layer 2
    "ARB", "OP", "IMX", "LRC", "STRK", "ZK",
    
    # AI & Compute
    "RNDR", "FET", "OCEAN", "AGIX", "TAO", "AKT",
    
    # Gaming & NFT
    "AXS", "SAND", "MANA", "ENJ", "GALA", "ILV", "IMX", "MAGIC",
    "GODS", "PRIME", "YGG", "RON",
    
    # Infrastructure
    "GRT", "ROSE", "AR", "KAVA", "INJ", "TIA", "PYTH",
    "JUP", "JTO", "BONK", "WIF",
    
    # Others
    "SHIB", "PEPE", "FLOKI", "TURBO", "WLD", "BLUR",
    "DYDX", "MASK", "ENS", "CHZ", "AUDIO", "SUPER",
    
    # Old Guard
    "ZEC", "DASH", "NEO", "WAVES", "QTUM", "ZIL", "ICX",
    "RLC", "OMG", "BAT", "KNC", "ZRX",
    
    # Renamed to avoid collision with Seagate (STX stock)
    "STACKS"  # Stacks crypto (formerly STX)
]


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
        except:
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
                        prediction["confidence"] = min(0.95, prediction["confidence"] * 1.15)
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
        
        NOT just 48 hours anymore! This considers:
        - Asset volatility (crypto = shorter, stocks = longer)
        - News catalyst (hot news = shorter)
        - Confidence level (high confidence = can wait longer)
        - Trend strength
        
        Returns hours to hold.
        """
        base_hours = 48  # Default
        
        # Crypto is more volatile - shorter hold
        if asset_type == "crypto":
            base_hours = 24  # Crypto moves faster
        else:
            base_hours = 72  # Stocks need more time for moves
        
        # News catalyst - shorter hold (ride the news wave)
        if prediction.get("news_influenced"):
            base_hours = min(base_hours, 24)  # News trades are quick
        
        # High confidence - can hold longer
        conf = prediction.get("confidence", 0.5)
        if conf >= 0.8:
            base_hours = int(base_hours * 1.5)  # Strong signal, wait for bigger move
        elif conf < 0.6:
            base_hours = int(base_hours * 0.75)  # Weak signal, don't wait too long
        
        return max(12, min(168, base_hours))  # Clamp between 12h and 1 week
    
    def _get_hold_reason(self, hours: int) -> str:
        """Get human-readable hold reason"""
        if hours <= 24:
            return "momentum_trade"
        elif hours <= 48:
            return "swing_trade"
        elif hours <= 72:
            return "position_trade"
        else:
            return "trend_trade"
    
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
        
        # Map symbol to CoinGecko ID
        symbol_to_id = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin",
            "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "matic-network",
            "LINK": "chainlink", "UNI": "uniswap", "LTC": "litecoin",
            "ATOM": "cosmos", "NEAR": "near", "ARB": "arbitrum",
            "OP": "optimism", "RNDR": "render-token", "INJ": "injective-protocol",
            "SUI": "sui", "APT": "aptos", "SEI": "sei-network",
            "FET": "fetch-ai", "OCEAN": "ocean-protocol", "TAO": "bittensor",
            "SHIB": "shiba-inu", "PEPE": "pepe", "WIF": "dogwifcoin",
            "BONK": "bonk", "FLOKI": "floki", "TURBO": "turbo",
            "AAVE": "aave", "MKR": "maker", "SNX": "havven",
            "CRV": "curve-dao-token", "COMP": "compound-coin",
            "IMX": "immutable-x", "AXS": "axie-infinity", "SAND": "the-sandbox",
            "MANA": "decentraland", "GALA": "gala", "ILV": "illuvium",
            "GRT": "the-graph", "FIL": "filecoin", "AR": "arweave",
            "STACKS": "blockstack", "TIA": "celestia", "CHZ": "chiliz",  # STACKS not STX (avoid collision)
            "EGLD": "elrond-erd-2", "ZEC": "zcash", "RLC": "iexec-rlc"
        }
        
        cg_id = symbol_to_id.get(symbol.upper())
        if not cg_id:
            # Try lowercase
            cg_id = symbol.lower()
        
        try:
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if cg_id in data:
                    return data[cg_id]["usd"]
        except:
            pass
        
        return None
    
    def _get_stock_price(self, symbol: str) -> Optional[float]:
        """Get stock price from Yahoo Finance"""
        import requests
        
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=5)
            
            if resp.status_code == 200:
                data = resp.json()
                price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
                return float(price)
        except:
            pass
        
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
        import requests
        
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
        
        try:
            # Get some historical data to make a better prediction
            if asset_type == "stock":
                # Yahoo Finance historical
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=30d"
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, headers=headers, timeout=5)
                
                if resp.status_code == 200:
                    data = resp.json()
                    closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
                    closes = [c for c in closes if c is not None]
                    
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
                # CoinGecko historical
                symbol_to_id = {
                    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"
                    # Add more mappings as needed
                }
                cg_id = symbol_to_id.get(symbol.upper(), symbol.lower())
                
                url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days=30"
                resp = requests.get(url, timeout=5)
                
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
def run_scouting_cycle() -> Dict:
    """Run a full scouting cycle"""
    scout = GhostScout()
    return scout.scout_all()


def resolve_trades(hours: int = 24) -> Dict:
    """Resolve pending trades"""
    resolver = GameResolver()
    return resolver.resolve_pending_trades(hours)


def get_game_status() -> Dict:
    """Get current game status"""
    from core.money_game_engine import get_money_game
    return get_money_game().get_game_status()
