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
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random

LOGGER = logging.getLogger("ghost.scout")


# ALL ASSETS IN THE GAME - Everyone competes!
ALL_STOCKS = [
    # Tech Giants
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA",
    "NFLX", "CRM", "ORCL", "ADBE", "INTC", "CSCO", "IBM",
    
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
    "GRT", "ROSE", "AR", "STX", "KAVA", "INJ", "TIA", "PYTH",
    "JUP", "JTO", "BONK", "WIF",
    
    # Others
    "SHIB", "PEPE", "FLOKI", "TURBO", "WLD", "BLUR",
    "DYDX", "MASK", "ENS", "CHZ", "AUDIO", "SUPER",
    
    # Old Guard
    "ZEC", "DASH", "NEO", "WAVES", "QTUM", "ZIL", "ICX",
    "RLC", "OMG", "BAT", "KNC", "ZRX"
]


class GhostScout:
    """
    🔍 The Scout finds MONEY MAKERS
    
    Every day, the scout:
    1. Looks at ALL assets
    2. Evaluates bullish potential
    3. Records predictions for EVERYONE
    4. Later: See who MADE MONEY
    
    The ones who make money = TOP 10
    The ones who lose money = Stay benched
    
    NO BLACKLIST. Everyone gets a fair shot.
    Prove yourself through PROFITS.
    """
    
    def __init__(self):
        self.stocks = ALL_STOCKS[:]
        self.crypto = ALL_CRYPTO[:]
        
        LOGGER.info(f"🔍 [SCOUT] Ready to find money makers!")
        LOGGER.info(f"   {len(self.stocks)} stocks competing")
        LOGGER.info(f"   {len(self.crypto)} crypto competing")
    
    def scout_all(self) -> Dict:
        """
        Run a full scouting cycle.
        
        This makes predictions for EVERY asset so we can
        track who's actually making money.
        """
        from core.money_game_engine import get_money_game
        
        game = get_money_game()
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "stocks_scouted": 0,
            "crypto_scouted": 0,
            "trades_recorded": []
        }
        
        LOGGER.info("🔍 [SCOUT] Starting full scouting run...")
        
        # Scout all stocks
        for symbol in self.stocks:
            try:
                prediction = self._make_prediction(symbol, "stock")
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
                        results["trades_recorded"].append({
                            "trade_id": trade_id,
                            "symbol": symbol,
                            "direction": prediction["direction"]
                        })
            except Exception as e:
                LOGGER.error(f"🔍 [SCOUT] Error scouting {symbol}: {e}")
        
        # Scout all crypto
        for symbol in self.crypto:
            try:
                prediction = self._make_prediction(symbol, "crypto")
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
                        results["trades_recorded"].append({
                            "trade_id": trade_id,
                            "symbol": symbol,
                            "direction": prediction["direction"]
                        })
            except Exception as e:
                LOGGER.error(f"🔍 [SCOUT] Error scouting {symbol}: {e}")
        
        LOGGER.info(f"🔍 [SCOUT] Scouting complete!")
        LOGGER.info(f"   Stocks: {results['stocks_scouted']}")
        LOGGER.info(f"   Crypto: {results['crypto_scouted']}")
        
        return results
    
    def _make_prediction(self, symbol: str, asset_type: str) -> Optional[Dict]:
        """
        Make a prediction for a symbol.
        
        This would integrate with the actual prediction engine.
        For now, we get current price and make a prediction.
        """
        try:
            # Get current price
            current_price = self._get_current_price(symbol, asset_type)
            if not current_price:
                return None
            
            # Get prediction from the real engine
            prediction = self._get_prediction_from_engine(symbol, asset_type, current_price)
            
            return prediction
        except Exception as e:
            LOGGER.error(f"🔍 [SCOUT] Prediction error for {symbol}: {e}")
            return None
    
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
        import requests
        
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
            "STX": "blockstack", "TIA": "celestia", "CHZ": "chiliz",
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
