#!/usr/bin/env python3
"""
🔍 GHOST SCOUT V2 - Smarter Scouting with Rate Limiting

Improvements:
- Batch price fetching (fewer API calls)
- Rate limiting (respect API limits)
- Cron-ready endpoints
- Direct Telegram integration
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger("ghost.scout_v2")

DATABASE_URL = os.getenv("DATABASE_URL")


# CoinGecko ID mappings for ALL crypto
CRYPTO_TO_COINGECKO = {
    # Majors
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "XRP": "ripple", "ADA": "cardano", "DOGE": "dogecoin",
    "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "matic-network",
    "LINK": "chainlink", "UNI": "uniswap", "LTC": "litecoin",
    "BCH": "bitcoin-cash", "ATOM": "cosmos", "XLM": "stellar",
    
    # Layer 1
    "NEAR": "near", "APT": "aptos", "SUI": "sui", "SEI": "sei-network",
    "FTM": "fantom", "ALGO": "algorand", "HBAR": "hedera-hashgraph",
    "VET": "vechain", "ICP": "internet-computer", "FIL": "filecoin",
    "THETA": "theta-token", "EOS": "eos", "XTZ": "tezos", "EGLD": "elrond-erd-2",
    
    # DeFi
    "AAVE": "aave", "CRV": "curve-dao-token", "MKR": "maker",
    "SNX": "havven", "COMP": "compound-governance-token", "SUSHI": "sushi",
    "YFI": "yearn-finance", "1INCH": "1inch", "BAL": "balancer",
    "LDO": "lido-dao", "PENDLE": "pendle", "GMX": "gmx",
    
    # Layer 2
    "ARB": "arbitrum", "OP": "optimism", "IMX": "immutable-x",
    "LRC": "loopring", "STRK": "starknet", "ZK": "zksync",
    
    # AI & Compute
    "RNDR": "render-token", "FET": "fetch-ai", "OCEAN": "ocean-protocol",
    "AGIX": "singularitynet", "TAO": "bittensor", "AKT": "akash-network",
    
    # Gaming & NFT
    "AXS": "axie-infinity", "SAND": "the-sandbox", "MANA": "decentraland",
    "ENJ": "enjincoin", "GALA": "gala", "ILV": "illuvium",
    "MAGIC": "magic", "GODS": "gods-unchained", "PRIME": "echelon-prime",
    "YGG": "yield-guild-games", "RON": "ronin",
    
    # Infrastructure
    "GRT": "the-graph", "ROSE": "oasis-network", "AR": "arweave",
    "STX": "blockstack", "KAVA": "kava", "INJ": "injective-protocol",
    "TIA": "celestia", "PYTH": "pyth-network", "JUP": "jupiter-exchange-solana",
    "JTO": "jito-governance-token", "BONK": "bonk", "WIF": "dogwifcoin",
    
    # Memes & Others
    "SHIB": "shiba-inu", "PEPE": "pepe", "FLOKI": "floki",
    "TURBO": "turbo", "WLD": "worldcoin-wld", "BLUR": "blur",
    "DYDX": "dydx-chain", "MASK": "mask-network", "ENS": "ethereum-name-service",
    "CHZ": "chiliz", "AUDIO": "audius", "SUPER": "superfarm",
    
    # Old Guard
    "ZEC": "zcash", "DASH": "dash", "NEO": "neo", "WAVES": "waves",
    "QTUM": "qtum", "ZIL": "zilliqa", "ICX": "icon",
    "RLC": "iexec-rlc", "OMG": "omisego", "BAT": "basic-attention-token",
    "KNC": "kyber-network-crystal", "ZRX": "0x"
}


class SmartScout:
    """
    🔍 Smart Scout with rate limiting and batch operations
    """
    
    def __init__(self):
        self.rate_limit_delay = 1.5  # Seconds between CoinGecko calls
        self.batch_size = 50  # Max coins per CoinGecko batch
        self.last_cg_call = 0
        
    def _rate_limit(self):
        """Respect CoinGecko rate limits"""
        elapsed = time.time() - self.last_cg_call
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_cg_call = time.time()
    
    def get_crypto_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch multiple crypto prices in one API call.
        CoinGecko allows up to 250 coins per request.
        """
        import requests
        
        prices = {}
        
        # Convert symbols to CoinGecko IDs
        ids_to_fetch = []
        symbol_to_id = {}
        for symbol in symbols:
            cg_id = CRYPTO_TO_COINGECKO.get(symbol.upper())
            if cg_id:
                ids_to_fetch.append(cg_id)
                symbol_to_id[cg_id] = symbol.upper()
        
        if not ids_to_fetch:
            return prices
        
        # Batch fetch in groups
        for i in range(0, len(ids_to_fetch), self.batch_size):
            batch = ids_to_fetch[i:i + self.batch_size]
            ids_str = ",".join(batch)
            
            self._rate_limit()
            
            try:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_str}&vs_currencies=usd"
                resp = requests.get(url, timeout=10)
                
                if resp.status_code == 200:
                    data = resp.json()
                    for cg_id, price_data in data.items():
                        symbol = symbol_to_id.get(cg_id)
                        if symbol and "usd" in price_data:
                            prices[symbol] = price_data["usd"]
                elif resp.status_code == 429:
                    LOGGER.warning("🔍 [SCOUT] CoinGecko rate limited, waiting 60s...")
                    time.sleep(60)
                    # Retry this batch
                    i -= self.batch_size
            except Exception as e:
                LOGGER.error(f"🔍 [SCOUT] Batch price error: {e}")
        
        return prices
    
    def get_stock_prices_batch(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch multiple stock prices.
        Uses Yahoo Finance batch endpoint.
        """
        import requests
        
        prices = {}
        
        # Yahoo allows multiple symbols
        symbols_str = ",".join(symbols)
        
        try:
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbols_str}"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=15)
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("quoteResponse", {}).get("result", [])
                for quote in results:
                    symbol = quote.get("symbol")
                    price = quote.get("regularMarketPrice")
                    if symbol and price:
                        prices[symbol] = float(price)
        except Exception as e:
            LOGGER.error(f"🔍 [SCOUT] Stock batch error: {e}")
        
        return prices
    
    def scout_stocks(self) -> Dict:
        """Scout all stocks with batch pricing"""
        from core.ghost_scout import ALL_STOCKS
        from core.money_game_engine import get_money_game
        
        game = get_money_game()
        results = {"scouted": 0, "failed": 0, "trades": []}
        
        LOGGER.info(f"🔍 [SCOUT] Scouting {len(ALL_STOCKS)} stocks...")
        
        # Get all prices at once
        prices = self.get_stock_prices_batch(ALL_STOCKS)
        LOGGER.info(f"🔍 [SCOUT] Got prices for {len(prices)} stocks")
        
        for symbol in ALL_STOCKS:
            price = prices.get(symbol)
            if not price:
                results["failed"] += 1
                continue
            
            # Make prediction
            prediction = self._predict(symbol, price)
            
            # Record trade
            trade_id = game.record_trade(
                symbol=symbol,
                asset_type="stock",
                direction=prediction["direction"],
                entry_price=price,
                target_price=prediction["target"],
                confidence=prediction["confidence"]
            )
            
            if trade_id > 0:
                results["scouted"] += 1
                results["trades"].append({
                    "id": trade_id,
                    "symbol": symbol,
                    "direction": prediction["direction"],
                    "entry": price,
                    "target": prediction["target"]
                })
        
        LOGGER.info(f"🔍 [SCOUT] Stocks complete: {results['scouted']} scouted, {results['failed']} failed")
        return results
    
    def scout_crypto(self) -> Dict:
        """Scout all crypto with batch pricing and rate limiting"""
        from core.ghost_scout import ALL_CRYPTO
        from core.money_game_engine import get_money_game
        
        game = get_money_game()
        results = {"scouted": 0, "failed": 0, "trades": []}
        
        LOGGER.info(f"🔍 [SCOUT] Scouting {len(ALL_CRYPTO)} crypto (with rate limiting)...")
        
        # Get all prices in batches
        prices = self.get_crypto_prices_batch(ALL_CRYPTO)
        LOGGER.info(f"🔍 [SCOUT] Got prices for {len(prices)} crypto")
        
        for symbol in ALL_CRYPTO:
            price = prices.get(symbol.upper())
            if not price:
                results["failed"] += 1
                continue
            
            # Make prediction
            prediction = self._predict(symbol, price)
            
            # Record trade
            trade_id = game.record_trade(
                symbol=symbol,
                asset_type="crypto",
                direction=prediction["direction"],
                entry_price=price,
                target_price=prediction["target"],
                confidence=prediction["confidence"]
            )
            
            if trade_id > 0:
                results["scouted"] += 1
                results["trades"].append({
                    "id": trade_id,
                    "symbol": symbol,
                    "direction": prediction["direction"],
                    "entry": price,
                    "target": prediction["target"]
                })
        
        LOGGER.info(f"🔍 [SCOUT] Crypto complete: {results['scouted']} scouted, {results['failed']} failed")
        return results
    
    def _predict(self, symbol: str, current_price: float) -> Dict:
        """
        Make a prediction for an asset.
        
        This integrates with Ghost's actual prediction engine when available.
        """
        # Try to use the real predictor
        try:
            from core.multi_crypto_predictor import MultiCryptoPredictor
            predictor = MultiCryptoPredictor()
            result = predictor.predict_symbol(symbol)
            
            if result and "confidence" in result:
                conf = abs(result["confidence"])
                direction = "BUY" if result["confidence"] > 0 else "SELL"
                
                if direction == "BUY":
                    target = current_price * (1 + (conf * 0.08))
                else:
                    target = current_price * (1 - (conf * 0.08))
                
                return {
                    "direction": direction,
                    "target": round(target, 6),
                    "confidence": conf
                }
        except:
            pass
        
        # Default: slight bullish bias
        direction = "BUY"
        confidence = 0.55
        target = current_price * 1.03  # 3% target
        
        return {
            "direction": direction,
            "target": round(target, 6),
            "confidence": confidence
        }
    
    def full_scout(self) -> Dict:
        """Run complete scouting cycle"""
        LOGGER.info("🔍 [SCOUT] ═══════════════════════════════════════")
        LOGGER.info("🔍 [SCOUT] FULL SCOUTING CYCLE STARTING")
        LOGGER.info("🔍 [SCOUT] ═══════════════════════════════════════")
        
        start = time.time()
        
        stocks = self.scout_stocks()
        crypto = self.scout_crypto()
        
        elapsed = time.time() - start
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "stocks": stocks,
            "crypto": crypto,
            "total_scouted": stocks["scouted"] + crypto["scouted"],
            "total_failed": stocks["failed"] + crypto["failed"]
        }
        
        LOGGER.info(f"🔍 [SCOUT] ═══════════════════════════════════════")
        LOGGER.info(f"🔍 [SCOUT] COMPLETE: {results['total_scouted']} assets in {elapsed:.1f}s")
        LOGGER.info(f"🔍 [SCOUT] ═══════════════════════════════════════")
        
        return results


class MoneyGameCron:
    """
    ⏰ Cron handler for Money Game automation
    
    Daily schedule:
    - 6:00 AM CT: Scout all assets
    - 7:00 AM CT: Resolve trades from 24h ago
    - 7:30 AM CT: Update rankings
    - 8:00 AM CT: Send TOP 10 alerts
    """
    
    def __init__(self):
        self.scout = SmartScout()
    
    def run_daily_scout(self) -> Dict:
        """Run the daily scouting cycle"""
        LOGGER.info("⏰ [CRON] Daily scout starting...")
        return self.scout.full_scout()
    
    def run_daily_resolve(self, hours: int = 24) -> Dict:
        """Resolve trades from X hours ago"""
        from core.ghost_scout import resolve_trades
        
        LOGGER.info(f"⏰ [CRON] Resolving trades older than {hours}h...")
        return resolve_trades(hours)
    
    def run_daily_rankings(self) -> Dict:
        """Update rankings after resolving"""
        from core.money_game_engine import get_money_game
        
        LOGGER.info("⏰ [CRON] Updating rankings...")
        game = get_money_game()
        return game.update_rankings()
    
    def get_elite_for_alerts(self) -> Dict:
        """
        Get the TOP 10 elite assets for Telegram alerts.
        
        Returns the proven money makers that should be sent
        in the 8 AM daily alert.
        """
        from core.money_game_engine import get_money_game
        
        game = get_money_game()
        
        elite_stocks = game.get_elite_stocks()
        elite_crypto = game.get_elite_crypto()
        
        # Get full stats for each
        stock_details = []
        for symbol in elite_stocks[:10]:
            stats = game.get_player_stats(symbol)
            if stats:
                stock_details.append(stats)
        
        crypto_details = []
        for symbol in elite_crypto[:10]:
            stats = game.get_player_stats(symbol)
            if stats:
                crypto_details.append(stats)
        
        return {
            "elite_stocks": elite_stocks[:10],
            "elite_crypto": elite_crypto[:10],
            "stock_details": stock_details,
            "crypto_details": crypto_details,
            "message": "These are the PROVEN money makers from the Money Game!"
        }
    
    def run_full_daily_cycle(self) -> Dict:
        """
        Run the complete daily cycle:
        1. Scout all assets
        2. Resolve yesterday's trades
        3. Update rankings
        4. Return elite for alerts
        """
        LOGGER.info("⏰ [CRON] ═══════════════════════════════════════")
        LOGGER.info("⏰ [CRON] FULL DAILY CYCLE")
        LOGGER.info("⏰ [CRON] ═══════════════════════════════════════")
        
        results = {}
        
        # Step 1: Scout
        results["scout"] = self.run_daily_scout()
        
        # Step 2: Resolve (24h old trades)
        results["resolve"] = self.run_daily_resolve(24)
        
        # Step 3: Rankings
        results["rankings"] = self.run_daily_rankings()
        
        # Step 4: Get elite
        results["elite"] = self.get_elite_for_alerts()
        
        LOGGER.info("⏰ [CRON] ═══════════════════════════════════════")
        LOGGER.info("⏰ [CRON] DAILY CYCLE COMPLETE")
        LOGGER.info("⏰ [CRON] ═══════════════════════════════════════")
        
        return results


# Convenience functions
def smart_scout_all() -> Dict:
    """Run smart scouting with rate limiting"""
    scout = SmartScout()
    return scout.full_scout()


def run_daily_cycle() -> Dict:
    """Run complete daily Money Game cycle"""
    cron = MoneyGameCron()
    return cron.run_full_daily_cycle()


def get_elite_predictions() -> Dict:
    """Get elite assets for Telegram alerts"""
    cron = MoneyGameCron()
    return cron.get_elite_for_alerts()
