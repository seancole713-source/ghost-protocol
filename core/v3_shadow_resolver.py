#!/usr/bin/env python3
"""
🎯 GHOST V3 - SHADOW OUTCOME RESOLVER

Resolves shadow predictions to track competition performance.
Runs hourly to check if 48h window has passed and determine WIN/LOSS.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

LOGGER = logging.getLogger("ghost.v3_resolver")

DATABASE_URL = os.getenv("DATABASE_URL")


class ShadowOutcomeResolver:
    """
    Resolves shadow predictions after their 48h window expires.
    Updates competitor win/loss records.
    """
    
    def __init__(self):
        self.use_postgres = bool(DATABASE_URL)
        if not self.use_postgres:
            LOGGER.warning("[RESOLVER] No DATABASE_URL - shadow resolution disabled")
    
    def _get_connection(self):
        """Get PostgreSQL connection via shared pool bridge."""
        from core.db_pool import get_sync_connection_raw
        return get_sync_connection_raw()
    
    async def resolve_pending(self, batch_size: int = 100) -> Dict:
        """
        Resolve all shadow predictions whose target_time has passed.
        
        Returns summary of resolutions.
        """
        if not self.use_postgres:
            return {"error": "No database"}
        
        from core.v3_competition import get_competition_system
        
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "resolved": 0,
            "wins": 0,
            "losses": 0,
            "errors": 0,
            "details": []
        }
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Get pending predictions where target_time has passed
            cur.execute("""
                SELECT id, symbol, asset_type, direction, entry_price, target_price, target_time
                FROM v3_shadow_predictions
                WHERE outcome IS NULL
                  AND target_time <= NOW()
                ORDER BY target_time ASC
                LIMIT %s
            """, (batch_size,))
            
            pending = cur.fetchall()
            LOGGER.info(f"[RESOLVER] Found {len(pending)} shadow predictions to resolve")
            
            competition = get_competition_system()
            
            for pred in pending:
                pred_id, symbol, asset_type, direction, entry_price, target_price, target_time = pred
                
                try:
                    # Get current price
                    current_price = await self._get_current_price(symbol, asset_type)
                    
                    if current_price is None:
                        results["errors"] += 1
                        continue
                    
                    # Determine outcome
                    if direction == "BUY":
                        # BUY wins if price went up toward target
                        outcome = "WIN" if current_price >= target_price else "LOSS"
                    else:
                        # SELL wins if price went down toward target  
                        outcome = "WIN" if current_price <= target_price else "LOSS"
                    
                    # Update database
                    cur.execute("""
                        UPDATE v3_shadow_predictions
                        SET outcome = %s, final_price = %s, resolved_at = NOW()
                        WHERE id = %s
                    """, (outcome, current_price, pred_id))
                    
                    # Update competitor metrics
                    cur.execute("""
                        UPDATE v3_competition_pool
                        SET 
                            total_predictions = total_predictions + 1,
                            wins = wins + CASE WHEN %s = 'WIN' THEN 1 ELSE 0 END,
                            losses = losses + CASE WHEN %s = 'LOSS' THEN 1 ELSE 0 END,
                            last_prediction = NOW(),
                            last_updated = NOW()
                        WHERE symbol = %s
                    """, (outcome, outcome, symbol))
                    
                    results["resolved"] += 1
                    if outcome == "WIN":
                        results["wins"] += 1
                    else:
                        results["losses"] += 1
                    
                    results["details"].append({
                        "symbol": symbol,
                        "direction": direction,
                        "entry": entry_price,
                        "target": target_price,
                        "final": current_price,
                        "outcome": outcome
                    })
                    
                except Exception as e:
                    LOGGER.error(f"[RESOLVER] Failed to resolve {symbol}: {e}")
                    results["errors"] += 1
            
            conn.commit()
            conn.close()
            
            # Update rankings after resolving batch
            if results["resolved"] > 0:
                LOGGER.info("[RESOLVER] Triggering ranking update...")
                competition.update_rankings()
            
            LOGGER.info(f"[RESOLVER] ✅ Resolved {results['resolved']}: {results['wins']} wins, {results['losses']} losses")
            return results
            
        except Exception as e:
            LOGGER.error(f"[RESOLVER] Failed to resolve pending: {e}")
            return {"error": str(e)}
    
    async def _get_current_price(self, symbol: str, asset_type: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            if asset_type == "crypto":
                return await self._get_crypto_price(symbol)
            else:
                return await self._get_stock_price(symbol)
        except Exception as e:
            LOGGER.error(f"[RESOLVER] Price fetch failed for {symbol}: {e}")
            return None
    
    async def _get_crypto_price(self, symbol: str) -> Optional[float]:
        """Get crypto price from CoinGecko or similar"""
        try:
            import aiohttp
            
            # Try CoinGecko
            async with aiohttp.ClientSession() as session:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={symbol.lower()}&vs_currencies=usd"
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if symbol.lower() in data:
                            return data[symbol.lower()]["usd"]
            
            # Fallback: Try Binance-style mapping
            symbol_map = {
                "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
                "ADA": "cardano", "XRP": "ripple", "DOT": "polkadot",
                "DOGE": "dogecoin", "AVAX": "avalanche-2", "MATIC": "matic-network",
                "LINK": "chainlink", "UNI": "uniswap", "LTC": "litecoin",
                "RNDR": "render-token", "FET": "fetch-ai", "OCEAN": "ocean-protocol",
                "CHZ": "chiliz", "ZEC": "zcash", "TURBO": "turbo"
            }
            
            if symbol.upper() in symbol_map:
                cg_id = symbol_map[symbol.upper()]
                async with aiohttp.ClientSession() as session:
                    url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
                    async with session.get(url, timeout=10) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if cg_id in data:
                                return data[cg_id]["usd"]
            
            return None
        except Exception as e:
            LOGGER.debug(f"[RESOLVER] Crypto price error for {symbol}: {e}")
            return None
    
    async def _get_stock_price(self, symbol: str) -> Optional[float]:
        """Get stock price from Yahoo Finance or similar"""
        try:
            import aiohttp
            
            # Try Yahoo Finance
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=1d"
            headers = {"User-Agent": "Mozilla/5.0"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("chart", {}).get("result", [])
                        if result:
                            meta = result[0].get("meta", {})
                            return meta.get("regularMarketPrice")
            
            return None
        except Exception as e:
            LOGGER.debug(f"[RESOLVER] Stock price error for {symbol}: {e}")
            return None
    
    def get_resolution_stats(self) -> Dict:
        """Get resolution statistics"""
        if not self.use_postgres:
            return {"error": "No database"}
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE outcome IS NULL) as pending,
                    COUNT(*) FILTER (WHERE outcome = 'WIN') as wins,
                    COUNT(*) FILTER (WHERE outcome = 'LOSS') as losses,
                    COUNT(*) as total
                FROM v3_shadow_predictions
            """)
            
            row = cur.fetchone()
            conn.close()
            
            return {
                "pending": row[0],
                "wins": row[1],
                "losses": row[2],
                "total": row[3],
                "win_rate": f"{(row[1]/(row[1]+row[2])*100):.1f}%" if (row[1]+row[2]) > 0 else "N/A"
            }
        except Exception as e:
            return {"error": str(e)}


# Singleton
_resolver: Optional[ShadowOutcomeResolver] = None


def get_shadow_resolver() -> ShadowOutcomeResolver:
    """Get or create resolver singleton"""
    global _resolver
    if _resolver is None:
        _resolver = ShadowOutcomeResolver()
    return _resolver


async def resolve_shadow_outcomes() -> Dict:
    """Convenience function to resolve pending outcomes"""
    resolver = get_shadow_resolver()
    return await resolver.resolve_pending()
