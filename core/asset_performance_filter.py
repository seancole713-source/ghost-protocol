"""
Asset Performance Filter for Ghost Protocol
============================================

Filters trading decisions based on historical win rates.
Only trade assets where Ghost has proven accuracy.

Design:
- BLACKLIST: Assets with <20% historical win rate (don't trade)
- WHITELIST: Assets with >50% historical win rate (prioritize)
- WATCHLIST: Assets with 20-50% win rate (monitor, lower confidence)
"""

import os
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

LOGGER = logging.getLogger(__name__)

# ============================================================================
# HISTORICAL PERFORMANCE DATA (from database analysis Jan 9, 2026)
# ============================================================================

# Assets with 0-20% win rate - DO NOT TRADE
BLACKLIST = {
    # Major cryptos - complete failures (0-3% win rate)
    "SOL": {"trades": 30, "wins": 0, "win_rate": 0.0, "reason": "0/30 - model doesn't understand SOL"},
    "ETH": {"trades": 29, "wins": 0, "win_rate": 0.0, "reason": "0/29 - model doesn't understand ETH"},
    "BNB": {"trades": 28, "wins": 0, "win_rate": 0.0, "reason": "0/28 - model doesn't understand BNB"},
    "XRP": {"trades": 28, "wins": 0, "win_rate": 0.0, "reason": "0/28 - model doesn't understand XRP"},
    "AVAX": {"trades": 27, "wins": 0, "win_rate": 0.0, "reason": "0/27 - model doesn't understand AVAX"},
    "LTC": {"trades": 26, "wins": 0, "win_rate": 0.0, "reason": "0/26 - model doesn't understand LTC"},
    "LINK": {"trades": 19, "wins": 0, "win_rate": 0.0, "reason": "0/19 - model doesn't understand LINK"},
    "DOGE": {"trades": 17, "wins": 0, "win_rate": 0.0, "reason": "0/17 - model doesn't understand DOGE"},
    "VET": {"trades": 16, "wins": 0, "win_rate": 0.0, "reason": "0/16 - model doesn't understand VET"},
    "ADA": {"trades": 15, "wins": 0, "win_rate": 0.0, "reason": "0/15 - model doesn't understand ADA"},
    "DOT": {"trades": 14, "wins": 0, "win_rate": 0.0, "reason": "0/14 - model doesn't understand DOT"},
    "XLM": {"trades": 16, "wins": 6, "win_rate": 37.5, "reason": "6/16 - below threshold"},
    "BTC": {"trades": 33, "wins": 1, "win_rate": 3.0, "reason": "1/33 - cannot predict BTC"},
}

# Assets with >50% win rate - PRIORITIZE
WHITELIST = {
    # Perfect performers (100% win rate)
    "CHZ": {"trades": 13, "wins": 13, "win_rate": 100.0, "category": "crypto", "note": "Stable low-vol"},
    "ZEC": {"trades": 7, "wins": 7, "win_rate": 100.0, "category": "crypto", "note": "Privacy coin"},
    "T": {"trades": 18, "wins": 18, "win_rate": 100.0, "category": "stock", "note": "AT&T - stable utility"},
    "ILV": {"trades": 13, "wins": 13, "win_rate": 100.0, "category": "crypto", "note": "Gaming token"},
    "RNDR": {"trades": 12, "wins": 12, "win_rate": 100.0, "category": "crypto", "note": "Render network"},
    "RLC": {"trades": 5, "wins": 5, "win_rate": 100.0, "category": "crypto", "note": "Cloud computing"},
    "EGLD": {"trades": 5, "wins": 5, "win_rate": 100.0, "category": "crypto", "note": "Elrond"},
    "TURBO": {"trades": 13, "wins": 13, "win_rate": 100.0, "category": "crypto", "note": "Meme token"},
    "DASH": {"trades": 1, "wins": 1, "win_rate": 100.0, "category": "crypto", "note": "Small sample"},
    "FLOW": {"trades": 7, "wins": 7, "win_rate": 100.0, "category": "crypto", "note": "NFT blockchain"},
    
    # Excellent performers (>90% win rate)
    "ICP": {"trades": 15, "wins": 14, "win_rate": 93.3, "category": "crypto", "note": "Internet Computer"},
    "BCH": {"trades": 16, "wins": 15, "win_rate": 93.8, "category": "crypto", "note": "Bitcoin Cash"},
    "OCEAN": {"trades": 10, "wins": 9, "win_rate": 90.0, "category": "crypto", "note": "Data marketplace"},
    
    # Strong performers (>70% win rate)
    "LRC": {"trades": 14, "wins": 12, "win_rate": 85.7, "category": "crypto", "note": "Loopring L2"},
    "CELO": {"trades": 12, "wins": 10, "win_rate": 83.3, "category": "crypto", "note": "Mobile payments"},
    "AAVE": {"trades": 14, "wins": 9, "win_rate": 64.3, "category": "crypto", "note": "DeFi lending"},
    "NMR": {"trades": 11, "wins": 8, "win_rate": 72.7, "category": "crypto", "note": "Numeraire"},
}

# Assets with 20-50% win rate - MONITOR (trade with caution)
WATCHLIST = {
    # These have some success but need careful monitoring
    # Will be populated dynamically from database
}


class AssetPerformanceFilter:
    """
    Filters trading decisions based on historical performance.
    """
    
    def __init__(self, db_conn=None):
        """
        Initialize filter.
        
        Args:
            db_conn: Optional database connection for dynamic updates
        """
        self.db_conn = db_conn
        self.blacklist = BLACKLIST
        self.whitelist = WHITELIST
        self.watchlist = WATCHLIST
        
        # Load dynamic data if database available
        if db_conn:
            self._update_from_database()
    
    def should_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        Check if symbol should be traded.
        
        Args:
            symbol: Trading symbol (e.g., "SOL", "CHZ")
        
        Returns:
            (should_trade: bool, reason: str)
        """
        symbol = symbol.upper().strip()
        
        # Check blacklist first
        if symbol in self.blacklist:
            info = self.blacklist[symbol]
            reason = info.get("reason", f"Blacklisted: {info['win_rate']:.1f}% win rate")
            LOGGER.warning(f"❌ {symbol} BLACKLISTED: {reason}")
            return False, reason
        
        # Check whitelist
        if symbol in self.whitelist:
            info = self.whitelist[symbol]
            reason = f"Whitelisted: {info['win_rate']:.1f}% win rate ({info['wins']}/{info['trades']})"
            LOGGER.info(f"✅ {symbol} WHITELISTED: {reason}")
            return True, reason
        
        # Check watchlist
        if symbol in self.watchlist:
            info = self.watchlist[symbol]
            reason = f"Watchlist: {info['win_rate']:.1f}% win rate - trade cautiously"
            LOGGER.info(f"⚠️ {symbol} WATCHLIST: {reason}")
            return True, reason  # Allow but with caution
        
        # Unknown symbol - allow by default but log
        LOGGER.info(f"🆕 {symbol} UNKNOWN - no historical data, allowing with default confidence")
        return True, "Unknown asset - no historical performance data"
    
    def get_confidence_adjustment(self, symbol: str, base_confidence: float) -> float:
        """
        Adjust confidence based on historical performance.
        
        Args:
            symbol: Trading symbol
            base_confidence: Original confidence (0.0-1.0)
        
        Returns:
            Adjusted confidence (0.0-1.0)
        """
        symbol = symbol.upper().strip()
        
        # Blacklisted - return 0 (don't trade)
        if symbol in self.blacklist:
            return 0.0
        
        # Whitelisted - boost confidence
        if symbol in self.whitelist:
            info = self.whitelist[symbol]
            win_rate = info["win_rate"] / 100.0
            
            # Boost based on win rate and sample size
            trades = info["trades"]
            if trades >= 10 and win_rate >= 0.90:
                # High confidence: 90%+ win rate with 10+ trades
                boost = 0.15
            elif trades >= 10 and win_rate >= 0.70:
                # Medium confidence: 70%+ win rate with 10+ trades
                boost = 0.10
            elif win_rate >= 0.90:
                # Small sample but perfect - moderate boost
                boost = 0.08
            else:
                # Good but cautious
                boost = 0.05
            
            adjusted = min(base_confidence + boost, 0.85)  # Cap at 85%
            LOGGER.info(f"✅ {symbol} confidence boost: {base_confidence:.1%} → {adjusted:.1%} (+{boost:.1%})")
            return adjusted
        
        # Watchlist - slight penalty
        if symbol in self.watchlist:
            info = self.watchlist[symbol]
            win_rate = info["win_rate"] / 100.0
            
            # Penalty based on how far below 50%
            if win_rate < 0.40:
                penalty = 0.10  # Bad performance
            elif win_rate < 0.45:
                penalty = 0.05  # Below average
            else:
                penalty = 0.02  # Slightly below average
            
            adjusted = max(base_confidence - penalty, 0.35)  # Floor at 35%
            LOGGER.info(f"⚠️ {symbol} confidence penalty: {base_confidence:.1%} → {adjusted:.1%} (-{penalty:.1%})")
            return adjusted
        
        # Unknown - use base confidence with slight penalty for uncertainty
        adjusted = base_confidence * 0.95  # 5% uncertainty discount
        LOGGER.debug(f"🆕 {symbol} unknown asset penalty: {base_confidence:.1%} → {adjusted:.1%}")
        return adjusted
    
    def get_position_size_multiplier(self, symbol: str) -> float:
        """
        Get position size multiplier based on historical performance.
        
        Args:
            symbol: Trading symbol
        
        Returns:
            Multiplier (0.0-2.0) where 1.0 is normal size
        """
        symbol = symbol.upper().strip()
        
        # Blacklisted - no position
        if symbol in self.blacklist:
            return 0.0
        
        # Whitelisted - larger positions
        if symbol in self.whitelist:
            info = self.whitelist[symbol]
            win_rate = info["win_rate"] / 100.0
            trades = info["trades"]
            
            if trades >= 10 and win_rate >= 0.90:
                return 1.5  # 50% larger position
            elif trades >= 10 and win_rate >= 0.70:
                return 1.3  # 30% larger position
            elif win_rate >= 0.90:
                return 1.2  # Small sample - moderate increase
            else:
                return 1.1  # Slight increase
        
        # Watchlist - smaller positions
        if symbol in self.watchlist:
            info = self.watchlist[symbol]
            win_rate = info["win_rate"] / 100.0
            
            if win_rate < 0.40:
                return 0.5  # Half size
            elif win_rate < 0.45:
                return 0.7  # 30% reduction
            else:
                return 0.8  # 20% reduction
        
        # Unknown - normal size
        return 1.0
    
    def _update_from_database(self):
        """
        Update whitelist/blacklist from live database stats.
        This allows the system to adapt as new data comes in.
        """
        if not self.db_conn:
            return
        
        try:
            # Query recent performance (last 90 days)
            query = """
                SELECT 
                    symbol,
                    COUNT(*) as trades,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    ROUND(100.0 * SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
                FROM paper_trades
                WHERE outcome IN ('WIN', 'LOSS')
                  AND created_at >= NOW() - INTERVAL '90 days'
                GROUP BY symbol
                HAVING COUNT(*) >= 5  -- Minimum 5 trades for significance
                ORDER BY win_rate DESC
            """
            
            cursor = self.db_conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            
            # Update lists based on fresh data
            for row in results:
                symbol = row[0]
                trades = row[1]
                wins = row[2]
                win_rate = float(row[3])
                
                info = {
                    "trades": trades,
                    "wins": wins,
                    "win_rate": win_rate,
                    "updated": datetime.now().isoformat()
                }
                
                # Categorize
                if win_rate < 20:
                    self.blacklist[symbol] = {**info, "reason": f"{wins}/{trades} - poor performance"}
                    LOGGER.warning(f"📊 Added {symbol} to BLACKLIST: {win_rate:.1f}% win rate")
                elif win_rate >= 50:
                    self.whitelist[symbol] = {**info, "category": "dynamic", "note": "Database updated"}
                    LOGGER.info(f"📊 Added {symbol} to WHITELIST: {win_rate:.1f}% win rate")
                else:
                    self.watchlist[symbol] = {**info}
                    LOGGER.debug(f"📊 Added {symbol} to WATCHLIST: {win_rate:.1f}% win rate")
            
            LOGGER.info(
                f"📊 Performance filter updated: "
                f"Blacklist={len(self.blacklist)}, "
                f"Whitelist={len(self.whitelist)}, "
                f"Watchlist={len(self.watchlist)}"
            )
        
        except Exception as e:
            LOGGER.error(f"Failed to update performance filter from database: {e}")
    
    def get_stats(self) -> Dict:
        """Get current filter statistics."""
        return {
            "blacklist_count": len(self.blacklist),
            "whitelist_count": len(self.whitelist),
            "watchlist_count": len(self.watchlist),
            "blacklist_symbols": list(self.blacklist.keys()),
            "whitelist_symbols": list(self.whitelist.keys()),
            "last_updated": datetime.now().isoformat()
        }


# Global singleton instance
_filter_instance: Optional[AssetPerformanceFilter] = None


def get_performance_filter(db_conn=None) -> AssetPerformanceFilter:
    """
    Get global performance filter instance.
    
    Args:
        db_conn: Optional database connection (used on first call)
    
    Returns:
        AssetPerformanceFilter instance
    """
    global _filter_instance
    
    if _filter_instance is None:
        _filter_instance = AssetPerformanceFilter(db_conn=db_conn)
        LOGGER.info("✅ Asset performance filter initialized")
    
    return _filter_instance


# Convenience functions for direct import
def should_trade_symbol(symbol: str) -> Tuple[bool, str]:
    """Check if symbol should be traded."""
    return get_performance_filter().should_trade(symbol)


def adjust_confidence_for_symbol(symbol: str, base_confidence: float) -> float:
    """Adjust confidence based on historical performance."""
    return get_performance_filter().get_confidence_adjustment(symbol, base_confidence)


def get_position_multiplier(symbol: str) -> float:
    """Get position size multiplier for symbol."""
    return get_performance_filter().get_position_size_multiplier(symbol)
