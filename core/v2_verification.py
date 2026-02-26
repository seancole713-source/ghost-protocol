#!/usr/bin/env python3
"""
🎯 GHOST PROTOCOL V2 - DATA VERIFICATION SYSTEM

This module provides GROUND TRUTH about Ghost's actual performance.
No more guessing - query the database for verified outcomes.

Phase 1: Clean the Data
- Verify true win rate
- Identify winning vs losing assets
- Create performance dashboard

IMPORTANT: V2 was launched Jan 14, 2026. All performance queries
should ONLY use data from this date forward to avoid pre-V2 garbage.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import psycopg2
from dataclasses import dataclass

LOGGER = logging.getLogger("ghost.v2_verification")

# V2 LAUNCH DATE - Only use data from this date forward!
# Pre-V2 data has ~20% win rate (garbage predictions)
V2_START_DATE = datetime(2026, 1, 14, 0, 0, 0)


@dataclass
class SymbolPerformance:
    """Performance metrics for a single symbol"""
    symbol: str
    asset_type: str
    total_predictions: int
    wins: int
    losses: int
    win_rate: float
    avg_confidence: float
    recent_performance: str  # "improving", "stable", "declining"


@dataclass
class PerformanceReport:
    """Complete performance analysis"""
    period: str
    total_predictions: int
    verified_wins: int
    verified_losses: int
    win_rate: float
    by_symbol: List[SymbolPerformance]
    by_asset_type: Dict[str, Dict[str, Any]]
    top_performers: List[str]
    bottom_performers: List[str]


class V2VerificationSystem:
    """
    Ground truth verification for Ghost Protocol V2.
    
    Answers critical questions:
    1. What is our ACTUAL win rate?
    2. Which assets do we predict well?
    3. Which assets should we stop predicting?
    4. Are we improving over time?
    """
    
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL not set")
        
        LOGGER.info("[V2] Verification system initialized")
    
    def _get_conn(self):
        """Get PostgreSQL connection via shared pool bridge."""
        from core.db_pool import get_sync_connection
        return get_sync_connection().__enter__()
    
    def get_verified_win_rate(self, days: int = 14) -> Dict[str, Any]:
        """
        Get VERIFIED win rate from last N days.
        
        Only counts predictions where:
        1. Outcome was logged
        2. Direction and target are valid
        3. We can verify against actual price movement
        
        Returns:
            {
                "total_predictions": int,
                "verified_wins": int,
                "verified_losses": int,
                "win_rate": float,
                "period_start": datetime,
                "period_end": datetime
            }
        """
        conn = self._get_conn()
        cur = conn.cursor()
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff.isoformat()  # Convert to string for TEXT column comparison
        
        # Query paper_trades for verified outcomes
        # Note: created_at might be TEXT, so use CAST to ensure proper comparison
        cur.execute("""
            SELECT 
                symbol,
                signal_direction,
                entry_price,
                target_price,
                outcome,
                profit_loss_pct,
                created_at
            FROM paper_trades
            WHERE CAST(created_at AS TIMESTAMP) > CAST(%s AS TIMESTAMP)
            AND outcome IS NOT NULL
            AND outcome != 'PENDING'
            AND entry_price > 0
            AND target_price > 0
            ORDER BY created_at DESC
        """, (cutoff_str,))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        if not rows:
            LOGGER.warning(f"[V2] No verified predictions found in last {days} days")
            return {
                "total_predictions": 0,
                "verified_wins": 0,
                "verified_losses": 0,
                "win_rate": 0.0,
                "period_start": cutoff,
                "period_end": datetime.utcnow()
            }
        
        total = len(rows)
        wins = sum(1 for row in rows if row[4] == 'WIN')
        losses = sum(1 for row in rows if row[4] == 'LOSS')
        
        win_rate = (wins / total) * 100 if total > 0 else 0.0
        
        LOGGER.info(f"[V2] Verified {days}d: {wins}/{total} wins ({win_rate:.1f}%)")
        
        return {
            "total_predictions": total,
            "verified_wins": wins,
            "verified_losses": losses,
            "win_rate": win_rate,
            "period_start": cutoff,
            "period_end": datetime.utcnow(),
            "data": rows  # Include raw data for further analysis
        }
    
    def get_symbol_performance(self, days: int = 30, min_predictions: int = 10) -> List[SymbolPerformance]:
        """
        Analyze performance by symbol.
        
        Returns list of SymbolPerformance objects, sorted by win rate.
        Only includes symbols with at least min_predictions.
        
        IMPORTANT: Only uses V2-era data (since Jan 14, 2026) to avoid
        polluting results with pre-V2 garbage predictions.
        """
        conn = self._get_conn()
        cur = conn.cursor()
        
        # Use the LATER of (days ago) or V2_START_DATE
        # This ensures we NEVER use pre-V2 garbage data
        days_ago = datetime.utcnow() - timedelta(days=days)
        cutoff = max(days_ago, V2_START_DATE)
        cutoff_str = cutoff.isoformat()  # Convert to string for TEXT column
        
        LOGGER.info(f"[V2] Using cutoff {cutoff.isoformat()} (V2 floor: {V2_START_DATE.isoformat()})")
        
        cur.execute("""
            SELECT 
                symbol,
                CASE 
                    WHEN symbol IN ('BTC', 'ETH', 'SOL', 'ADA', 'AVAX', 'BNB', 'DOGE', 'DOT', 'LINK', 'LTC', 'VET', 'XLM', 'XRP')
                    THEN 'crypto'
                    ELSE 'stock'
                END as asset_type,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                ROUND(100.0 * SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate,
                AVG(signal_confidence) as avg_confidence
            FROM paper_trades
            WHERE CAST(created_at AS TIMESTAMP) > CAST(%s AS TIMESTAMP)
            AND outcome IS NOT NULL
            AND outcome != 'PENDING'
            GROUP BY symbol
            HAVING COUNT(*) >= %s
            ORDER BY win_rate DESC
        """, (cutoff, min_predictions))
        
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        results = []
        for row in rows:
            symbol, asset_type, total, wins, losses, win_rate, avg_conf = row
            
            # Determine trend (compare first half vs second half of period)
            trend = self._get_performance_trend(symbol, days)
            
            results.append(SymbolPerformance(
                symbol=symbol,
                asset_type=asset_type,
                total_predictions=total,
                wins=wins,
                losses=losses,
                win_rate=win_rate,
                avg_confidence=float(avg_conf) if avg_conf else 0.0,
                recent_performance=trend
            ))
        
        LOGGER.info(f"[V2] Analyzed {len(results)} symbols with {min_predictions}+ predictions")
        return results
    
    def _get_performance_trend(self, symbol: str, days: int) -> str:
        """
        Determine if symbol performance is improving, stable, or declining.
        Compares first half vs second half of period.
        """
        try:
            conn = self._get_conn()
            cur = conn.cursor()
            
            cutoff = datetime.utcnow() - timedelta(days=days)
            midpoint = cutoff + timedelta(days=days // 2)
            cutoff_str = cutoff.isoformat()
            midpoint_str = midpoint.isoformat()
            
            # First half win rate
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins
                FROM paper_trades
                WHERE symbol = %s
                AND CAST(created_at AS TIMESTAMP) BETWEEN CAST(%s AS TIMESTAMP) AND CAST(%s AS TIMESTAMP)
                AND outcome IS NOT NULL
                AND outcome != 'PENDING'
            """, (symbol, cutoff_str, midpoint_str))
            
            first_half = cur.fetchone()
            first_wr = (first_half[1] / first_half[0]) if first_half[0] > 0 else 0
            
            # Second half win rate
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins
                FROM paper_trades
                WHERE symbol = %s
                AND CAST(created_at AS TIMESTAMP) > CAST(%s AS TIMESTAMP)
                AND outcome IS NOT NULL
                AND outcome != 'PENDING'
            """, (symbol, midpoint_str))
            
            second_half = cur.fetchone()
            second_wr = (second_half[1] / second_half[0]) if second_half[0] > 0 else 0
            
            cur.close()
            conn.close()
            
            # Determine trend
            if second_wr > first_wr + 0.1:  # 10% improvement
                return "improving"
            elif second_wr < first_wr - 0.1:  # 10% decline
                return "declining"
            else:
                return "stable"
        
        except Exception as e:
            LOGGER.debug(f"[V2] Trend analysis failed for {symbol}: {e}")
            return "unknown"
    
    def get_top_performers(self, days: int = 30, min_predictions: int = 20, top_n: int = 20) -> List[str]:
        """
        Get TOP N performing symbols.
        
        Criteria:
        - At least min_predictions in period
        - Sorted by win rate
        - Returns symbol names only
        """
        performances = self.get_symbol_performance(days, min_predictions)
        top = performances[:top_n]
        
        symbols = [p.symbol for p in top]
        LOGGER.info(f"[V2] Top {top_n} performers: {', '.join(symbols)}")
        return symbols
    
    def get_bottom_performers(self, days: int = 30, min_predictions: int = 20, bottom_n: int = 20) -> List[str]:
        """
        Get BOTTOM N performing symbols (candidates for blacklist).
        
        Criteria:
        - At least min_predictions in period
        - Sorted by win rate (worst first)
        - Returns symbol names only
        """
        performances = self.get_symbol_performance(days, min_predictions)
        bottom = performances[-bottom_n:] if len(performances) >= bottom_n else performances
        
        symbols = [p.symbol for p in bottom]
        LOGGER.info(f"[V2] Bottom {bottom_n} performers: {', '.join(symbols)}")
        return symbols
    
    def generate_performance_report(self, days: int = 14) -> PerformanceReport:
        """
        Generate comprehensive performance report.
        
        This is the "ground truth" dashboard for Ghost Protocol V2.
        """
        # Overall metrics
        overall = self.get_verified_win_rate(days)
        
        # Per-symbol breakdown
        symbol_perf = self.get_symbol_performance(days, min_predictions=5)
        
        # By asset type
        crypto_perf = [p for p in symbol_perf if p.asset_type == 'crypto']
        stock_perf = [p for p in symbol_perf if p.asset_type == 'stock']
        
        crypto_total = sum(p.total_predictions for p in crypto_perf)
        crypto_wins = sum(p.wins for p in crypto_perf)
        crypto_wr = (crypto_wins / crypto_total * 100) if crypto_total > 0 else 0
        
        stock_total = sum(p.total_predictions for p in stock_perf)
        stock_wins = sum(p.wins for p in stock_perf)
        stock_wr = (stock_wins / stock_total * 100) if stock_total > 0 else 0
        
        # Top/bottom performers
        top = [p.symbol for p in symbol_perf[:10]]
        bottom = [p.symbol for p in symbol_perf[-10:]]
        
        report = PerformanceReport(
            period=f"last_{days}_days",
            total_predictions=overall['total_predictions'],
            verified_wins=overall['verified_wins'],
            verified_losses=overall['verified_losses'],
            win_rate=overall['win_rate'],
            by_symbol=symbol_perf,
            by_asset_type={
                "crypto": {
                    "total": crypto_total,
                    "wins": crypto_wins,
                    "win_rate": crypto_wr
                },
                "stock": {
                    "total": stock_total,
                    "wins": stock_wins,
                    "win_rate": stock_wr
                }
            },
            top_performers=top,
            bottom_performers=bottom
        )
        
        LOGGER.info(f"[V2] Performance report generated: {days}d, {overall['total_predictions']} predictions, {overall['win_rate']:.1f}% WR")
        return report
    
    def recommend_whitelist_blacklist(self, days: int = 30) -> Dict[str, List[str]]:
        """
        Recommend which assets to keep (whitelist) vs drop (blacklist).
        
        Whitelist criteria:
        - Win rate >= 55%
        - At least 20 predictions
        - Not declining
        
        Blacklist criteria:
        - Win rate < 45%
        - At least 20 predictions
        - Declining or stable (not improving)
        """
        performances = self.get_symbol_performance(days, min_predictions=20)
        
        whitelist = []
        blacklist = []
        
        for p in performances:
            if p.win_rate >= 55.0 and p.recent_performance != "declining":
                whitelist.append(p.symbol)
            elif p.win_rate < 45.0:
                blacklist.append(p.symbol)
        
        LOGGER.info(f"[V2] Recommendation: {len(whitelist)} whitelist, {len(blacklist)} blacklist")
        return {
            "whitelist": whitelist,
            "blacklist": blacklist,
            "criteria": {
                "whitelist": "win_rate >= 55%, min 20 predictions, not declining",
                "blacklist": "win_rate < 45%, min 20 predictions"
            }
        }


# ============================================================================
# Singleton
# ============================================================================

_verifier: Optional[V2VerificationSystem] = None

def get_verifier() -> V2VerificationSystem:
    """Get singleton verifier"""
    global _verifier
    if _verifier is None:
        _verifier = V2VerificationSystem()
    return _verifier


# ============================================================================
# CLI for testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    verifier = get_verifier()
    
    print("\n" + "=" * 70)
    print("🎯 GHOST PROTOCOL V2 - PERFORMANCE VERIFICATION")
    print("=" * 70)
    
    # Last 14 days
    print("\n📊 LAST 14 DAYS:")
    wr_14d = verifier.get_verified_win_rate(14)
    print(f"   Total Predictions: {wr_14d['total_predictions']}")
    print(f"   Wins: {wr_14d['verified_wins']}")
    print(f"   Losses: {wr_14d['verified_losses']}")
    print(f"   Win Rate: {wr_14d['win_rate']:.1f}%")
    
    # Last 30 days
    print("\n📊 LAST 30 DAYS:")
    wr_30d = verifier.get_verified_win_rate(30)
    print(f"   Total Predictions: {wr_30d['total_predictions']}")
    print(f"   Wins: {wr_30d['verified_wins']}")
    print(f"   Losses: {wr_30d['verified_losses']}")
    print(f"   Win Rate: {wr_30d['win_rate']:.1f}%")
    
    # Top performers
    print("\n🏆 TOP 10 PERFORMING SYMBOLS (30d, min 20 predictions):")
    top = verifier.get_symbol_performance(30, 20)[:10]
    for i, p in enumerate(top, 1):
        emoji = "📈" if p.recent_performance == "improving" else "📊" if p.recent_performance == "stable" else "📉"
        print(f"   {i:2d}. {p.symbol:6s} — {p.win_rate:5.1f}% ({p.wins}/{p.total_predictions}) {emoji} {p.asset_type}")
    
    # Bottom performers
    print("\n⚠️  BOTTOM 10 PERFORMING SYMBOLS (30d, min 20 predictions):")
    bottom = verifier.get_symbol_performance(30, 20)[-10:]
    for i, p in enumerate(bottom, 1):
        emoji = "📈" if p.recent_performance == "improving" else "📊" if p.recent_performance == "stable" else "📉"
        print(f"   {i:2d}. {p.symbol:6s} — {p.win_rate:5.1f}% ({p.wins}/{p.total_predictions}) {emoji} {p.asset_type}")
    
    # Recommendations
    print("\n💡 WHITELIST/BLACKLIST RECOMMENDATIONS:")
    rec = verifier.recommend_whitelist_blacklist(30)
    print(f"   ✅ WHITELIST ({len(rec['whitelist'])}): {', '.join(rec['whitelist'][:15])}")
    if len(rec['whitelist']) > 15:
        print(f"      ... and {len(rec['whitelist']) - 15} more")
    print(f"   ❌ BLACKLIST ({len(rec['blacklist'])}): {', '.join(rec['blacklist'])}")
    
    print("\n" + "=" * 70)
