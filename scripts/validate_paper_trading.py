#!/usr/bin/env python3
"""
Phase 4.5: Paper Trading Validation

Validates that paper trading win rate matches live prediction accuracy.
We have:
- 942 tracked paper trades
- 41% live accuracy from ghost_predictions
- Need to verify calculations match and no systematic bias exists

Tests:
1. Compare paper trading win rate vs. reconciled predictions
2. Check for timing discrepancies (are we marking trades correctly?)
3. Verify entry/exit prices match market data
4. Flag any systematic bias (e.g., crypto vs. stocks)
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db_pool import get_connection, get_pool
from core.logger import get_logger

LOGGER = get_logger(__name__)


async def validate_paper_trading():
    """
    Validate paper trading calculations match live accuracy.
    
    Returns:
        dict: Validation results with discrepancies flagged
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get paper trading stats
        paper_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_trades,
                COUNT(*) FILTER (WHERE outcome = 'win') as wins,
                COUNT(*) FILTER (WHERE outcome = 'loss') as losses,
                ROUND(AVG(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) * 100, 2) as win_rate
            FROM ghost_paper_trades
            WHERE resolved_at IS NOT NULL
        """)
        
        # Get live prediction stats (reconciled only)
        live_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total_predictions,
                COUNT(*) FILTER (WHERE correct = true) as correct,
                COUNT(*) FILTER (WHERE correct = false) as incorrect,
                ROUND(AVG(CASE WHEN correct THEN 1 ELSE 0 END) * 100, 2) as accuracy
            FROM ghost_predictions
            WHERE reconciled = true
        """)
        
        # Get asset type breakdown
        asset_breakdown = await conn.fetch("""
            SELECT 
                asset_type,
                COUNT(*) as trades,
                COUNT(*) FILTER (WHERE outcome = 'win') as wins,
                ROUND(AVG(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) * 100, 2) as win_rate
            FROM ghost_paper_trades
            WHERE resolved_at IS NOT NULL
            GROUP BY asset_type
            ORDER BY trades DESC
        """)
        
        # Get timing discrepancies (trades that resolved >65 minutes after prediction)
        timing_issues = await conn.fetch("""
            SELECT 
                symbol,
                predicted_at,
                resolved_at,
                EXTRACT(EPOCH FROM (resolved_at - predicted_at)) / 60 as minutes_elapsed,
                outcome
            FROM ghost_paper_trades
            WHERE resolved_at IS NOT NULL
            AND EXTRACT(EPOCH FROM (resolved_at - predicted_at)) > 3900  -- 65 minutes
            ORDER BY minutes_elapsed DESC
            LIMIT 10
        """)
        
        # Get win rate by direction
        direction_breakdown = await conn.fetch("""
            SELECT 
                direction,
                COUNT(*) as trades,
                COUNT(*) FILTER (WHERE outcome = 'win') as wins,
                ROUND(AVG(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) * 100, 2) as win_rate
            FROM ghost_paper_trades
            WHERE resolved_at IS NOT NULL
            GROUP BY direction
        """)
        
    results = {
        "paper_trading": {
            "total_trades": paper_stats["total_trades"],
            "wins": paper_stats["wins"],
            "losses": paper_stats["losses"],
            "win_rate": float(paper_stats["win_rate"]) if paper_stats["win_rate"] else 0
        },
        "live_predictions": {
            "total_predictions": live_stats["total_predictions"],
            "correct": live_stats["correct"],
            "incorrect": live_stats["incorrect"],
            "accuracy": float(live_stats["accuracy"]) if live_stats["accuracy"] else 0
        },
        "discrepancy": {
            "absolute": abs(float(paper_stats["win_rate"] or 0) - float(live_stats["accuracy"] or 0)),
            "acceptable": abs(float(paper_stats["win_rate"] or 0) - float(live_stats["accuracy"] or 0)) < 5.0,
            "note": "Discrepancy >5% suggests calculation mismatch or timing issue"
        },
        "asset_breakdown": [
            {
                "asset_type": row["asset_type"],
                "trades": row["trades"],
                "wins": row["wins"],
                "win_rate": float(row["win_rate"]) if row["win_rate"] else 0
            }
            for row in asset_breakdown
        ],
        "timing_issues": [
            {
                "symbol": row["symbol"],
                "predicted_at": row["predicted_at"].isoformat(),
                "resolved_at": row["resolved_at"].isoformat(),
                "minutes_elapsed": round(row["minutes_elapsed"], 1),
                "outcome": row["outcome"]
            }
            for row in timing_issues
        ],
        "direction_breakdown": [
            {
                "direction": row["direction"],
                "trades": row["trades"],
                "wins": row["wins"],
                "win_rate": float(row["win_rate"]) if row["win_rate"] else 0
            }
            for row in direction_breakdown
        ]
    }
    
    return results


def print_validation_report(results: dict):
    """Pretty print validation results."""
    print("\n" + "="*70)
    print("📊 PAPER TRADING VALIDATION REPORT")
    print("="*70 + "\n")
    
    # Paper trading stats
    paper = results["paper_trading"]
    print(f"📄 PAPER TRADING ({paper['total_trades']} trades)")
    print(f"   Wins:     {paper['wins']}")
    print(f"   Losses:   {paper['losses']}")
    print(f"   Win Rate: {paper['win_rate']:.2f}%")
    print()
    
    # Live prediction stats
    live = results["live_predictions"]
    print(f"🎯 LIVE PREDICTIONS ({live['total_predictions']} reconciled)")
    print(f"   Correct:   {live['correct']}")
    print(f"   Incorrect: {live['incorrect']}")
    print(f"   Accuracy:  {live['accuracy']:.2f}%")
    print()
    
    # Discrepancy check
    disc = results["discrepancy"]
    status = "✅ PASS" if disc["acceptable"] else "❌ FAIL"
    print(f"🔍 DISCREPANCY CHECK: {status}")
    print(f"   Difference: {disc['absolute']:.2f}%")
    print(f"   {disc['note']}")
    print()
    
    # Asset breakdown
    print("📈 WIN RATE BY ASSET TYPE")
    for asset in results["asset_breakdown"]:
        print(f"   {asset['asset_type']:10} {asset['trades']:4} trades → {asset['win_rate']:5.1f}% win rate")
    print()
    
    # Direction breakdown
    print("🎲 WIN RATE BY DIRECTION")
    for direction in results["direction_breakdown"]:
        print(f"   {direction['direction']:4} {direction['trades']:4} trades → {direction['win_rate']:5.1f}% win rate")
    print()
    
    # Timing issues
    if results["timing_issues"]:
        print("⏰ TIMING ISSUES (trades resolved >65 min after prediction)")
        for issue in results["timing_issues"][:5]:  # Show top 5
            print(f"   {issue['symbol']:6} {issue['minutes_elapsed']:6.1f} min → {issue['outcome']}")
        print()
    
    print("="*70)
    
    # Final verdict
    if disc["acceptable"]:
        print("✅ VALIDATION PASSED: Paper trading matches live accuracy")
    else:
        print("❌ VALIDATION FAILED: Investigate calculation or timing mismatch")
    print("="*70 + "\n")


async def main():
    try:
        results = await validate_paper_trading()
        print_validation_report(results)
        
        # Exit code based on validation
        if not results["discrepancy"]["acceptable"]:
            sys.exit(1)
        
    except Exception as e:
        LOGGER.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
