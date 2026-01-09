#!/usr/bin/env python3
"""
Re-evaluate all paper trades with corrected evaluation logic.

The old logic was broken - it triggered stop losses during the 6-48h period,
marking correct predictions as losses. This script re-evaluates all trades
using the FIXED logic that only checks outcome at target time.

Usage:
    python scripts/reevaluate_paper_trades.py --dry-run     # Preview changes
    python scripts/reevaluate_paper_trades.py               # Apply changes
"""

import os
import sys
import logging
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection from environment."""
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        raise ValueError("DATABASE_URL environment variable not set")
    
    # Convert postgres:// to postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    
    try:
        import psycopg2
        from urllib.parse import urlparse
        
        result = urlparse(db_url)
        conn = psycopg2.connect(
            database=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
        )
        conn.row_factory = psycopg2.extras.RealDictCursor
        return conn
    except Exception as e:
        LOGGER.error(f"Failed to connect to PostgreSQL: {e}")
        raise


def evaluate_trade_corrected(trade: Dict) -> Dict:
    """
    Evaluate trade using CORRECTED logic (outcome at target time only).
    
    Args:
        trade: Trade record from database
    
    Returns:
        {
            "outcome": "WIN|LOSS|BREAK_EVEN",
            "profit_loss_pct": float,
            "profit_loss": float
        }
    """
    entry_price = float(trade["entry_price"])
    target_price = float(trade.get("target_price") or entry_price)
    signal_direction = trade["signal_direction"]
    position_size = float(trade["position_size"])
    
    # Calculate price change at target time
    price_change_pct = (target_price - entry_price) / entry_price
    
    # Simple win/loss evaluation based on prediction direction
    is_up_prediction = signal_direction in ("LONG", "UP")
    is_down_prediction = signal_direction in ("SHORT", "DOWN")
    
    if is_up_prediction:
        # UP prediction: WIN if price went up at target time
        if price_change_pct > 0.01:  # Up more than 1%
            outcome = "WIN"
            pnl_pct = price_change_pct
        elif price_change_pct < -0.01:  # Down more than 1%
            outcome = "LOSS"
            pnl_pct = price_change_pct
        else:
            outcome = "BREAK_EVEN"
            pnl_pct = 0.0
    
    elif is_down_prediction:
        # DOWN prediction: WIN if price went down at target time
        if price_change_pct < -0.01:  # Down more than 1%
            outcome = "WIN"
            pnl_pct = abs(price_change_pct)  # Positive P&L for correct DOWN prediction
        elif price_change_pct > 0.01:  # Up more than 1%
            outcome = "LOSS"
            pnl_pct = -abs(price_change_pct)  # Negative P&L for wrong DOWN prediction
        else:
            outcome = "BREAK_EVEN"
            pnl_pct = 0.0
    
    else:
        # Unknown direction
        outcome = "BREAK_EVEN"
        pnl_pct = 0.0
    
    pnl = position_size * pnl_pct
    
    return {
        "outcome": outcome,
        "profit_loss_pct": pnl_pct,
        "profit_loss": pnl
    }


def reevaluate_all_trades(dry_run: bool = True):
    """
    Re-evaluate all resolved paper trades with corrected logic.
    
    Args:
        dry_run: If True, only preview changes without updating database
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get all resolved trades
    cur.execute("""
        SELECT * FROM paper_trades
        WHERE outcome != 'PENDING'
        ORDER BY created_at DESC
    """)
    
    trades = cur.fetchall()
    
    LOGGER.info(f"Found {len(trades)} resolved trades to re-evaluate")
    
    changes = {
        "total": 0,
        "changed": 0,
        "old_wins": 0,
        "new_wins": 0,
        "old_losses": 0,
        "new_losses": 0,
        "old_stopped": 0,
        "old_pnl": 0.0,
        "new_pnl": 0.0
    }
    
    for trade in trades:
        trade_id = trade["paper_trade_id"]
        old_outcome = trade["outcome"]
        old_pnl = float(trade.get("profit_loss") or 0.0)
        
        # Re-evaluate with corrected logic
        new_eval = evaluate_trade_corrected(trade)
        new_outcome = new_eval["outcome"]
        new_pnl = new_eval["profit_loss"]
        
        changes["total"] += 1
        
        # Track old outcomes
        if old_outcome == "WIN":
            changes["old_wins"] += 1
        elif old_outcome == "LOSS":
            changes["old_losses"] += 1
        elif old_outcome == "STOPPED":
            changes["old_stopped"] += 1
        
        changes["old_pnl"] += old_pnl
        
        # Track new outcomes
        if new_outcome == "WIN":
            changes["new_wins"] += 1
        elif new_outcome == "LOSS":
            changes["new_losses"] += 1
        
        changes["new_pnl"] += new_pnl
        
        # Check if changed
        if old_outcome != new_outcome or abs(old_pnl - new_pnl) > 0.01:
            changes["changed"] += 1
            
            if changes["changed"] <= 10:  # Show first 10 changes
                LOGGER.info(
                    f"Trade {trade_id} ({trade['symbol']} {trade['signal_direction']}): "
                    f"{old_outcome} (${old_pnl:.2f}) -> {new_outcome} (${new_pnl:.2f})"
                )
            
            if not dry_run:
                # Update database
                cur.execute("""
                    UPDATE paper_trades
                    SET outcome = %s,
                        profit_loss = %s,
                        profit_loss_pct = %s
                    WHERE paper_trade_id = %s
                """, (
                    new_outcome,
                    new_pnl,
                    new_eval["profit_loss_pct"],
                    trade_id
                ))
    
    if not dry_run:
        conn.commit()
        LOGGER.info("✅ Database updated with corrected evaluations")
    else:
        LOGGER.info("🔍 DRY RUN - No changes made to database")
    
    # Summary
    LOGGER.info("\n" + "="*60)
    LOGGER.info("RE-EVALUATION SUMMARY")
    LOGGER.info("="*60)
    LOGGER.info(f"Total trades re-evaluated: {changes['total']}")
    LOGGER.info(f"Changed: {changes['changed']} ({changes['changed']/changes['total']*100:.1f}%)")
    LOGGER.info("")
    LOGGER.info("OLD RESULTS (BROKEN LOGIC):")
    LOGGER.info(f"  Wins: {changes['old_wins']}")
    LOGGER.info(f"  Losses: {changes['old_losses']}")
    LOGGER.info(f"  Stopped: {changes['old_stopped']}")
    LOGGER.info(f"  Win Rate: {changes['old_wins']/(changes['old_wins']+changes['old_losses']+changes['old_stopped'])*100:.2f}%")
    LOGGER.info(f"  Total P&L: ${changes['old_pnl']:.2f}")
    LOGGER.info("")
    LOGGER.info("NEW RESULTS (FIXED LOGIC):")
    LOGGER.info(f"  Wins: {changes['new_wins']}")
    LOGGER.info(f"  Losses: {changes['new_losses']}")
    LOGGER.info(f"  Win Rate: {changes['new_wins']/(changes['new_wins']+changes['new_losses'])*100:.2f}%")
    LOGGER.info(f"  Total P&L: ${changes['new_pnl']:.2f}")
    LOGGER.info(f"  Improvement: ${changes['new_pnl'] - changes['old_pnl']:.2f}")
    LOGGER.info("="*60)
    
    conn.close()
    
    return changes


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate paper trades with corrected logic")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without updating database")
    args = parser.parse_args()
    
    LOGGER.info("="*60)
    LOGGER.info("PAPER TRADE RE-EVALUATION SCRIPT")
    LOGGER.info("="*60)
    LOGGER.info(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'LIVE (will update database)'}")
    LOGGER.info("")
    
    try:
        changes = reevaluate_all_trades(dry_run=args.dry_run)
        
        if args.dry_run:
            LOGGER.info("\n✅ Dry run complete. Run without --dry-run to apply changes.")
        else:
            LOGGER.info("\n✅ Re-evaluation complete! New stats should reflect corrected logic.")
    
    except Exception as e:
        LOGGER.error(f"❌ Re-evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
