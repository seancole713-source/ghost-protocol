#!/usr/bin/env python3
"""
WOLF Portfolio Normalization Script

Applies 120:1 reverse split adjustment to WOLF holdings.
Delisted WOLF (2025-10-01) requires normalization to reflect true P&L.

Method: Adjust avg_cost by 120x to match post-split prices.
"""

import os
import sqlite3
import sys
from datetime import datetime

# Database path
DB_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")


def normalize_wolf_portfolio():
    """Apply 120:1 reverse split adjustment to WOLF position."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # Get current WOLF position
        cur.execute("""
            SELECT symbol, quantity, avg_cost, last_known_price
            FROM portfolio_positions
            WHERE symbol = 'WOLF'
        """)
        position = cur.fetchone()

        if not position:
            print("✅ No WOLF position found - nothing to normalize")
            conn.close()
            return

        symbol, quantity, avg_cost, last_price = position

        print("\n" + "="*60)
        print("🔧 WOLF PORTFOLIO NORMALIZATION")
        print("="*60 + "\n")

        print("📊 BEFORE (Pre-Split Prices):")
        print(f"   Symbol:     {symbol}")
        print(f"   Quantity:   {quantity:.8f}")
        print(f"   Avg Cost:   ${avg_cost:.2f}")
        print(f"   Last Price: ${last_price:.2f}")
        print(f"   P&L:        {((last_price - avg_cost) / avg_cost * 100):.2f}%")

        # Apply 120:1 reverse split adjustment to avg_cost
        adjusted_avg_cost = avg_cost * 120

        print("\n⚙️  APPLYING 120:1 REVERSE SPLIT ADJUSTMENT:")
        print(f"   Old Avg Cost: ${avg_cost:.2f}")
        print(f"   New Avg Cost: ${adjusted_avg_cost:.2f} (multiplied by 120)")

        # Update database using symbol as key
        cur.execute("""
            UPDATE portfolio_positions
            SET avg_cost = ?,
                entry_price = ?,
                updated_at = ?
            WHERE symbol = ?
        """, (adjusted_avg_cost, adjusted_avg_cost, int(datetime.now().timestamp()), symbol))

        conn.commit()

        # Calculate new P&L
        new_pnl_pct = ((last_price - adjusted_avg_cost) / adjusted_avg_cost * 100)

        print("\n✅ AFTER (Post-Split Normalized):")
        print(f"   Symbol:     {symbol}")
        print(f"   Quantity:   {quantity:.8f}")
        print(f"   Avg Cost:   ${adjusted_avg_cost:.2f}")
        print(f"   Last Price: ${last_price:.2f}")
        print(f"   P&L:        {new_pnl_pct:.2f}%")

        print("\n" + "="*60)
        print("✅ WOLF portfolio normalized successfully")
        print("="*60 + "\n")

        conn.close()
        return True

    except Exception as e:
        print(f"❌ Error normalizing WOLF portfolio: {e}")
        return False


if __name__ == "__main__":
    success = normalize_wolf_portfolio()
    sys.exit(0 if success else 1)
