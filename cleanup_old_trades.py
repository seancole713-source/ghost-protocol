#!/usr/bin/env python3
"""
Clean up pending trades older than V2 filter implementation.
V2 filter was deployed around Jan 12-14, 2026.
"""

import os
import sys
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, '/workspaces/ghost-protocol')

from core.prediction_store import PostgresBackend

def cleanup_old_pending_trades():
    """Mark old pending trades as EXPIRED to clean up stats."""
    
    backend = PostgresBackend()
    
    # V2 filter was deployed around Jan 14, 2026
    # Any pending trades before Jan 14 should be expired
    cutoff_date = datetime(2026, 1, 14, 0, 0, 0)
    
    print(f"🔍 Finding pending trades created before {cutoff_date.strftime('%Y-%m-%d')}...")
    
    # Query for old pending trades
    query = """
    SELECT 
        COUNT(*) as count,
        MIN(created_at) as oldest,
        MAX(created_at) as newest
    FROM paper_trades
    WHERE outcome = 'PENDING'
      AND created_at < %s
    """
    
    with backend.conn.cursor() as cur:
        cur.execute(query, (cutoff_date,))
        result = cur.fetchone()
        
        if result and result[0] > 0:
            print(f"📊 Found {result[0]:,} old pending trades")
            print(f"   Oldest: {result[1]}")
            print(f"   Newest: {result[2]}")
            print()
            
            # Ask for confirmation
            response = input(f"❓ Mark {result[0]:,} old trades as EXPIRED? (yes/no): ")
            
            if response.lower() in ['yes', 'y']:
                # Update old pending trades to EXPIRED
                update_query = """
                UPDATE paper_trades
                SET 
                    outcome = 'EXPIRED',
                    checked_at = NOW(),
                    notes = 'Auto-expired: Pre-V2 filter trade'
                WHERE outcome = 'PENDING'
                  AND created_at < %s
                """
                
                cur.execute(update_query, (cutoff_date,))
                backend.conn.commit()
                
                print(f"✅ Marked {cur.rowcount:,} trades as EXPIRED")
                print()
                
                # Show updated stats
                stats_query = """
                SELECT 
                    outcome,
                    COUNT(*) as count
                FROM paper_trades
                GROUP BY outcome
                ORDER BY count DESC
                """
                
                cur.execute(stats_query)
                print("📊 Updated trade counts:")
                for row in cur.fetchall():
                    print(f"   {row[0]}: {row[1]:,}")
                    
            else:
                print("❌ Cancelled - no changes made")
        else:
            print("✅ No old pending trades found")
    
    backend.close()

if __name__ == "__main__":
    cleanup_old_pending_trades()
