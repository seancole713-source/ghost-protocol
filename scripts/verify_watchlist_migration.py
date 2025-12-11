#!/usr/bin/env python3
"""
Personal Watchlist Migration Verification Script
================================================

Verifies that ghost_watchlist_items table exists and is ready for use.
Can also seed initial data for testing.

Usage:
    python scripts/verify_watchlist_migration.py
    python scripts/verify_watchlist_migration.py --seed
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

LOGGER = logging.getLogger(__name__)


def verify_table_exists() -> bool:
    """Check if ghost_watchlist_items table exists."""
    try:
        from core.db_engine import get_db_connection, IS_POSTGRES
        
        if not IS_POSTGRES:
            LOGGER.info("✅ Running in SQLite mode - no migration needed")
            return True
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check table existence
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    AND table_name = 'ghost_watchlist_items'
                ) as exists
            """)
            
            result = cursor.fetchone()
            table_exists = result['exists'] if result else False
            
            if table_exists:
                LOGGER.info("✅ ghost_watchlist_items table EXISTS")
                
                # Get row count
                cursor.execute("SELECT COUNT(*) as count FROM ghost_watchlist_items WHERE removed_at IS NULL")
                count_result = cursor.fetchone()
                active_count = count_result['count'] if count_result else 0
                
                LOGGER.info(f"   Active items: {active_count}")
                
                # Show sample data
                cursor.execute("""
                    SELECT symbol, asset_type, owns_position, priority, added_at
                    FROM ghost_watchlist_items
                    WHERE removed_at IS NULL
                    ORDER BY priority DESC, added_at DESC
                    LIMIT 5
                """)
                
                samples = cursor.fetchall()
                if samples:
                    LOGGER.info("   Sample items:")
                    for row in samples:
                        LOGGER.info(f"      {row['symbol']} ({row['asset_type']}) - Priority: {row['priority']}, Owns: {row['owns_position']}")
                
                return True
            else:
                LOGGER.error("❌ ghost_watchlist_items table DOES NOT EXIST")
                LOGGER.info("   Run: python -m core.migration_runner")
                return False
                
    except Exception as e:
        LOGGER.error(f"❌ Verification failed: {e}", exc_info=True)
        return False


def seed_test_data() -> bool:
    """Seed initial test data."""
    try:
        from core.db_engine import get_db_connection, IS_POSTGRES
        
        if not IS_POSTGRES:
            LOGGER.info("SQLite mode - skipping seed")
            return True
        
        # Test symbols (mix of stocks and crypto)
        test_items = [
            {"symbol": "WOLF", "asset_type": "stock", "owns_position": True, "priority": 3, "notes": "Primary focus"},
            {"symbol": "NVDA", "asset_type": "stock", "owns_position": True, "priority": 3, "notes": "AI leader"},
            {"symbol": "TSLA", "asset_type": "stock", "owns_position": False, "priority": 2, "notes": "High volatility"},
            {"symbol": "BTC", "asset_type": "crypto", "owns_position": True, "priority": 3, "notes": "Crypto king"},
            {"symbol": "ETH", "asset_type": "crypto", "owns_position": True, "priority": 2, "notes": "Smart contracts"},
            {"symbol": "XRP", "asset_type": "crypto", "owns_position": False, "priority": 2, "notes": "Banking integration"},
            {"symbol": "PLTR", "asset_type": "stock", "owns_position": True, "priority": 2, "notes": "Data analytics"},
            {"symbol": "AMD", "asset_type": "stock", "owns_position": False, "priority": 1, "notes": "GPU competitor"},
        ]
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            inserted = 0
            for item in test_items:
                try:
                    cursor.execute("""
                        INSERT INTO ghost_watchlist_items 
                            (symbol, asset_type, owns_position, priority, notes, alert_threshold_pct)
                        VALUES (%(symbol)s, %(asset_type)s, %(owns_position)s, %(priority)s, %(notes)s, 5.0)
                        ON CONFLICT (symbol, asset_type) DO NOTHING
                    """, item)
                    
                    if cursor.rowcount > 0:
                        inserted += 1
                        LOGGER.info(f"   ✅ Added: {item['symbol']} ({item['asset_type']})")
                    
                except Exception as e:
                    LOGGER.warning(f"   ⚠️  Skipped {item['symbol']}: {e}")
            
            conn.commit()
            
            LOGGER.info(f"✅ Seeded {inserted}/{len(test_items)} test items")
            return True
            
    except Exception as e:
        LOGGER.error(f"❌ Seed failed: {e}", exc_info=True)
        return False


def test_api_endpoint() -> bool:
    """Test the /api/v3/watchlist/user endpoint."""
    try:
        import os
        import requests
        
        # Get base URL from environment or use localhost
        base_url = os.getenv("API_BASE_URL", "http://localhost:8080")
        endpoint = f"{base_url}/api/v3/watchlist/user"
        
        LOGGER.info(f"Testing API endpoint: {endpoint}")
        
        response = requests.get(endpoint, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            LOGGER.info(f"✅ API endpoint working: {len(items)} items returned")
            
            if items:
                LOGGER.info("   Sample items:")
                for item in items[:3]:
                    LOGGER.info(f"      {item['symbol']} ({item['asset_type']}) - Owns: {item['owns_position']}")
            
            return True
        else:
            LOGGER.error(f"❌ API returned {response.status_code}: {response.text[:200]}")
            return False
            
    except Exception as e:
        LOGGER.error(f"❌ API test failed: {e}", exc_info=True)
        return False


def main():
    """Main verification flow."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify personal watchlist migration")
    parser.add_argument("--seed", action="store_true", help="Seed test data")
    parser.add_argument("--test-api", action="store_true", help="Test API endpoint")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PERSONAL WATCHLIST MIGRATION VERIFICATION")
    print("="*60 + "\n")
    
    # Step 1: Verify table exists
    LOGGER.info("Step 1: Verifying table existence...")
    table_ok = verify_table_exists()
    
    if not table_ok:
        LOGGER.error("\n❌ VERIFICATION FAILED: Table does not exist")
        LOGGER.info("Run migrations first: python -m core.migration_runner")
        return False
    
    # Step 2: Seed test data (optional)
    if args.seed:
        LOGGER.info("\nStep 2: Seeding test data...")
        seed_ok = seed_test_data()
        if not seed_ok:
            LOGGER.warning("⚠️  Seeding had issues but continuing...")
    
    # Step 3: Test API endpoint (optional)
    if args.test_api:
        LOGGER.info("\nStep 3: Testing API endpoint...")
        api_ok = test_api_endpoint()
        if not api_ok:
            LOGGER.warning("⚠️  API test failed but table verification passed")
    
    print("\n" + "="*60)
    print("✅ VERIFICATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("  1. Deploy to Railway: git push")
    print("  2. Check logs: railway logs --tail 100")
    print("  3. Test endpoint: curl https://your-app.railway.app/api/v3/watchlist/user")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
