#!/usr/bin/env python3
"""
Direct PostgreSQL Whitelist Update Script
==========================================
Bypasses the API and updates V2 quality filters directly in PostgreSQL.

Usage:
    # Set DATABASE_URL or use Railway's connection string
    export DATABASE_URL="postgresql://user:pass@host:port/db"
    
    # Or pass as argument
    python update_whitelist_direct.py --db-url "postgresql://..."
    
    # Apply crypto-only preset
    python update_whitelist_direct.py --crypto-only
    
    # Custom whitelist
    python update_whitelist_direct.py --whitelist RNDR,CHZ,TURBO,ZEC
"""

import os
import sys
import json
import argparse
from datetime import datetime

# Crypto-only preset based on Jan 25 loser analysis
CRYPTO_ONLY_WHITELIST = [
    "RNDR",   # 47.6% (89/187) - BEST
    "CHZ",    # 37.1% (75/202)
    "TURBO",  # 35.5% (27/76)
    "ZEC",    # 31.1% (46/148)
    "EGLD",   # ~27%
    "ILV",    # ~30%
    "RLC",    # ~28%
    "OCEAN",  # ~26%
]

CRYPTO_ONLY_PINNED = ["RNDR", "CHZ", "TURBO", "ZEC"]

CRYPTO_ONLY_BLACKLIST = [
    # Losing stocks (4.5% WR = broken)
    "ABCL", "GME", "BMBL", "ITRI", "TGTX", "XPO", "SOUN",
    "ARCT", "CVNA", "IQ", "T",
    # Bad crypto
    "XRP", "DOT", "AVAX", "UNI", "PEPE", "SNX", "1INCH",
    "LDO", "ETC", "ALGO", "BTC", "ETH", "SOL", "ADA",
    "BNB", "LTC", "ICP", "LRC"
]


def get_db_connection(db_url: str = None):
    """Get PostgreSQL connection."""
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)
    
    url = db_url or os.getenv("DATABASE_URL")
    if not url:
        print("❌ DATABASE_URL not set. Pass --db-url or set environment variable.")
        print("\nFind your Railway PostgreSQL URL in:")
        print("  Railway Dashboard → ghost-protocol → Variables → DATABASE_URL")
        sys.exit(1)
    
    try:
        conn = psycopg2.connect(url)
        print(f"✅ Connected to PostgreSQL")
        return conn
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)


def get_current_config(conn) -> dict:
    """Fetch current V2 quality config from PostgreSQL."""
    cur = conn.cursor()
    
    # v2_quality_config is the actual table used by the system
    try:
        cur.execute("""
            SELECT value FROM v2_quality_config 
            WHERE key = 'config'
        """)
        row = cur.fetchone()
        if row:
            return json.loads(row[0])
    except Exception as e:
        print(f"⚠️  v2_quality_config table query failed: {e}")
    
    return None


def update_config(conn, whitelist: list, blacklist: list, pinned: list = None, note: str = ""):
    """Update V2 quality config in PostgreSQL."""
    cur = conn.cursor()
    
    config = {
        "whitelist": sorted(whitelist),
        "blacklist": sorted(blacklist),
        "pinned_whitelist": sorted(pinned or whitelist[:4]),
        "quarantine": [],
        "metrics": {},
        "last_updated": datetime.utcnow().isoformat(),
        "config": {
            "min_predictions": 20,
            "whitelist_wr": 0.55,
            "blacklist_wr": 0.45,
            "note": note or f"Direct update via script - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        }
    }
    
    config_json = json.dumps(config)
    
    # Ensure table exists
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS v2_quality_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"⚠️  Table creation check: {e}")
        conn.rollback()
    
    # Update config
    try:
        cur.execute("""
            INSERT INTO v2_quality_config (key, value, updated_at)
            VALUES ('config', %s, NOW())
            ON CONFLICT (key) DO UPDATE SET value = %s, updated_at = NOW()
        """, (config_json, config_json))
        conn.commit()
        print("✅ Updated v2_quality_config table")
        return True
    except Exception as e:
        print(f"❌ v2_quality_config update failed: {e}")
        conn.rollback()
        return False


def list_tables(conn):
    """List all tables in the database."""
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [row[0] for row in cur.fetchall()]
    return tables


def main():
    parser = argparse.ArgumentParser(description="Direct PostgreSQL V2 Whitelist Update")
    parser.add_argument("--db-url", help="PostgreSQL connection URL")
    parser.add_argument("--crypto-only", action="store_true", help="Apply crypto-only preset")
    parser.add_argument("--whitelist", help="Comma-separated whitelist symbols")
    parser.add_argument("--blacklist", help="Comma-separated blacklist symbols")
    parser.add_argument("--show-current", action="store_true", help="Show current config")
    parser.add_argument("--list-tables", action="store_true", help="List database tables")
    parser.add_argument("--note", help="Note to add to config")
    
    args = parser.parse_args()
    
    # Connect to database
    conn = get_db_connection(args.db_url)
    
    # List tables if requested
    if args.list_tables:
        tables = list_tables(conn)
        print(f"\n📋 Database tables ({len(tables)}):")
        for t in tables:
            print(f"   • {t}")
        conn.close()
        return
    
    # Show current config if requested
    if args.show_current:
        config = get_current_config(conn)
        if config:
            print("\n📊 Current V2 Quality Config:")
            print(f"   Whitelist ({len(config.get('whitelist', []))}): {config.get('whitelist', [])}")
            print(f"   Blacklist ({len(config.get('blacklist', []))}): {config.get('blacklist', [])}")
            print(f"   Pinned: {config.get('pinned_whitelist', [])}")
            print(f"   Last Updated: {config.get('last_updated', 'unknown')}")
            print(f"   Note: {config.get('config', {}).get('note', 'none')}")
        else:
            print("❌ No config found in database")
        conn.close()
        return
    
    # Determine what to update
    if args.crypto_only:
        whitelist = CRYPTO_ONLY_WHITELIST
        blacklist = CRYPTO_ONLY_BLACKLIST
        pinned = CRYPTO_ONLY_PINNED
        note = "CRYPTO-ONLY: Jan 25 loser analysis - Stocks=4.5% WR (broken), Crypto=38.7% WR"
        print("\n🎯 Applying CRYPTO-ONLY preset:")
        print(f"   Whitelist: {whitelist}")
        print(f"   Pinned: {pinned}")
        print(f"   Blacklist: {len(blacklist)} symbols")
    elif args.whitelist:
        whitelist = [s.strip().upper() for s in args.whitelist.split(",")]
        blacklist = [s.strip().upper() for s in args.blacklist.split(",")] if args.blacklist else []
        pinned = whitelist[:4]
        note = args.note or "Custom whitelist via script"
        print(f"\n📝 Applying custom whitelist: {whitelist}")
    else:
        print("\n❌ No action specified. Use --crypto-only or --whitelist")
        print("\nExamples:")
        print("  python update_whitelist_direct.py --crypto-only")
        print("  python update_whitelist_direct.py --whitelist RNDR,CHZ,TURBO,ZEC")
        print("  python update_whitelist_direct.py --show-current")
        conn.close()
        return
    
    # Update the config
    print("\n🔄 Updating PostgreSQL...")
    success = update_config(conn, whitelist, blacklist, pinned, note)
    
    if success:
        print("\n✅ SUCCESS! V2 whitelist updated in PostgreSQL.")
        print("\n⚠️  IMPORTANT: You need to trigger a config reload on the running server:")
        print("   curl -X POST https://ghost-protocol-production.up.railway.app/api/v2/quality/reload")
        print("\n   Or restart the Railway service to pick up the new config.")
    else:
        print("\n❌ FAILED to update config. Check database permissions.")
    
    conn.close()


if __name__ == "__main__":
    main()
