#!/usr/bin/env python3
"""
Ghost Initialization System
Loads portfolio holdings and watchlist from ghost_init_data.json into the database on startup.
Ensures Ghost UI is never empty - your data persists across restarts.
"""

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Configuration
INIT_DATA_FILE = Path(__file__).parent / "ghost_init_data.json"
DB_PATH = Path(__file__).parent / "data" / "wolf.db"


def load_init_data():
    """Load initial portfolio and watchlist configuration"""
    if not INIT_DATA_FILE.exists():
        print(f"⚠️  Init data file not found: {INIT_DATA_FILE}")
        return None

    try:
        with open(INIT_DATA_FILE) as f:
            data = json.load(f)
            print(
                f"✅ Loaded init data: {len(data.get('portfolio', {}).get('positions', []))} positions, {len(data.get('watchlist', []))} watchlist symbols"
            )
            return data
    except Exception as e:
        print(f"❌ Failed to load init data: {e}")
        return None


def ensure_database_tables(conn):
    """Create necessary tables if they don't exist"""
    cursor = conn.cursor()

    # Portfolio positions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            shares REAL NOT NULL,
            cost_basis REAL NOT NULL,
            entry_date TEXT,
            notes TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Watchlist table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT UNIQUE NOT NULL,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    """)

    # Portfolio metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    print("✅ Database tables verified")


def initialize_portfolio(conn, portfolio_data):
    """Load portfolio positions into database"""
    cursor = conn.cursor()

    positions = portfolio_data.get("positions", [])
    if not positions:
        print("ℹ️  No positions to load")
        return

    # Clear existing positions (fresh start on each initialization)
    cursor.execute("DELETE FROM portfolio_positions")

    # Insert each position
    for pos in positions:
        cursor.execute(
            """
            INSERT INTO portfolio_positions (symbol, shares, cost_basis, entry_date, notes)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                pos["symbol"],
                pos["shares"],
                pos["cost_basis"],
                pos.get("entry_date", datetime.now().isoformat()),
                pos.get("notes", ""),
            ),
        )
        print(f"  📈 Loaded {pos['shares']} shares of {pos['symbol']} @ ${pos['cost_basis']}")

    # Store cash balances
    cash = portfolio_data.get("cash", {})
    cursor.execute(
        "INSERT OR REPLACE INTO portfolio_metadata (key, value) VALUES ('stock_cash', ?)",
        (str(cash.get("stock", 0.0)),),
    )
    cursor.execute(
        "INSERT OR REPLACE INTO portfolio_metadata (key, value) VALUES ('crypto_cash', ?)",
        (str(cash.get("crypto", 0.0)),),
    )

    conn.commit()
    print(f"✅ Loaded {len(positions)} position(s) into portfolio")


def initialize_watchlist(conn, watchlist_symbols):
    """Load watchlist symbols into database"""
    cursor = conn.cursor()

    if not watchlist_symbols:
        print("ℹ️  No watchlist symbols to load")
        return

    # Clear existing watchlist
    cursor.execute("DELETE FROM watchlist")

    # Insert each symbol
    loaded = 0
    for symbol in watchlist_symbols:
        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO watchlist (symbol)
                VALUES (?)
            """,
                (symbol.upper(),),
            )
            if cursor.rowcount > 0:
                loaded += 1
        except Exception as e:
            print(f"  ⚠️  Failed to add {symbol}: {e}")

    conn.commit()
    print(f"✅ Loaded {loaded} symbol(s) into watchlist")


def run_initialization():
    """Main initialization routine"""
    print("\n🚀 Ghost Initialization System")
    print("=" * 50)

    # Load configuration
    init_data = load_init_data()
    if not init_data:
        print("❌ Cannot proceed without init data")
        return False

    # Check if auto-load is enabled
    if not init_data.get("config", {}).get("auto_load_on_startup", True):
        print("ℹ️  Auto-load disabled in config")
        return False

    # Connect to database
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        print(f"✅ Connected to database: {DB_PATH}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

    try:
        # Create tables if needed
        ensure_database_tables(conn)

        # Load portfolio
        if "portfolio" in init_data:
            initialize_portfolio(conn, init_data["portfolio"])

        # Load watchlist
        if "watchlist" in init_data:
            initialize_watchlist(conn, init_data["watchlist"])

        # Mark initialization complete
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO portfolio_metadata (key, value, updated_at)
            VALUES ('last_initialized', ?, CURRENT_TIMESTAMP)
        """,
            (datetime.now().isoformat(),),
        )
        conn.commit()

        print("\n✅ Ghost initialization complete!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    success = run_initialization()
    sys.exit(0 if success else 1)
