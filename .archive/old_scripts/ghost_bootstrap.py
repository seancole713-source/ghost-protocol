#!/usr/bin/env python3
"""
Ghost Bootstrap System
Loads initial portfolio and watchlist on startup so Ghost never starts empty
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path

LOGGER = logging.getLogger("ghost.bootstrap")

INIT_DATA_FILE = Path(__file__).parent / "ghost_init_data.json"
# Handle /data vs data/ path - use data/ if /data isn't accessible
_raw_path = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
if _raw_path.startswith("/data") and not Path("/data").exists():
    WOLF_DB_PATH = "data" + _raw_path[5:]  # Convert /data/wolf.db -> data/wolf.db
else:
    WOLF_DB_PATH = _raw_path


def load_init_data():
    """Load initialization data from ghost_init_data.json"""
    if not INIT_DATA_FILE.exists():
        LOGGER.warning(f"Init data file not found: {INIT_DATA_FILE}")
        return None

    try:
        with open(INIT_DATA_FILE) as f:
            data = json.load(f)
        LOGGER.info(
            f"Loaded init data: {len(data.get('portfolio', {}).get('positions', []))} positions, {len(data.get('watchlist', []))} watchlist symbols"
        )
        return data
    except Exception as e:
        LOGGER.exception(f"Failed to load init data: {e}")
        return None


def ensure_tables_exist(conn):
    """Ensure portfolio tables exist (using PortfolioStore schema)"""
    try:
        cursor = conn.cursor()

        # Portfolio positions table (matches PortfolioStore schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio_positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL,
                entry_price REAL,
                entry_date INTEGER,
                last_known_price REAL,
                last_price_update INTEGER,
                last_provider TEXT,
                notes TEXT,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)

        conn.commit()
        LOGGER.info("Database tables verified")
    except Exception as e:
        LOGGER.exception(f"Failed to create tables: {e}")


def bootstrap_portfolio(conn, positions):
    """Load portfolio positions into database using PortfolioStore schema"""
    if not positions:
        LOGGER.info("No positions to bootstrap")
        return 0

    cursor = conn.cursor()
    loaded = 0

    for pos in positions:
        try:
            symbol = pos.get("symbol", "").upper()
            shares = pos.get("shares", 0)
            cost_basis = pos.get("cost_basis", 0.0)
            entry_date = pos.get("entry_date", datetime.now().strftime("%Y-%m-%d"))
            notes = pos.get("notes", "")

            # Convert entry_date to timestamp
            try:
                entry_ts = int(datetime.strptime(entry_date, "%Y-%m-%d").timestamp())
            except Exception:
                entry_ts = int(datetime.now().timestamp())

            # Check if position already exists
            cursor.execute("SELECT symbol FROM portfolio_positions WHERE symbol = ?", (symbol,))
            exists = cursor.fetchone()

            if exists:
                # Update existing position
                cursor.execute(
                    """
                    UPDATE portfolio_positions
                    SET quantity = ?, avg_cost = ?, entry_price = ?, entry_date = ?,
                        notes = ?, updated_at = ?
                    WHERE symbol = ?
                """,
                    (
                        shares,
                        cost_basis,
                        cost_basis,
                        entry_ts,
                        notes,
                        int(datetime.now().timestamp()),
                        symbol,
                    ),
                )
                LOGGER.info(
                    f"Updated portfolio position: {symbol} ({shares} shares @ ${cost_basis})"
                )
            else:
                # Insert new position
                cursor.execute(
                    """
                    INSERT INTO portfolio_positions
                    (symbol, quantity, avg_cost, entry_price, entry_date, notes, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        shares,
                        cost_basis,
                        cost_basis,
                        entry_ts,
                        notes,
                        int(datetime.now().timestamp()),
                    ),
                )
                LOGGER.info(
                    f"Loaded portfolio position: {symbol} ({shares} shares @ ${cost_basis})"
                )

            loaded += 1
        except Exception as e:
            LOGGER.exception(f"Failed to load position {pos}: {e}")

    conn.commit()
    return loaded


def bootstrap_watchlist(watchlist_db_path, symbols):
    """Load watchlist symbols into watchlist.db"""
    if not symbols:
        LOGGER.info("No watchlist symbols to bootstrap")
        return 0

    try:
        # Connect to watchlist.db (separate from wolf.db)
        watchlist_conn = sqlite3.connect(watchlist_db_path)
        cursor = watchlist_conn.cursor()

        # Ensure watchlist table exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                added_at TEXT,
                last_updated TEXT,
                metadata TEXT
            )
        """)
        watchlist_conn.commit()

        loaded = 0
        now = datetime.now().isoformat()

        for symbol in symbols:
            try:
                symbol = symbol.upper()

                # Check if symbol already exists
                cursor.execute("SELECT symbol FROM watchlist WHERE symbol = ?", (symbol,))
                exists = cursor.fetchone()

                if not exists:
                    cursor.execute(
                        """
                        INSERT INTO watchlist (symbol, name, added_at, last_updated, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (symbol, "", now, now, "Auto-loaded from init data"),
                    )
                    LOGGER.info(f"Added watchlist symbol: {symbol}")
                    loaded += 1
                else:
                    LOGGER.debug(f"Watchlist symbol already exists: {symbol}")
            except Exception as e:
                LOGGER.exception(f"Failed to load watchlist symbol {symbol}: {e}")

        watchlist_conn.commit()
        watchlist_conn.close()
        return loaded
    except Exception as e:
        LOGGER.exception(f"Failed to connect to watchlist database: {e}")
        return 0


def run_bootstrap():
    """Main bootstrap entry point - loads all initialization data"""
    try:
        # Load init data
        init_data = load_init_data()
        if not init_data:
            LOGGER.warning("No init data to bootstrap")
            return False

        # Check if auto-load is enabled
        if not init_data.get("config", {}).get("auto_load_on_startup", True):
            LOGGER.info("Auto-load disabled in config, skipping bootstrap")
            return False

        # Connect to portfolio database (wolf.db)
        db_path = Path(WOLF_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))

        # Ensure tables exist
        ensure_tables_exist(conn)

        # Bootstrap portfolio
        positions = init_data.get("portfolio", {}).get("positions", [])
        portfolio_count = bootstrap_portfolio(conn, positions)

        conn.close()

        # Bootstrap watchlist (separate database)
        watchlist = init_data.get("watchlist", [])
        watchlist_db_path = str(Path(db_path).parent / "watchlist.db")
        watchlist_count = bootstrap_watchlist(watchlist_db_path, watchlist)

        LOGGER.info(
            f"Bootstrap complete: {portfolio_count} positions, {watchlist_count} watchlist symbols"
        )
        return True

    except Exception as e:
        LOGGER.exception(f"Bootstrap failed: {e}")
        return False


def get_bootstrap_status():
    """Check if bootstrap data is loaded in database"""
    try:
        db_path = Path(WOLF_DB_PATH)
        if not db_path.exists():
            return {"loaded": False, "reason": "Database not found"}

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check portfolio
        try:
            cursor.execute("SELECT COUNT(*) FROM portfolio_positions")
            portfolio_count = cursor.fetchone()[0]
        except Exception:
            portfolio_count = 0

        conn.close()

        # Check watchlist
        watchlist_count = 0
        try:
            watchlist_db_path = Path(db_path).parent / "watchlist.db"
            if watchlist_db_path.exists():
                watchlist_conn = sqlite3.connect(str(watchlist_db_path))
                wl_cursor = watchlist_conn.cursor()
                wl_cursor.execute("SELECT COUNT(*) FROM watchlist")
                watchlist_count = wl_cursor.fetchone()[0]
                watchlist_conn.close()
        except Exception:
            pass

        return {
            "loaded": portfolio_count > 0 or watchlist_count > 0,
            "portfolio_count": portfolio_count,
            "watchlist_count": watchlist_count,
        }
    except Exception as e:
        return {"loaded": False, "error": str(e)}


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("🚀 Ghost Bootstrap - Initializing system data...")
    success = run_bootstrap()

    if success:
        print("✅ Bootstrap complete!")
        status = get_bootstrap_status()
        print(f"   Portfolio: {status.get('portfolio_count', 0)} positions")
        print(f"   Watchlist: {status.get('watchlist_count', 0)} symbols")
    else:
        print("❌ Bootstrap failed - check logs")
        exit(1)
