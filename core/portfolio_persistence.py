"""
Portfolio Persistence Layer
============================

Ensures Ghost always remembers your portfolio state, even when:
- Markets are closed
- Data providers are rate-limited/unavailable
- Server restarts

Features:
- Persistent price cache with timestamps
- Daily portfolio snapshots
- Automatic fallback to cached values
- Live data refresh when available
"""

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

# Default paths - handle /data permission issues
_raw_db_path = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
if _raw_db_path.startswith("/data") and not Path("/data").exists():
    # Convert /data/wolf.db -> data/wolf.db when /data isn't accessible
    DEFAULT_DB_PATH = "data" + _raw_db_path[5:]
else:
    DEFAULT_DB_PATH = _raw_db_path
DEFAULT_SNAPSHOT_DIR = "data/snapshots"


class PortfolioStore:
    """Persistent storage for portfolio state and price history."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """Create tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)

        # Portfolio positions table
        conn.execute("""
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

        # Price history table (for fallback)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                prev_close REAL,
                provider TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                market_status TEXT,
                PRIMARY KEY (symbol, timestamp)
            )
        """)

        # Daily snapshots table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                snapshot_date TEXT PRIMARY KEY,
                timestamp INTEGER NOT NULL,
                portfolio_value REAL NOT NULL,
                cash_balance REAL NOT NULL,
                positions TEXT NOT NULL,
                prices TEXT NOT NULL,
                notes TEXT,
                created_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Cash balances table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cash_balances (
                account_type TEXT PRIMARY KEY DEFAULT 'main',
                balance REAL NOT NULL,
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)

        conn.commit()
        conn.close()

    def save_position(
        self,
        symbol: str,
        quantity: float,
        avg_cost: float,
        last_price: float | None = None,
        provider: str | None = None,
    ) -> bool:
        """Save or update a portfolio position."""
        try:
            conn = sqlite3.connect(self.db_path)
            now = int(time.time())

            # Check if position exists
            cur = conn.cursor()
            cur.execute(
                "SELECT quantity, entry_date FROM portfolio_positions WHERE symbol = ?", (symbol,)
            )
            existing = cur.fetchone()

            if existing:
                # Update existing position
                conn.execute(
                    """
                    UPDATE portfolio_positions
                    SET quantity = ?, avg_cost = ?, last_known_price = ?,
                        last_price_update = ?, last_provider = ?, updated_at = ?
                    WHERE symbol = ?
                """,
                    (quantity, avg_cost, last_price, now, provider, now, symbol),
                )
            else:
                # Insert new position
                conn.execute(
                    """
                    INSERT INTO portfolio_positions
                    (symbol, quantity, avg_cost, entry_price, entry_date,
                     last_known_price, last_price_update, last_provider, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (symbol, quantity, avg_cost, avg_cost, now, last_price, now, provider, now),
                )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[PORTFOLIO] Failed to save position {symbol}: {e}")
            return False

    def get_position(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve a portfolio position with last known price."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM portfolio_positions WHERE symbol = ?", (symbol,))
            row = cur.fetchone()
            conn.close()

            if row:
                return dict(row)
            return None
        except Exception as e:
            print(f"[PORTFOLIO] Failed to get position {symbol}: {e}")
            return None

    def get_all_positions(self) -> list[dict[str, Any]]:
        """Get all portfolio positions."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM portfolio_positions WHERE quantity > 0 ORDER BY symbol")
            rows = cur.fetchall()
            conn.close()
            return [dict(row) for row in rows]
        except Exception as e:
            print(f"[PORTFOLIO] Failed to get all positions: {e}")
            return []

    def save_price(
        self,
        symbol: str,
        price: float,
        prev_close: float | None = None,
        provider: str = "unknown",
        market_status: str = "unknown",
    ) -> bool:
        """Save price history for fallback."""
        try:
            conn = sqlite3.connect(self.db_path)
            now = int(time.time())

            conn.execute(
                """
                INSERT OR REPLACE INTO price_history
                (symbol, price, prev_close, provider, timestamp, market_status)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (symbol, price, prev_close, provider, now, market_status),
            )

            # Also update position's last_known_price
            conn.execute(
                """
                UPDATE portfolio_positions
                SET last_known_price = ?, last_price_update = ?, last_provider = ?
                WHERE symbol = ?
            """,
                (price, now, provider, symbol),
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[PORTFOLIO] Failed to save price {symbol}: {e}")
            return False

    def get_last_price(
        self, symbol: str, max_age_seconds: int = 86400
    ) -> tuple[float, float | None, str, int] | None:
        """
        Get last known price for symbol.

        Returns: (price, prev_close, provider, timestamp) or None
        max_age_seconds: Only return prices newer than this (default: 24h)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            now = int(time.time())
            min_timestamp = now - max_age_seconds

            cur.execute(
                """
                SELECT price, prev_close, provider, timestamp
                FROM price_history
                WHERE symbol = ? AND timestamp >= ?
                ORDER BY timestamp DESC LIMIT 1
            """,
                (symbol, min_timestamp),
            )

            row = cur.fetchone()
            conn.close()

            if row:
                return (float(row[0]), float(row[1]) if row[1] else None, row[2], int(row[3]))
            return None
        except Exception as e:
            print(f"[PORTFOLIO] Failed to get last price {symbol}: {e}")
            return None

    def save_daily_snapshot(
        self,
        date: str,
        portfolio_value: float,
        cash: float,
        positions: list[dict],
        prices: dict[str, float],
        notes: str = "",
    ) -> bool:
        """Save daily portfolio snapshot."""
        try:
            conn = sqlite3.connect(self.db_path)
            now = int(time.time())

            conn.execute(
                """
                INSERT OR REPLACE INTO daily_snapshots
                (snapshot_date, timestamp, portfolio_value, cash_balance, positions, prices, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    date,
                    now,
                    portfolio_value,
                    cash,
                    json.dumps(positions),
                    json.dumps(prices),
                    notes,
                ),
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[PORTFOLIO] Failed to save daily snapshot: {e}")
            return False

    def get_daily_snapshot(self, date: str) -> dict[str, Any] | None:
        """Retrieve daily snapshot by date."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM daily_snapshots WHERE snapshot_date = ?", (date,))
            row = cur.fetchone()
            conn.close()

            if row:
                data = dict(row)
                data["positions"] = json.loads(data["positions"])
                data["prices"] = json.loads(data["prices"])
                return data
            return None
        except Exception as e:
            print(f"[PORTFOLIO] Failed to get daily snapshot: {e}")
            return None

    def get_latest_snapshot(self) -> dict[str, Any] | None:
        """Get most recent daily snapshot."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM daily_snapshots ORDER BY snapshot_date DESC LIMIT 1")
            row = cur.fetchone()
            conn.close()

            if row:
                data = dict(row)
                data["positions"] = json.loads(data["positions"])
                data["prices"] = json.loads(data["prices"])
                return data
            return None
        except Exception as e:
            print(f"[PORTFOLIO] Failed to get latest snapshot: {e}")
            return None

    def save_cash_balance(self, balance: float, account_type: str = "main") -> bool:
        """Save cash balance."""
        try:
            conn = sqlite3.connect(self.db_path)
            now = int(time.time())

            conn.execute(
                """
                INSERT OR REPLACE INTO cash_balances (account_type, balance, updated_at)
                VALUES (?, ?, ?)
            """,
                (account_type, balance, now),
            )

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[PORTFOLIO] Failed to save cash balance: {e}")
            return False

    def get_cash_balance(self, account_type: str = "main") -> float:
        """Get cash balance."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT balance FROM cash_balances WHERE account_type = ?", (account_type,))
            row = cur.fetchone()
            conn.close()

            if row:
                return float(row[0])
            return 0.0
        except Exception as e:
            print(f"[PORTFOLIO] Failed to get cash balance: {e}")
            return 0.0

    def cleanup_old_prices(self, days_to_keep: int = 30):
        """Remove price history older than N days."""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff = int(time.time()) - (days_to_keep * 86400)
            conn.execute("DELETE FROM price_history WHERE timestamp < ?", (cutoff,))
            deleted = conn.total_changes
            conn.commit()
            conn.close()
            return deleted
        except Exception as e:
            print(f"[PORTFOLIO] Failed to cleanup old prices: {e}")
            return 0


# Global instance
_store: PortfolioStore | None = None


def get_portfolio_store(db_path: str = DEFAULT_DB_PATH) -> PortfolioStore:
    """Get singleton portfolio store instance."""
    global _store
    if _store is None or _store.db_path != db_path:
        _store = PortfolioStore(db_path)
    return _store
