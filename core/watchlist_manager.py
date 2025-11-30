"""
GHOST Watchlist Manager
Manages symbols in watchlist and filters them based on GHOST scoring logic.
Only symbols passing GHOST criteria appear in top movers.
"""

import sqlite3
from datetime import datetime
from pathlib import Path


class WatchlistManager:
    """Manages watchlist symbols and GHOST scoring integration."""

    def __init__(self, db_path: str = "watchlist.db"):
        self.db_path = Path(db_path)
        self._init_database()
        self._load_default_watchlist()

    def _init_database(self):
        """Initialize watchlist database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Watchlist symbols table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                added_at TEXT,
                last_updated TEXT,
                metadata TEXT
            )
        """)

        # GHOST scores table (historical tracking)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ghost_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                gps_score REAL,
                price REAL,
                change_pct REAL,
                volume REAL,
                market_cap REAL,
                passed_threshold INTEGER DEFAULT 0,
                FOREIGN KEY (symbol) REFERENCES watchlist(symbol)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scores_symbol_time
            ON ghost_scores(symbol, timestamp DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scores_passed_threshold
            ON ghost_scores(passed_threshold, gps_score DESC)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scores_gps
            ON ghost_scores(gps_score DESC, change_pct)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlist_updated
            ON watchlist(last_updated DESC)
        """)

        conn.commit()
        conn.close()

    def _load_default_watchlist(self):
        """Load expanded default watchlist - NO LIMITS on symbol tracking."""
        default_symbols = [
            # Original watchlist (preserved for continuity)
            ("WFC", "Wells Fargo & Company"),
            ("SLB", "Schlumberger Limited"),
            ("HLN", "Haleon plc"),
            ("CNH", "CNH Industrial N.V."),
            ("KDP", "Keurig Dr Pepper Inc."),
            ("CORZ", "Core Scientific, Inc."),
            ("SBUX", "Starbucks Corporation"),
            ("UWMC", "UWM Holdings Corporation"),
            ("EQT", "EQT Corporation"),
            ("MDT", "Medtronic plc"),
            ("HPQ", "HP Inc."),
            ("ETSY", "Etsy, Inc."),
            ("PBA", "Pembina Pipeline Corporation"),
            ("LVS", "Las Vegas Sands Corp."),
            ("PGY", "Pagaya Technologies Ltd."),
            ("CTRA", "Coterra Energy Inc."),
            ("HBM", "Hudbay Minerals Inc."),
            ("MRNA", "Moderna, Inc."),
            ("SBSW", "Sibanye Stillwater Limited"),
            ("CVS", "CVS Health Corporation"),
            ("KHC", "The Kraft Heinz Company"),
            ("M", "Macy's, Inc."),
            ("VTRS", "Viatris Inc."),
            ("PDD", "PDD Holdings Inc."),
            ("ELAN", "Elanco Animal Health Incorporated"),
            ("CFG", "Citizens Financial Group, Inc."),
            ("CRM", "Salesforce, Inc."),
            ("ENVX", "Enovix Corporation"),
            ("SCHW", "The Charles Schwab Corporation"),
            ("WRD", "WeRide Inc."),
            ("NWL", "Newell Brands Inc."),
            ("CL", "Colgate-Palmolive Company"),
            ("UAA", "Under Armour, Inc."),
            ("EBAY", "eBay Inc."),
            ("IPG", "The Interpublic Group of Companies, Inc."),
            ("NG", "NovaGold Resources Inc."),
            ("SIRI", "Sirius XM Holdings Inc."),
            ("CAH", "Cardinal Health, Inc."),
            ("WMB", "The Williams Companies, Inc."),
            ("PPL", "PPL Corporation"),
            ("MDU", "MDU Resources Group, Inc."),
            ("TFC", "Truist Financial Corporation"),
            ("AEO", "American Eagle Outfitters, Inc."),
            ("GAP", "The Gap, Inc."),
            ("MAT", "Mattel, Inc."),
            ("STUB", "StubHub Holdings, Inc."),
            ("APH", "Amphenol Corporation"),
            ("CNP", "CenterPoint Energy, Inc."),
            ("ANET", "Arista Networks Inc"),
            ("MDLZ", "Mondelez International, Inc."),
            ("USB", "U.S. Bancorp"),
            ("CRDO", "Credo Technology Group Holding Ltd"),
            # Expanded coverage - Mega caps
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft Corporation"),
            ("GOOGL", "Alphabet Inc."),
            ("AMZN", "Amazon.com Inc."),
            ("META", "Meta Platforms Inc."),
            ("TSLA", "Tesla, Inc."),
            ("NVDA", "NVIDIA Corporation"),
            ("BRK.B", "Berkshire Hathaway Inc."),
            # Tech sector expansion
            ("ORCL", "Oracle Corporation"),
            ("ADBE", "Adobe Inc."),
            ("NFLX", "Netflix Inc."),
            ("INTC", "Intel Corporation"),
            ("AMD", "Advanced Micro Devices Inc."),
            ("CSCO", "Cisco Systems Inc."),
            ("IBM", "International Business Machines"),
            ("QCOM", "QUALCOMM Incorporated"),
            ("TXN", "Texas Instruments Incorporated"),
            ("AVGO", "Broadcom Inc."),
            # Finance expansion
            ("JPM", "JPMorgan Chase & Co."),
            ("BAC", "Bank of America Corporation"),
            ("GS", "The Goldman Sachs Group Inc."),
            ("MS", "Morgan Stanley"),
            ("C", "Citigroup Inc."),
            ("BLK", "BlackRock Inc."),
            ("COF", "Capital One Financial Corporation"),
            ("AXP", "American Express Company"),
            ("PNC", "The PNC Financial Services Group"),
            # Healthcare expansion
            ("UNH", "UnitedHealth Group Incorporated"),
            ("JNJ", "Johnson & Johnson"),
            ("PFE", "Pfizer Inc."),
            ("ABBV", "AbbVie Inc."),
            ("TMO", "Thermo Fisher Scientific Inc."),
            ("ABT", "Abbott Laboratories"),
            ("MRK", "Merck & Co. Inc."),
            ("LLY", "Eli Lilly and Company"),
            ("AMGN", "Amgen Inc."),
            ("GILD", "Gilead Sciences Inc."),
            ("BMY", "Bristol-Myers Squibb Company"),
            # Consumer & Retail
            ("WMT", "Walmart Inc."),
            ("HD", "The Home Depot Inc."),
            ("MCD", "McDonald's Corporation"),
            ("NKE", "NIKE Inc."),
            ("TGT", "Target Corporation"),
            ("LOW", "Lowe's Companies Inc."),
            ("DIS", "The Walt Disney Company"),
            ("BKNG", "Booking Holdings Inc."),
            ("ABNB", "Airbnb Inc."),
            # Energy sector
            ("XOM", "Exxon Mobil Corporation"),
            ("CVX", "Chevron Corporation"),
            ("COP", "ConocoPhillips"),
            ("EOG", "EOG Resources Inc."),
            ("PXD", "Pioneer Natural Resources Company"),
            ("MPC", "Marathon Petroleum Corporation"),
            ("PSX", "Phillips 66"),
            ("VLO", "Valero Energy Corporation"),
            ("OXY", "Occidental Petroleum Corporation"),
            # Industrials
            ("BA", "The Boeing Company"),
            ("CAT", "Caterpillar Inc."),
            ("GE", "General Electric Company"),
            ("HON", "Honeywell International Inc."),
            ("UPS", "United Parcel Service Inc."),
            ("LMT", "Lockheed Martin Corporation"),
            ("RTX", "Raytheon Technologies Corporation"),
            ("MMM", "3M Company"),
            ("DE", "Deere & Company"),
            ("UNP", "Union Pacific Corporation"),
            # High momentum/volatility
            ("WOLF", "Wolfspeed Inc."),
            ("GME", "GameStop Corp."),
            ("AMC", "AMC Entertainment Holdings Inc."),
            ("PLTR", "Palantir Technologies Inc."),
            ("SOFI", "SoFi Technologies Inc."),
            ("RIVN", "Rivian Automotive Inc."),
            ("LCID", "Lucid Group Inc."),
            ("NIO", "NIO Inc."),
            ("SNAP", "Snap Inc."),
            ("PINS", "Pinterest Inc."),
            ("UBER", "Uber Technologies Inc."),
            ("LYFT", "Lyft Inc."),
        ]

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Check if watchlist is empty
        cursor.execute("SELECT COUNT(*) FROM watchlist")
        count = cursor.fetchone()[0]

        if count == 0:
            # Add default symbols
            now = datetime.utcnow().isoformat()
            for symbol, name in default_symbols:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO watchlist (symbol, name, added_at, last_updated)
                    VALUES (?, ?, ?, ?)
                """,
                    (symbol, name, now, now),
                )

            conn.commit()

        conn.close()

    def add_symbol(self, symbol: str, name: str = "", metadata: str = "") -> dict:
        """Add symbol to watchlist."""
        symbol = symbol.upper()
        now = datetime.utcnow().isoformat()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO watchlist (symbol, name, added_at, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?)
            """,
                (symbol, name, now, now, metadata),
            )

            conn.commit()
            return {"success": True, "symbol": symbol, "name": name, "added_at": now}
        except sqlite3.IntegrityError:
            return {"success": False, "error": f"Symbol {symbol} already in watchlist"}
        finally:
            conn.close()

    def remove_symbol(self, symbol: str) -> dict:
        """Remove symbol from watchlist."""
        symbol = symbol.upper()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        deleted = cursor.rowcount

        # Also delete historical scores
        cursor.execute("DELETE FROM ghost_scores WHERE symbol = ?", (symbol,))

        conn.commit()
        conn.close()

        return {"success": deleted > 0, "symbol": symbol, "deleted": deleted > 0}

    def get_watchlist(self) -> list[dict]:
        """Get all symbols in watchlist."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT symbol, name, added_at, last_updated, metadata
            FROM watchlist
            ORDER BY symbol
        """)

        symbols = []
        for row in cursor.fetchall():
            symbols.append(
                {
                    "symbol": row[0],
                    "name": row[1],
                    "added_at": row[2],
                    "last_updated": row[3],
                    "metadata": row[4],
                }
            )

        conn.close()
        return symbols

    def update_ghost_score(
        self,
        symbol: str,
        gps_score: float,
        price: float,
        change_pct: float,
        volume: float | None = None,
        market_cap: float | None = None,
        threshold: float = 7.0,
    ) -> dict:
        """
        Update GHOST score for a symbol.

        Args:
            symbol: Stock symbol
            gps_score: GHOST Performance Score (0-10)
            price: Current price
            change_pct: Percent change
            volume: Trading volume
            market_cap: Market capitalization
            threshold: GPS threshold for top movers (default: 7.0)

        Returns:
            Dict with update status and whether symbol passed threshold
        """
        symbol = symbol.upper()
        now = datetime.utcnow().isoformat()
        passed = 1 if gps_score >= threshold else 0

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Check if symbol is in watchlist
        cursor.execute("SELECT symbol FROM watchlist WHERE symbol = ?", (symbol,))
        if not cursor.fetchone():
            conn.close()
            return {"success": False, "error": f"Symbol {symbol} not in watchlist"}

        # Insert score
        cursor.execute(
            """
            INSERT INTO ghost_scores
            (symbol, timestamp, gps_score, price, change_pct, volume, market_cap, passed_threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (symbol, now, gps_score, price, change_pct, volume, market_cap, passed),
        )

        # Update last_updated in watchlist
        cursor.execute(
            """
            UPDATE watchlist SET last_updated = ? WHERE symbol = ?
        """,
            (now, symbol),
        )

        conn.commit()
        conn.close()

        return {
            "success": True,
            "symbol": symbol,
            "gps_score": gps_score,
            "passed_threshold": passed == 1,
            "threshold": threshold,
            "timestamp": now,
        }

    def get_top_movers(
        self, threshold: float = 7.0, limit: int = 20, min_change_pct: float = 0.0
    ) -> list[dict]:
        """
        Get symbols that passed GHOST threshold and qualify as top movers.

        Args:
            threshold: GPS threshold (default: 7.0)
            limit: Maximum number of results
            min_change_pct: Minimum absolute change percent

        Returns:
            List of symbols with their latest scores that passed threshold
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Get latest score for each symbol that passed threshold
        cursor.execute(
            """
            SELECT
                gs.symbol,
                w.name,
                gs.gps_score,
                gs.price,
                gs.change_pct,
                gs.volume,
                gs.market_cap,
                gs.timestamp
            FROM ghost_scores gs
            INNER JOIN watchlist w ON gs.symbol = w.symbol
            INNER JOIN (
                SELECT symbol, MAX(timestamp) as max_time
                FROM ghost_scores
                WHERE passed_threshold = 1
                GROUP BY symbol
            ) latest ON gs.symbol = latest.symbol AND gs.timestamp = latest.max_time
            WHERE gs.gps_score >= ?
              AND ABS(gs.change_pct) >= ?
            ORDER BY gs.gps_score DESC, ABS(gs.change_pct) DESC
            LIMIT ?
        """,
            (threshold, min_change_pct, limit),
        )

        movers = []
        for row in cursor.fetchall():
            movers.append(
                {
                    "symbol": row[0],
                    "sym": row[0],  # Alias for compatibility
                    "name": row[1],
                    "gps": round(row[2], 2),
                    "price": round(row[3], 2) if row[3] else 0.0,
                    "change_pct": round(row[4], 2) if row[4] else 0.0,
                    "volume": row[5],
                    "market_cap": row[6],
                    "timestamp": row[7],
                }
            )

        conn.close()
        return movers

    def get_symbol_history(self, symbol: str, limit: int = 100) -> list[dict]:
        """Get historical GHOST scores for a symbol."""
        symbol = symbol.upper()

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                timestamp,
                gps_score,
                price,
                change_pct,
                volume,
                market_cap,
                passed_threshold
            FROM ghost_scores
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (symbol, limit),
        )

        history = []
        for row in cursor.fetchall():
            history.append(
                {
                    "timestamp": row[0],
                    "gps_score": row[1],
                    "price": row[2],
                    "change_pct": row[3],
                    "volume": row[4],
                    "market_cap": row[5],
                    "passed_threshold": row[6] == 1,
                }
            )

        conn.close()
        return history

    def get_statistics(self) -> dict:
        """Get watchlist statistics."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Total symbols
        cursor.execute("SELECT COUNT(*) FROM watchlist")
        total_symbols = cursor.fetchone()[0]

        # Symbols with scores
        cursor.execute("SELECT COUNT(DISTINCT symbol) FROM ghost_scores")
        symbols_with_scores = cursor.fetchone()[0]

        # Symbols currently passing threshold
        cursor.execute("""
            SELECT COUNT(DISTINCT gs.symbol)
            FROM ghost_scores gs
            INNER JOIN (
                SELECT symbol, MAX(timestamp) as max_time
                FROM ghost_scores
                GROUP BY symbol
            ) latest ON gs.symbol = latest.symbol AND gs.timestamp = latest.max_time
            WHERE gs.passed_threshold = 1
        """)
        symbols_passing = cursor.fetchone()[0]

        # Average GPS score (latest for each symbol)
        cursor.execute("""
            SELECT AVG(gs.gps_score)
            FROM ghost_scores gs
            INNER JOIN (
                SELECT symbol, MAX(timestamp) as max_time
                FROM ghost_scores
                GROUP BY symbol
            ) latest ON gs.symbol = latest.symbol AND gs.timestamp = latest.max_time
        """)
        avg_gps = cursor.fetchone()[0] or 0.0

        conn.close()

        return {
            "total_symbols": total_symbols,
            "symbols_with_scores": symbols_with_scores,
            "symbols_passing_threshold": symbols_passing,
            "average_gps_score": round(avg_gps, 2),
            "pass_rate_pct": round((symbols_passing / total_symbols * 100), 2)
            if total_symbols > 0
            else 0.0,
        }


# Singleton instance
_watchlist_manager: WatchlistManager | None = None


def get_watchlist_manager() -> WatchlistManager:
    """Get singleton watchlist manager instance."""
    global _watchlist_manager
    if _watchlist_manager is None:
        _watchlist_manager = WatchlistManager()
    return _watchlist_manager
