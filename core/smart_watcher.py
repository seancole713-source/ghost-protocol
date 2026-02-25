"""
GHOST Smart Watcher - 25-Ticker Real-Time Market Hunter
Level 10 autonomous watchlist with news→ticker linking, proactive signals,
and self-calibrating learning loop.

Features:
- 25-ticker simultaneous tracking
- Real-time quote monitoring
- News→Ticker auto-linking with sentiment
- Proactive Buy/Sell/Hold signals with reasons
- Hit-rate tracking and self-adjustment
- Macro risk radar (SPY/QQQ/VIX)
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

LOGGER = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types"""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    AVOID = "AVOID"


class MacroRegime(Enum):
    """Market macro regime"""

    BULL = "bull"
    BEAR = "bear"
    VOLATILE = "volatile"
    SIDEWAYS = "sideways"


@dataclass
class WatchlistTicker:
    """Individual ticker in watchlist"""

    symbol: str
    added_at: int
    last_price: float
    price_24h_ago: float
    price_change_pct: float
    volume: int
    avg_volume_20d: int
    sentiment_score: float  # Aggregated from news
    signal: str  # BUY/SELL/HOLD/AVOID
    signal_confidence: float  # 0-100
    signal_reason: str
    signal_timestamp: int
    last_updated: int


@dataclass
class TradingSignal:
    """Proactive trading signal"""

    signal_id: str
    symbol: str
    signal_type: str  # BUY/SELL/HOLD/AVOID
    confidence: float  # 0-100
    reason: str
    price_at_signal: float
    target_price: float | None
    stop_loss: float | None
    timestamp: int
    news_drivers: list[str]  # Headlines that influenced decision
    technical_factors: list[str]  # Technical indicators
    macro_context: str  # Overall market condition

    # Outcome tracking (filled later)
    price_24h: float | None = None
    price_48h: float | None = None
    outcome: str | None = None  # "profitable", "loss", "neutral"
    actual_return_pct: float | None = None


@dataclass
class SignalPerformance:
    """Performance metrics for signals"""

    symbol: str
    signal_type: str
    total_signals: int
    profitable: int
    losses: int
    neutral: int
    hit_rate: float  # % profitable
    avg_return_pct: float
    best_return_pct: float
    worst_return_pct: float
    avg_confidence: float
    last_updated: int


@dataclass
class MacroSnapshot:
    """Market macro conditions"""

    timestamp: int
    spy_price: float
    spy_change_pct: float
    qqq_price: float
    qqq_change_pct: float
    vix_level: float
    vix_change_pct: float
    regime: str
    risk_level: str  # "low", "medium", "high", "extreme"
    pause_signals: bool  # Auto-pause if volatility too high


class SmartWatcher:
    """
    Level 10 Smart Watcher + Market Hunter
    Autonomous 25-ticker monitoring with self-calibrating signals
    """

    def __init__(self, db_path: str = "data/smart_watcher.db"):
        self.db_path = db_path
        self.max_tickers = 25
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Watchlist table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                symbol TEXT PRIMARY KEY,
                added_at INTEGER NOT NULL,
                last_price REAL,
                price_24h_ago REAL,
                price_change_pct REAL,
                volume INTEGER,
                avg_volume_20d INTEGER,
                sentiment_score REAL DEFAULT 0.0,
                signal TEXT,
                signal_confidence REAL,
                signal_reason TEXT,
                signal_timestamp INTEGER,
                last_updated INTEGER
            )
        """)

        # Trading signals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trading_signals (
                signal_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT,
                price_at_signal REAL NOT NULL,
                target_price REAL,
                stop_loss REAL,
                timestamp INTEGER NOT NULL,
                news_drivers TEXT,
                technical_factors TEXT,
                macro_context TEXT,
                price_24h REAL,
                price_48h REAL,
                outcome TEXT,
                actual_return_pct REAL,
                FOREIGN KEY (symbol) REFERENCES watchlist(symbol)
            )
        """)

        # Signal performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                total_signals INTEGER DEFAULT 0,
                profitable INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                neutral INTEGER DEFAULT 0,
                hit_rate REAL DEFAULT 0.0,
                avg_return_pct REAL DEFAULT 0.0,
                best_return_pct REAL DEFAULT 0.0,
                worst_return_pct REAL DEFAULT 0.0,
                avg_confidence REAL DEFAULT 0.0,
                last_updated INTEGER,
                PRIMARY KEY (symbol, signal_type)
            )
        """)

        # Macro snapshots table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS macro_snapshots (
                timestamp INTEGER PRIMARY KEY,
                spy_price REAL,
                spy_change_pct REAL,
                qqq_price REAL,
                qqq_change_pct REAL,
                vix_level REAL,
                vix_change_pct REAL,
                regime TEXT,
                risk_level TEXT,
                pause_signals INTEGER DEFAULT 0
            )
        """)

        # News-ticker linkage table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_ticker_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                sentiment_score REAL,
                relevance_score REAL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY (symbol) REFERENCES watchlist(symbol)
            )
        """)

        conn.commit()
        conn.close()

        LOGGER.info(f"Smart Watcher initialized: {self.db_path}")

    def add_ticker(self, symbol: str) -> dict[str, Any]:
        """Add ticker to watchlist (max 25)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if already exists
        cursor.execute("SELECT symbol FROM watchlist WHERE symbol = ?", (symbol,))
        if cursor.fetchone():
            conn.close()
            return {"success": False, "message": f"{symbol} already in watchlist"}

        # Check capacity
        cursor.execute("SELECT COUNT(*) FROM watchlist")
        count = cursor.fetchone()[0]
        if count >= self.max_tickers:
            conn.close()
            return {"success": False, "message": f"Watchlist full ({self.max_tickers} max)"}

        # Add ticker
        cursor.execute(
            """
            INSERT INTO watchlist (symbol, added_at, last_updated)
            VALUES (?, ?, ?)
        """,
            (symbol, int(time.time()), int(time.time())),
        )

        conn.commit()
        conn.close()

        LOGGER.info(f"Added {symbol} to watchlist")
        return {"success": True, "symbol": symbol, "position": count + 1}

    def remove_ticker(self, symbol: str) -> bool:
        """Remove ticker from watchlist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        if deleted:
            LOGGER.info(f"Removed {symbol} from watchlist")
        return deleted

    def get_watchlist(self) -> list[WatchlistTicker]:
        """Get all tickers in watchlist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT symbol, added_at, last_price, price_24h_ago, price_change_pct,
                   volume, avg_volume_20d, sentiment_score, signal, signal_confidence,
                   signal_reason, signal_timestamp, last_updated
            FROM watchlist
            ORDER BY added_at
        """)

        tickers = []
        for row in cursor.fetchall():
            tickers.append(
                WatchlistTicker(
                    symbol=row[0],
                    added_at=row[1],
                    last_price=row[2] or 0.0,
                    price_24h_ago=row[3] or 0.0,
                    price_change_pct=row[4] or 0.0,
                    volume=row[5] or 0,
                    avg_volume_20d=row[6] or 0,
                    sentiment_score=row[7] or 0.0,
                    signal=row[8] or "HOLD",
                    signal_confidence=row[9] or 0.0,
                    signal_reason=row[10] or "",
                    signal_timestamp=row[11] or 0,
                    last_updated=row[12] or 0,
                )
            )

        conn.close()
        return tickers

    def update_ticker_price(self, symbol: str, price: float, volume: int, avg_volume: int) -> bool:
        """Update ticker price and volume data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get previous price
        cursor.execute(
            """
            SELECT last_price, price_24h_ago, last_updated
            FROM watchlist WHERE symbol = ?
        """,
            (symbol,),
        )

        row = cursor.fetchone()
        if not row:
            conn.close()
            return False

        prev_price = row[0] or price
        price_24h_ago = row[1] or price
        last_updated = row[2] or 0

        # If 24h passed, update 24h_ago price
        now = int(time.time())
        if now - last_updated >= 86400:  # 24 hours
            price_24h_ago = prev_price

        # Calculate change
        if price_24h_ago > 0:
            price_change_pct = ((price - price_24h_ago) / price_24h_ago) * 100
        else:
            price_change_pct = 0.0

        cursor.execute(
            """
            UPDATE watchlist
            SET last_price = ?, price_24h_ago = ?, price_change_pct = ?,
                volume = ?, avg_volume_20d = ?, last_updated = ?
            WHERE symbol = ?
        """,
            (price, price_24h_ago, price_change_pct, volume, avg_volume, now, symbol),
        )

        conn.commit()
        conn.close()
        return True

    def update_ticker_sentiment(self, symbol: str, sentiment_score: float) -> bool:
        """Update ticker sentiment from news"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE watchlist SET sentiment_score = ?, last_updated = ?
            WHERE symbol = ?
        """,
            (sentiment_score, int(time.time()), symbol),
        )

        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()

        return updated

    def generate_signal(
        self,
        symbol: str,
        forecast_data: dict[str, Any],
        news_headlines: list[str],
        technical_factors: list[str],
        macro_context: str,
    ) -> TradingSignal:
        """
        Generate proactive trading signal
        Combines: forecast + sentiment + technical + macro
        """
        # Get current price
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT last_price, sentiment_score, price_change_pct
            FROM watchlist WHERE symbol = ?
        """,
            (symbol,),
        )

        row = cursor.fetchone()
        if not row:
            conn.close()
            raise ValueError(f"Ticker {symbol} not in watchlist")

        current_price = row[0] or 0.0
        sentiment = row[1] or 0.0
        momentum = row[2] or 0.0

        # Calculate signal components
        forecast_score = forecast_data.get("predicted_return", 0.0)
        risk_level = forecast_data.get("risk_level", "medium")

        # Weighted signal calculation
        # 40% forecast, 30% sentiment, 20% momentum, 10% macro
        signal_score = (
            forecast_score * 0.4
            + sentiment * 30.0 * 0.3  # Scale sentiment -1 to 1 → -30 to 30
            + momentum * 0.2
            + self._macro_adjustment(macro_context) * 0.1
        )

        # Get asset-specific stop from classifier
        try:
            from core.asset_classifier import AssetClassifier
            targets = AssetClassifier.get_target_stop(symbol, horizon_hours=48)
            stop_pct = targets["stop_pct"]
        except Exception:
            stop_pct = 4.0  # Fallback
        
        # Determine signal type
        if signal_score > 5.0 and risk_level != "extreme":
            signal_type = SignalType.BUY
            target_price = current_price * (1 + signal_score / 100)
            stop_loss = current_price * (1 - stop_pct / 100)
        elif signal_score < -5.0:
            signal_type = SignalType.SELL
            target_price = current_price * (1 + signal_score / 100)
            stop_loss = current_price * (1 + stop_pct / 100)
        elif risk_level == "extreme":
            signal_type = SignalType.AVOID
            target_price = None
            stop_loss = None
        else:
            signal_type = SignalType.HOLD
            target_price = None
            stop_loss = None

        # Calculate confidence
        confidence = min(100, abs(signal_score) * 5)  # Scale to 0-100

        # Build reason
        reason = self._build_signal_reason(
            forecast_score, sentiment, momentum, risk_level, macro_context
        )

        # Create signal
        signal = TradingSignal(
            signal_id=f"{symbol}_{int(time.time())}",
            symbol=symbol,
            signal_type=signal_type.value,
            confidence=confidence,
            reason=reason,
            price_at_signal=current_price,
            target_price=target_price,
            stop_loss=stop_loss,
            timestamp=int(time.time()),
            news_drivers=news_headlines[:5],  # Top 5
            technical_factors=technical_factors,
            macro_context=macro_context,
        )

        # Save signal
        self._save_signal(signal)

        # Update watchlist with signal
        cursor.execute(
            """
            UPDATE watchlist
            SET signal = ?, signal_confidence = ?, signal_reason = ?,
                signal_timestamp = ?
            WHERE symbol = ?
        """,
            (signal_type.value, confidence, reason, signal.timestamp, symbol),
        )

        conn.commit()
        conn.close()

        LOGGER.info(
            f"Generated {signal_type.value} signal for {symbol}: {confidence:.1f}% confidence"
        )
        return signal

    def _macro_adjustment(self, macro_context: str) -> float:
        """Convert macro context to adjustment score"""
        if "bull" in macro_context.lower():
            return 5.0
        elif "bear" in macro_context.lower():
            return -5.0
        elif "volatile" in macro_context.lower() or "high" in macro_context.lower():
            return -3.0
        else:
            return 0.0

    def _build_signal_reason(
        self, forecast: float, sentiment: float, momentum: float, risk: str, macro: str
    ) -> str:
        """Build human-readable signal reason"""
        parts = []

        if abs(forecast) > 5:
            parts.append(f"Forecast: {forecast:+.1f}% expected return")

        if sentiment > 0.3:
            parts.append(f"Strong positive news sentiment ({sentiment:.2f})")
        elif sentiment < -0.3:
            parts.append(f"Strong negative news sentiment ({sentiment:.2f})")

        if momentum > 3:
            parts.append(f"Strong upward momentum ({momentum:+.1f}%)")
        elif momentum < -3:
            parts.append(f"Downward momentum ({momentum:+.1f}%)")

        if risk == "extreme":
            parts.append("⚠️ EXTREME risk detected")

        if "volatile" in macro.lower():
            parts.append("Market volatility elevated")

        return " | ".join(parts) if parts else "Neutral conditions"

    def _save_signal(self, signal: TradingSignal):
        """Save signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO trading_signals (
                signal_id, symbol, signal_type, confidence, reason,
                price_at_signal, target_price, stop_loss, timestamp,
                news_drivers, technical_factors, macro_context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                signal.signal_id,
                signal.symbol,
                signal.signal_type,
                signal.confidence,
                signal.reason,
                signal.price_at_signal,
                signal.target_price,
                signal.stop_loss,
                signal.timestamp,
                json.dumps(signal.news_drivers),
                json.dumps(signal.technical_factors),
                signal.macro_context,
            ),
        )

        conn.commit()
        conn.close()

    def update_signal_outcome(
        self, signal_id: str, price_24h: float, price_48h: float | None = None
    ):
        """Update signal outcome after 24h/48h"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get signal
        cursor.execute(
            """
            SELECT signal_type, price_at_signal, symbol
            FROM trading_signals WHERE signal_id = ?
        """,
            (signal_id,),
        )

        row = cursor.fetchone()
        if not row:
            conn.close()
            return

        signal_type = row[0]
        price_at_signal = row[1]
        symbol = row[2]

        # Calculate return
        return_24h = ((price_24h - price_at_signal) / price_at_signal) * 100

        # Determine outcome
        if signal_type == "BUY":
            if return_24h > 2.0:
                outcome = "profitable"
            elif return_24h < -2.0:
                outcome = "loss"
            else:
                outcome = "neutral"
        elif signal_type == "SELL":
            if return_24h < -2.0:
                outcome = "profitable"
            elif return_24h > 2.0:
                outcome = "loss"
            else:
                outcome = "neutral"
        else:
            outcome = "neutral"

        # Update signal
        cursor.execute(
            """
            UPDATE trading_signals
            SET price_24h = ?, price_48h = ?, outcome = ?, actual_return_pct = ?
            WHERE signal_id = ?
        """,
            (price_24h, price_48h, outcome, return_24h, signal_id),
        )

        conn.commit()

        # Update performance stats
        self._update_performance_stats(symbol, signal_type)

        conn.close()
        LOGGER.info(f"Updated outcome for {signal_id}: {outcome} ({return_24h:+.2f}%)")

    def _update_performance_stats(self, symbol: str, signal_type: str):
        """Update performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all completed signals for this symbol/type
        cursor.execute(
            """
            SELECT outcome, actual_return_pct, confidence
            FROM trading_signals
            WHERE symbol = ? AND signal_type = ? AND outcome IS NOT NULL
        """,
            (symbol, signal_type),
        )

        rows = cursor.fetchall()
        if not rows:
            conn.close()
            return

        total = len(rows)
        profitable = sum(1 for r in rows if r[0] == "profitable")
        losses = sum(1 for r in rows if r[0] == "loss")
        neutral = sum(1 for r in rows if r[0] == "neutral")

        returns = [r[1] for r in rows if r[1] is not None]
        confidences = [r[2] for r in rows if r[2] is not None]

        hit_rate = (profitable / total * 100) if total > 0 else 0.0
        avg_return = sum(returns) / len(returns) if returns else 0.0
        best_return = max(returns) if returns else 0.0
        worst_return = min(returns) if returns else 0.0
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Upsert performance
        cursor.execute(
            """
            INSERT OR REPLACE INTO signal_performance (
                symbol, signal_type, total_signals, profitable, losses, neutral,
                hit_rate, avg_return_pct, best_return_pct, worst_return_pct,
                avg_confidence, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                symbol,
                signal_type,
                total,
                profitable,
                losses,
                neutral,
                hit_rate,
                avg_return,
                best_return,
                worst_return,
                avg_confidence,
                int(time.time()),
            ),
        )

        conn.commit()
        conn.close()

    def get_performance(self, symbol: str | None = None) -> list[SignalPerformance]:
        """Get signal performance stats"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if symbol:
            cursor.execute(
                """
                SELECT * FROM signal_performance WHERE symbol = ?
            """,
                (symbol,),
            )
        else:
            cursor.execute("SELECT * FROM signal_performance")

        stats = []
        for row in cursor.fetchall():
            stats.append(
                SignalPerformance(
                    symbol=row[0],
                    signal_type=row[1],
                    total_signals=row[2],
                    profitable=row[3],
                    losses=row[4],
                    neutral=row[5],
                    hit_rate=row[6],
                    avg_return_pct=row[7],
                    best_return_pct=row[8],
                    worst_return_pct=row[9],
                    avg_confidence=row[10],
                    last_updated=row[11],
                )
            )

        conn.close()
        return stats

    def update_macro_snapshot(
        self, spy_price: float, qqq_price: float, vix_level: float
    ) -> MacroSnapshot:
        """Update macro market conditions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get previous snapshot
        cursor.execute("""
            SELECT spy_price, qqq_price, vix_level
            FROM macro_snapshots
            ORDER BY timestamp DESC LIMIT 1
        """)

        row = cursor.fetchone()
        prev_spy = row[0] if row else spy_price
        prev_qqq = row[1] if row else qqq_price
        prev_vix = row[2] if row else vix_level

        # Calculate changes
        spy_change = ((spy_price - prev_spy) / prev_spy * 100) if prev_spy > 0 else 0.0
        qqq_change = ((qqq_price - prev_qqq) / prev_qqq * 100) if prev_qqq > 0 else 0.0
        vix_change = ((vix_level - prev_vix) / prev_vix * 100) if prev_vix > 0 else 0.0

        # Determine regime
        if spy_change > 1.0 and qqq_change > 1.0 and vix_level < 20:
            regime = MacroRegime.BULL.value
            risk_level = "low"
        elif spy_change < -1.0 and qqq_change < -1.0:
            regime = MacroRegime.BEAR.value
            risk_level = "high"
        elif vix_level > 30:
            regime = MacroRegime.VOLATILE.value
            risk_level = "extreme"
        else:
            regime = MacroRegime.SIDEWAYS.value
            risk_level = "medium"

        # Auto-pause if extreme
        pause_signals = risk_level == "extreme"

        # Save snapshot
        snapshot = MacroSnapshot(
            timestamp=int(time.time()),
            spy_price=spy_price,
            spy_change_pct=spy_change,
            qqq_price=qqq_price,
            qqq_change_pct=qqq_change,
            vix_level=vix_level,
            vix_change_pct=vix_change,
            regime=regime,
            risk_level=risk_level,
            pause_signals=pause_signals,
        )

        cursor.execute(
            """
            INSERT INTO macro_snapshots (
                timestamp, spy_price, spy_change_pct, qqq_price, qqq_change_pct,
                vix_level, vix_change_pct, regime, risk_level, pause_signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                snapshot.timestamp,
                snapshot.spy_price,
                snapshot.spy_change_pct,
                snapshot.qqq_price,
                snapshot.qqq_change_pct,
                snapshot.vix_level,
                snapshot.vix_change_pct,
                snapshot.regime,
                snapshot.risk_level,
                int(snapshot.pause_signals),
            ),
        )

        conn.commit()
        conn.close()

        LOGGER.info(f"Macro update: {regime} / Risk: {risk_level} / VIX: {vix_level:.1f}")
        return snapshot

    def get_latest_macro(self) -> MacroSnapshot | None:
        """Get latest macro snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM macro_snapshots
            ORDER BY timestamp DESC LIMIT 1
        """)

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return MacroSnapshot(
            timestamp=row[0],
            spy_price=row[1],
            spy_change_pct=row[2],
            qqq_price=row[3],
            qqq_change_pct=row[4],
            vix_level=row[5],
            vix_change_pct=row[6],
            regime=row[7],
            risk_level=row[8],
            pause_signals=bool(row[9]),
        )

    def link_news_to_ticker(
        self, article_id: str, symbol: str, sentiment_score: float, relevance_score: float
    ):
        """Link news article to ticker"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO news_ticker_links (
                article_id, symbol, sentiment_score, relevance_score, timestamp
            ) VALUES (?, ?, ?, ?, ?)
        """,
            (article_id, symbol, sentiment_score, relevance_score, int(time.time())),
        )

        conn.commit()
        conn.close()

    def get_ticker_news(self, symbol: str, hours: int = 24) -> list[dict[str, Any]]:
        """Get news linked to ticker in last N hours"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = int(time.time()) - (hours * 3600)

        cursor.execute(
            """
            SELECT article_id, sentiment_score, relevance_score, timestamp
            FROM news_ticker_links
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
        """,
            (symbol, cutoff),
        )

        news = []
        for row in cursor.fetchall():
            news.append(
                {
                    "article_id": row[0],
                    "sentiment_score": row[1],
                    "relevance_score": row[2],
                    "timestamp": row[3],
                }
            )

        conn.close()
        return news


# Singleton instance
_smart_watcher_instance: SmartWatcher | None = None


def get_smart_watcher() -> SmartWatcher:
    """Get singleton Smart Watcher instance"""
    global _smart_watcher_instance
    if _smart_watcher_instance is None:
        _smart_watcher_instance = SmartWatcher()
    return _smart_watcher_instance
