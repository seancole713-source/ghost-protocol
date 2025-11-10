"""
Algorithmic Footprint Detection
Identifies machine-driven trading patterns: HFT bursts, VWAP bots, spoofing, liquidity sweeps
"""

import logging
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class AlgoPattern:
    """Detected algorithmic trading pattern"""

    pattern_id: str
    symbol: str
    pattern_type: str  # "hft_burst", "vwap_bot", "spoofing", "liquidity_sweep", "momentum_ignition"
    confidence: float  # 0-100
    detected_at: int
    description: str
    indicators: dict[str, Any]  # Can contain floats, strings, etc.
    risk_level: str  # "low", "medium", "high"
    recommendation: str


@dataclass
class MicrostructureSnapshot:
    """Order book microstructure snapshot"""

    symbol: str
    timestamp: int
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    spread: float
    spread_pct: float
    last_trade_size: int
    last_trade_price: float
    volume_1min: int


class AlgoFootprintDetector:
    """
    Detects algorithmic trading patterns in market microstructure
    Uses statistical analysis of price/volume/spread patterns
    """

    def __init__(self, db_path: str = "data/algo_patterns.db"):
        self.db_path = db_path
        self._init_db()

        # Circular buffers for pattern detection
        self.price_buffers: dict[str, deque] = {}
        self.volume_buffers: dict[str, deque] = {}
        self.spread_buffers: dict[str, deque] = {}
        self.trade_size_buffers: dict[str, deque] = {}

        # Buffer sizes
        self.buffer_size = 300  # 5 minutes at 1-second resolution

    def _init_db(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS algo_patterns (
                pattern_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                detected_at INTEGER NOT NULL,
                description TEXT,
                indicators TEXT,
                risk_level TEXT,
                recommendation TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS microstructure_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bid REAL,
                ask REAL,
                bid_size INTEGER,
                ask_size INTEGER,
                spread REAL,
                spread_pct REAL,
                last_trade_size INTEGER,
                last_trade_price REAL,
                volume_1min INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def update_microstructure(self, snapshot: MicrostructureSnapshot):
        """Update microstructure data and detect patterns"""
        symbol = snapshot.symbol

        # Initialize buffers if needed
        if symbol not in self.price_buffers:
            self.price_buffers[symbol] = deque(maxlen=self.buffer_size)
            self.volume_buffers[symbol] = deque(maxlen=self.buffer_size)
            self.spread_buffers[symbol] = deque(maxlen=self.buffer_size)
            self.trade_size_buffers[symbol] = deque(maxlen=self.buffer_size)

        # Add to buffers
        self.price_buffers[symbol].append(snapshot.last_trade_price)
        self.volume_buffers[symbol].append(snapshot.volume_1min)
        self.spread_buffers[symbol].append(snapshot.spread_pct)
        self.trade_size_buffers[symbol].append(snapshot.last_trade_size)

        # Save snapshot
        self._save_snapshot(snapshot)

        # Detect patterns (requires enough data)
        if len(self.price_buffers[symbol]) >= 60:  # At least 1 minute
            patterns = self._detect_patterns(symbol)

            for pattern in patterns:
                self._save_pattern(pattern)
                LOGGER.info(
                    f"Detected {pattern.pattern_type} on {symbol}: {pattern.confidence:.1f}% confidence"
                )

    def _detect_patterns(self, symbol: str) -> list[AlgoPattern]:
        """Run all pattern detection algorithms"""
        patterns = []

        # 1. HFT Burst Detection
        hft = self._detect_hft_burst(symbol)
        if hft:
            patterns.append(hft)

        # 2. VWAP Bot Detection
        vwap = self._detect_vwap_bot(symbol)
        if vwap:
            patterns.append(vwap)

        # 3. Spoofing Detection
        spoof = self._detect_spoofing(symbol)
        if spoof:
            patterns.append(spoof)

        # 4. Liquidity Sweep Detection
        sweep = self._detect_liquidity_sweep(symbol)
        if sweep:
            patterns.append(sweep)

        # 5. Momentum Ignition Detection
        momentum = self._detect_momentum_ignition(symbol)
        if momentum:
            patterns.append(momentum)

        return patterns

    def _detect_hft_burst(self, symbol: str) -> AlgoPattern | None:
        """
        Detect high-frequency trading bursts
        Indicators: Sudden spike in trade count with small lot sizes
        """
        trade_sizes = list(self.trade_size_buffers[symbol])
        volumes = list(self.volume_buffers[symbol])

        if len(trade_sizes) < 60:
            return None

        # Recent 30 seconds
        recent_sizes = trade_sizes[-30:]
        recent_volumes = volumes[-30:]

        # Baseline from previous 30 seconds
        baseline_sizes = trade_sizes[-60:-30]
        baseline_volumes = volumes[-60:-30]

        # Small lot size increase (HFT uses small orders)
        avg_recent_size = np.mean(recent_sizes)
        avg_baseline_size = np.mean(baseline_sizes)

        # Volume spike
        avg_recent_volume = np.mean(recent_volumes)
        avg_baseline_volume = np.mean(baseline_volumes)

        # HFT signature: high volume but small average trade size
        if (
            avg_recent_volume > avg_baseline_volume * 2.0
            and avg_recent_size < avg_baseline_size * 0.7
        ):
            # Calculate confidence
            volume_ratio = avg_recent_volume / max(1, avg_baseline_volume)
            size_ratio = avg_baseline_size / max(1, avg_recent_size)

            confidence = float(min(100, (volume_ratio + size_ratio) * 15))

            if confidence > 60:
                return AlgoPattern(
                    pattern_id=f"hft_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    pattern_type="hft_burst",
                    confidence=confidence,
                    detected_at=int(time.time()),
                    description="Detected algorithmic momentum ignition - likely HFT activity",
                    indicators={
                        "volume_spike": float(volume_ratio),
                        "avg_trade_size": float(avg_recent_size),
                        "trade_count_estimate": float(
                            int(avg_recent_volume / max(1, avg_recent_size))
                        ),
                    },
                    risk_level="medium",
                    recommendation="Monitor for volatility spike; consider widening stop-loss",
                )

        return None

    def _detect_vwap_bot(self, symbol: str) -> AlgoPattern | None:
        """
        Detect VWAP algorithmic trading
        Indicators: Repeating trade patterns at regular intervals
        """
        volumes = list(self.volume_buffers[symbol])

        if len(volumes) < 120:  # Need 2 minutes
            return None

        # Check for periodic spikes (VWAP bots trade in intervals)
        recent = volumes[-120:]  # Last 2 minutes

        # Calculate autocorrelation at different lags
        # VWAP bots show peaks at regular intervals (e.g., every 15 seconds)
        lags_to_check = [15, 30, 60]  # Common VWAP intervals

        max_correlation = 0.0
        best_lag = 0

        for lag in lags_to_check:
            if lag >= len(recent):
                continue

            # Split into segments
            segments = [recent[i : i + lag] for i in range(0, len(recent) - lag, lag)]
            if len(segments) < 3:
                continue

            # Calculate variance across segments
            segment_avgs = [np.mean(seg) for seg in segments if len(seg) == lag]

            if len(segment_avgs) < 2:
                continue

            # If segments have similar volumes → periodic pattern
            variance = np.var(segment_avgs)
            mean_vol = np.mean(segment_avgs)

            if mean_vol > 0:
                cv = np.sqrt(variance) / mean_vol  # Coefficient of variation

                # Low CV = repeating pattern
                if cv < 0.3:
                    correlation = 1.0 - cv
                    if correlation > max_correlation:
                        max_correlation = correlation
                        best_lag = lag

        if max_correlation > 0.7:
            confidence = float(max_correlation * 100)

            return AlgoPattern(
                pattern_id=f"vwap_{symbol}_{int(time.time())}",
                symbol=symbol,
                pattern_type="vwap_bot",
                confidence=confidence,
                detected_at=int(time.time()),
                description="Pattern suggests automated institutional accumulation (VWAP bot)",
                indicators={
                    "periodicity": best_lag,
                    "correlation": max_correlation,
                    "avg_volume_per_interval": float(np.mean(volumes[-60:])),
                },
                risk_level="low",
                recommendation="Institutional buying detected; consider following the trend",
            )

        return None

    def _detect_spoofing(self, symbol: str) -> AlgoPattern | None:
        """
        Detect spoofing patterns
        Indicators: Large orders that disappear quickly without execution
        """
        spreads = list(self.spread_buffers[symbol])

        if len(spreads) < 60:
            return None

        recent_spreads = spreads[-30:]
        baseline_spreads = spreads[-60:-30]

        # Spoofing causes temporary spread widening
        avg_recent = np.mean(recent_spreads)
        avg_baseline = np.mean(baseline_spreads)

        # Check for sudden spread spike that reverses quickly
        if avg_recent > avg_baseline * 1.5:
            # Check if spread is reverting (last 10 seconds)
            last_10 = spreads[-10:]

            if np.mean(last_10) < avg_recent * 0.8:
                # Spread spiked and is reverting → possible spoof

                confidence = float(min(100, ((avg_recent / avg_baseline) - 1) * 100))

                if confidence > 50:
                    return AlgoPattern(
                        pattern_id=f"spoof_{symbol}_{int(time.time())}",
                        symbol=symbol,
                        pattern_type="spoofing",
                        confidence=confidence,
                        detected_at=int(time.time()),
                        description="Detected possible spoof sequence - volatility risk elevated",
                        indicators={
                            "spread_spike_pct": ((avg_recent / avg_baseline) - 1) * 100,
                            "reversion_speed": (avg_recent - np.mean(last_10)) / avg_recent,
                        },
                        risk_level="high",
                        recommendation="Avoid trading; wait for price stability",
                    )

        return None

    def _detect_liquidity_sweep(self, symbol: str) -> AlgoPattern | None:
        """
        Detect liquidity sweep patterns
        Indicators: Rapid price movement with volume spike
        """
        prices = list(self.price_buffers[symbol])
        volumes = list(self.volume_buffers[symbol])

        if len(prices) < 60:
            return None

        # Check last 30 seconds
        recent_prices = prices[-30:]
        recent_volumes = volumes[-30:]

        baseline_volumes = volumes[-60:-30]

        # Price moved quickly
        price_change_pct = ((recent_prices[-1] - recent_prices[0]) / recent_prices[0]) * 100

        # Volume spiked
        avg_recent_vol = np.mean(recent_volumes)
        avg_baseline_vol = np.mean(baseline_volumes)

        volume_ratio = avg_recent_vol / max(1, avg_baseline_vol)

        # Sweep: large price move + high volume
        if abs(price_change_pct) > 0.5 and volume_ratio > 2.0:
            direction = "upward" if price_change_pct > 0 else "downward"

            confidence = float(min(100, (abs(price_change_pct) * 10 + volume_ratio * 15)))

            if confidence > 60:
                return AlgoPattern(
                    pattern_id=f"sweep_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    pattern_type="liquidity_sweep",
                    confidence=confidence,
                    detected_at=int(time.time()),
                    description=f"Liquidity sweep detected: {direction} pressure",
                    indicators={
                        "price_change_pct": price_change_pct,
                        "volume_ratio": volume_ratio,
                        "direction": direction,
                    },
                    risk_level="high",
                    recommendation=f"Strong {direction} momentum; {'ride the wave' if direction == 'upward' else 'consider exit'}",
                )

        return None

    def _detect_momentum_ignition(self, symbol: str) -> AlgoPattern | None:
        """
        Detect momentum ignition algorithms
        Indicators: Sudden coordinated buying/selling across multiple time frames
        """
        prices = list(self.price_buffers[symbol])
        volumes = list(self.volume_buffers[symbol])

        if len(prices) < 120:
            return None

        # Multiple timeframes
        tf_1min = prices[-60:]
        tf_2min = prices[-120:]

        vol_1min = volumes[-60:]
        vol_2min = volumes[-120:]

        # Price momentum
        momentum_1min = ((tf_1min[-1] - tf_1min[0]) / tf_1min[0]) * 100
        momentum_2min = ((tf_2min[-1] - tf_2min[0]) / tf_2min[0]) * 100

        # Volume acceleration
        vol_recent = np.mean(vol_1min)
        vol_baseline = np.mean(vol_2min[:60])

        vol_accel = vol_recent / max(1, vol_baseline)

        # Momentum ignition: accelerating price + volume across timeframes
        if abs(momentum_1min) > 1.0 and momentum_1min * momentum_2min > 0 and vol_accel > 2.5:
            direction = "bullish" if momentum_1min > 0 else "bearish"

            confidence = float(min(100, (abs(momentum_1min) * 10 + vol_accel * 15)))

            if confidence > 70:
                return AlgoPattern(
                    pattern_id=f"ignition_{symbol}_{int(time.time())}",
                    symbol=symbol,
                    pattern_type="momentum_ignition",
                    confidence=confidence,
                    detected_at=int(time.time()),
                    description=f"Momentum ignition detected: {direction} thrust",
                    indicators={
                        "momentum_1min_pct": momentum_1min,
                        "momentum_2min_pct": momentum_2min,
                        "volume_acceleration": vol_accel,
                        "direction": direction,
                    },
                    risk_level="medium",
                    recommendation=f"Algo-driven {direction} momentum; {'follow with tight stop' if direction == 'bullish' else 'avoid/exit positions'}",
                )

        return None

    def _save_snapshot(self, snapshot: MicrostructureSnapshot):
        """Save microstructure snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO microstructure_snapshots (
                symbol, timestamp, bid, ask, bid_size, ask_size,
                spread, spread_pct, last_trade_size, last_trade_price, volume_1min
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                snapshot.symbol,
                snapshot.timestamp,
                snapshot.bid,
                snapshot.ask,
                snapshot.bid_size,
                snapshot.ask_size,
                snapshot.spread,
                snapshot.spread_pct,
                snapshot.last_trade_size,
                snapshot.last_trade_price,
                snapshot.volume_1min,
            ),
        )

        conn.commit()
        conn.close()

    def _save_pattern(self, pattern: AlgoPattern):
        """Save detected pattern"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        import json

        cursor.execute(
            """
            INSERT OR REPLACE INTO algo_patterns (
                pattern_id, symbol, pattern_type, confidence, detected_at,
                description, indicators, risk_level, recommendation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pattern.pattern_id,
                pattern.symbol,
                pattern.pattern_type,
                pattern.confidence,
                pattern.detected_at,
                pattern.description,
                json.dumps(pattern.indicators),
                pattern.risk_level,
                pattern.recommendation,
            ),
        )

        conn.commit()
        conn.close()

    def get_recent_patterns(self, symbol: str | None = None, hours: int = 24) -> list[AlgoPattern]:
        """Get recently detected patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cutoff = int(time.time()) - (hours * 3600)

        if symbol:
            cursor.execute(
                """
                SELECT * FROM algo_patterns
                WHERE symbol = ? AND detected_at > ?
                ORDER BY detected_at DESC
            """,
                (symbol, cutoff),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM algo_patterns
                WHERE detected_at > ?
                ORDER BY detected_at DESC
            """,
                (cutoff,),
            )

        patterns = []
        import json

        for row in cursor.fetchall():
            patterns.append(
                AlgoPattern(
                    pattern_id=row[0],
                    symbol=row[1],
                    pattern_type=row[2],
                    confidence=row[3],
                    detected_at=row[4],
                    description=row[5],
                    indicators=json.loads(row[6]),
                    risk_level=row[7],
                    recommendation=row[8],
                )
            )

        conn.close()
        return patterns


# Singleton
_algo_detector: AlgoFootprintDetector | None = None


def get_algo_detector() -> AlgoFootprintDetector:
    """Get singleton algo detector"""
    global _algo_detector
    if _algo_detector is None:
        _algo_detector = AlgoFootprintDetector()
    return _algo_detector
