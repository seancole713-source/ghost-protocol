"""
PILLAR 3: Volume & Volatility Engine
====================================

Analyzes trading volume patterns and price volatility.

Signals:
- Volume spikes detection
- Realized volatility (20-day, 60-day)
- Volume moving averages
- Volume rate-of-change
- Abnormal volume indicators

Author: Ghost AI
Date: November 21, 2025
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from core.data_pillars.base_pillar import BasePillar, DataSignal, PillarResponse

logger = logging.getLogger(__name__)


class VolumeEngine(BasePillar):
    """Volume and volatility analysis engine."""

    def __init__(self):
        super().__init__(pillar_name="volume_engine")

    def get_signals(self, symbol: str, **kwargs) -> PillarResponse:
        """
        Analyze volume and volatility for a symbol.
        
        Returns:
            Signals: VOLUME_SPIKE, VOLATILITY_20D, VOLATILITY_60D, 
                     VOLUME_MA_20, VOLUME_ROC
        """
        self._start_timer()
        signals = []
        errors = []

        try:
            period_days = kwargs.get("period", 90)
            hist_data = self._fetch_historical_data(symbol, period_days)

            if hist_data is None or len(hist_data) < 20:
                errors.append(f"Insufficient volume data for {symbol}")
                signals = self._create_unavailable_signals()
            else:
                signals = self._calculate_volume_signals(hist_data, symbol)

        except Exception as e:
            logger.error(f"Volume engine failed for {symbol}: {e}")
            errors.append(f"Volume calculation exception: {str(e)}")
            signals = self._create_unavailable_signals()

        return PillarResponse(
            pillar_name=self.pillar_name,
            symbol=symbol,
            signals=signals,
            errors=errors,
            execution_time_ms=self._get_execution_time_ms(),
            timestamp=time.time(),
            cached=False,
        )

    def _fetch_historical_data(self, symbol: str, days: int) -> pd.DataFrame | None:
        """Fetch historical price/volume data using yfinance"""
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist is None or len(hist) < 20:
                return None

            df = hist.reset_index()
            df = df.rename(columns={
                "Date": "timestamp",
                "Close": "close",
                "Volume": "volume"
            })
            
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(int) // 10**9
            
            if "close" not in df.columns:
                df["close"] = 0
            if "volume" not in df.columns:
                df["volume"] = 0

            return df

        except Exception as e:
            logger.error(f"Historical data fetch failed for {symbol}: {e}")
            return None

    def _calculate_volume_signals(
        self, df: pd.DataFrame, symbol: str
    ) -> list[DataSignal]:
        """Calculate volume and volatility signals"""
        signals = []
        ts = time.time()

        try:
            close = df["close"]
            volume = df["volume"]

            # Volume spike detection (current vs 20-day average)
            if len(volume) >= 20:
                vol_ma_20 = volume.rolling(window=20).mean()
                current_vol = volume.iloc[-1]
                avg_vol = vol_ma_20.iloc[-1]

                if avg_vol > 0:
                    volume_spike = (current_vol / avg_vol) - 1.0
                    signals.append(
                        DataSignal(
                            name="VOLUME_SPIKE",
                            value=round(float(volume_spike), 2),
                            confidence=1.0,
                            data_available=True,
                            source="calculated",
                            timestamp=ts,
                            metadata={"current_vol": int(current_vol), "avg_vol": int(avg_vol)},
                        )
                    )

            # 20-day realized volatility
            if len(close) >= 20:
                returns = close.pct_change().dropna()
                vol_20d = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized %
                signals.append(
                    DataSignal(
                        name="VOLATILITY_20D",
                        value=round(float(vol_20d), 2),
                        confidence=1.0,
                        data_available=True,
                        source="calculated",
                        timestamp=ts,
                        metadata={"period": 20, "annualized": True},
                    )
                )

            # 60-day realized volatility
            if len(close) >= 60:
                returns = close.pct_change().dropna()
                vol_60d = returns.tail(60).std() * np.sqrt(252) * 100
                signals.append(
                    DataSignal(
                        name="VOLATILITY_60D",
                        value=round(float(vol_60d), 2),
                        confidence=1.0,
                        data_available=True,
                        source="calculated",
                        timestamp=ts,
                        metadata={"period": 60, "annualized": True},
                    )
                )

            # Volume moving average
            if len(volume) >= 20:
                vol_ma = volume.rolling(window=20).mean().iloc[-1]
                signals.append(
                    DataSignal(
                        name="VOLUME_MA_20",
                        value=int(vol_ma),
                        confidence=1.0,
                        data_available=True,
                        source="calculated",
                        timestamp=ts,
                        metadata={"period": 20},
                    )
                )

            # Volume rate of change (10-day)
            if len(volume) >= 10:
                vol_roc = (volume.iloc[-1] / volume.iloc[-10] - 1.0) * 100
                signals.append(
                    DataSignal(
                        name="VOLUME_ROC",
                        value=round(float(vol_roc), 2),
                        confidence=1.0,
                        data_available=True,
                        source="calculated",
                        timestamp=ts,
                        metadata={"period": 10},
                    )
                )

        except Exception as e:
            logger.error(f"Volume signal calculation failed for {symbol}: {e}")

        return signals

    def _create_unavailable_signals(self) -> list[DataSignal]:
        """Create unavailable signals when data missing"""
        return [
            self._create_unavailable_signal(name, "Volume data unavailable")
            for name in self.get_signal_names()
        ]

    def get_signal_names(self) -> list[str]:
        """Get list of volume signal names"""
        return [
            "VOLUME_SPIKE",
            "VOLATILITY_20D",
            "VOLATILITY_60D",
            "VOLUME_MA_20",
            "VOLUME_ROC",
        ]

    def health_check(self) -> dict[str, Any]:
        """Verify volume engine can calculate signals"""
        results = {
            "ok": True,
            "pillar": self.pillar_name,
            "providers": [],
            "errors": [],
        }

        try:
            spy_response = self.get_signals("SPY", period=90)

            if spy_response.available_signal_count() >= 3:
                results["providers"].append(
                    {
                        "name": "volume_calculator",
                        "status": "ok",
                        "latency_ms": spy_response.execution_time_ms,
                        "signals_computed": spy_response.available_signal_count(),
                    }
                )
            else:
                results["ok"] = False
                results["errors"].append("Volume engine failed health check")

        except Exception as e:
            results["ok"] = False
            results["errors"].append(f"Volume engine health check failed: {e}")

        return results
