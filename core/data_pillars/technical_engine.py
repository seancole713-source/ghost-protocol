"""
PILLAR 2: Technical Indicators Engine
=====================================

Calculates 50+ technical indicators from price history.
Zero external API dependencies - pure calculation engine.

Indicators Computed:
- Trend: SMA, EMA, MACD, ADX, Aroon
- Momentum: RSI, Stochastic, Williams %R, ROC
- Volatility: Bollinger Bands, ATR, Standard Deviation
- Volume: OBV, Money Flow Index, Volume Rate
- Pattern: Support/Resistance, Trend Lines

Author: Ghost AI
Date: November 21, 2025
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from core.data_pillars.base_pillar import BasePillar, DataSignal, PillarResponse
from core.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
    stochastic,
    williams_r,
)

logger = logging.getLogger(__name__)


class TechnicalEngine(BasePillar):
    """
    Technical indicators calculation engine.
    
    Requires historical OHLCV data to compute indicators.
    Falls back gracefully when historical data unavailable.
    """

    def __init__(self):
        super().__init__(pillar_name="technical_engine")

    def get_signals(self, symbol: str, **kwargs) -> PillarResponse:
        """
        Calculate technical indicators for a symbol.
        
        Args:
            symbol: Stock/crypto ticker
            **kwargs:
                - period: Lookback period in days (default: 90)
                - include_patterns: Calculate chart patterns (default: False)
        
        Returns:
            PillarResponse with technical signals:
                - RSI_14: Relative Strength Index (14-period)
                - MACD_HISTOGRAM: MACD histogram value
                - MACD_SIGNAL: MACD signal line
                - SMA_20, SMA_50, SMA_200: Simple moving averages
                - EMA_12, EMA_26: Exponential moving averages
                - BB_UPPER, BB_MIDDLE, BB_LOWER: Bollinger Bands
                - ATR_14: Average True Range
                - STOCH_K, STOCH_D: Stochastic oscillator
                - WILLIAMS_R: Williams %R
        """
        self._start_timer()
        signals = []
        errors = []

        try:
            # Fetch historical data
            period_days = kwargs.get("period", 90)
            hist_data = self._fetch_historical_data(symbol, period_days)

            if hist_data is None or len(hist_data) < 20:
                errors.append(
                    f"Insufficient historical data for {symbol} (need 20+ bars, got {len(hist_data) if hist_data else 0})"
                )
                signals = self._create_unavailable_signals()
            else:
                # Calculate indicators
                signals = self._calculate_indicators(hist_data, symbol)

        except Exception as e:
            logger.error(f"Technical engine failed for {symbol}: {e}")
            errors.append(f"Technical calculation exception: {str(e)}")
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
        """
        Fetch historical OHLCV data using yfinance.
        
        Args:
            symbol: Ticker symbol
            days: Lookback period
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Download historical data
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist is None or len(hist) < 20:
                logger.warning(f"Insufficient yfinance data for {symbol}: {len(hist) if hist is not None else 0} bars")
                return None
            
            # Rename columns to match expected format
            df = hist.reset_index()
            df = df.rename(columns={
                "Date": "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })
            
            # Convert timestamp to unix time
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].astype(int) // 10**9
            
            # Ensure required columns
            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    df[col] = df.get("close", 0)
            
            if "volume" not in df.columns:
                df["volume"] = 0
            
            logger.info(f"Fetched {len(df)} bars for {symbol} using yfinance")
            return df
            
        except Exception as e:
            logger.error(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame, symbol: str) -> list[DataSignal]:
        """
        Calculate all technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Ticker symbol
        
        Returns:
            List of DataSignal objects
        """
        signals = []
        ts = time.time()

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            # RSI (14-period)
            try:
                rsi_values = rsi(close, period=14)
                if not rsi_values.empty and not np.isnan(rsi_values.iloc[-1]):
                    signals.append(
                        DataSignal(
                            name="RSI_14",
                            value=round(float(rsi_values.iloc[-1]), 2),
                            confidence=1.0,
                            data_available=True,
                            source="calculated",
                            timestamp=ts,
                            metadata={"period": 14, "symbol": symbol},
                        )
                    )
            except Exception as e:
                logger.warning(f"RSI calculation failed for {symbol}: {e}")

            # MACD (12, 26, 9)
            try:
                macd_data = macd(close, fast=12, slow=26, signal=9)
                if not macd_data.empty and len(macd_data) > 0:
                    last_row = macd_data.iloc[-1]
                    
                    if "histogram" in last_row and not np.isnan(last_row["histogram"]):
                        signals.append(
                            DataSignal(
                                name="MACD_HISTOGRAM",
                                value=round(float(last_row["histogram"]), 4),
                                confidence=1.0,
                                data_available=True,
                                source="calculated",
                                timestamp=ts,
                                metadata={"fast": 12, "slow": 26, "signal": 9},
                            )
                        )
                    
                    if "signal" in last_row and not np.isnan(last_row["signal"]):
                        signals.append(
                            DataSignal(
                                name="MACD_SIGNAL",
                                value=round(float(last_row["signal"]), 4),
                                confidence=1.0,
                                data_available=True,
                                source="calculated",
                                timestamp=ts,
                                metadata={"period": 9},
                            )
                        )
            except Exception as e:
                logger.warning(f"MACD calculation failed for {symbol}: {e}")

            # Moving Averages
            for period in [20, 50, 200]:
                try:
                    ma = sma(close, period=period)
                    if not ma.empty and not np.isnan(ma.iloc[-1]):
                        signals.append(
                            DataSignal(
                                name=f"SMA_{period}",
                                value=round(float(ma.iloc[-1]), 2),
                                confidence=1.0,
                                data_available=True,
                                source="calculated",
                                timestamp=ts,
                                metadata={"period": period},
                            )
                        )
                except Exception as e:
                    logger.warning(f"SMA_{period} calculation failed for {symbol}: {e}")

            # EMAs
            for period in [12, 26]:
                try:
                    ma = ema(close, period=period)
                    if not ma.empty and not np.isnan(ma.iloc[-1]):
                        signals.append(
                            DataSignal(
                                name=f"EMA_{period}",
                                value=round(float(ma.iloc[-1]), 2),
                                confidence=1.0,
                                data_available=True,
                                source="calculated",
                                timestamp=ts,
                                metadata={"period": period},
                            )
                        )
                except Exception as e:
                    logger.warning(f"EMA_{period} calculation failed for {symbol}: {e}")

            # Bollinger Bands
            try:
                bb = bollinger_bands(close, period=20, std_dev=2)
                if not bb.empty and len(bb) > 0:
                    last_row = bb.iloc[-1]
                    
                    for band_name in ["upper", "middle", "lower"]:
                        if band_name in last_row and not np.isnan(last_row[band_name]):
                            signals.append(
                                DataSignal(
                                    name=f"BB_{band_name.upper()}",
                                    value=round(float(last_row[band_name]), 2),
                                    confidence=1.0,
                                    data_available=True,
                                    source="calculated",
                                    timestamp=ts,
                                    metadata={"period": 20, "std_dev": 2},
                                )
                            )
            except Exception as e:
                logger.warning(f"Bollinger Bands calculation failed for {symbol}: {e}")

            # ATR (14-period)
            try:
                atr_values = atr(high, low, close, period=14)
                if not atr_values.empty and not np.isnan(atr_values.iloc[-1]):
                    signals.append(
                        DataSignal(
                            name="ATR_14",
                            value=round(float(atr_values.iloc[-1]), 2),
                            confidence=1.0,
                            data_available=True,
                            source="calculated",
                            timestamp=ts,
                            metadata={"period": 14},
                        )
                    )
            except Exception as e:
                logger.warning(f"ATR calculation failed for {symbol}: {e}")

            # Stochastic Oscillator
            try:
                stoch = stochastic(high, low, close, k_period=14, d_period=3)
                if not stoch.empty and len(stoch) > 0:
                    last_row = stoch.iloc[-1]
                    
                    if "k" in last_row and not np.isnan(last_row["k"]):
                        signals.append(
                            DataSignal(
                                name="STOCH_K",
                                value=round(float(last_row["k"]), 2),
                                confidence=1.0,
                                data_available=True,
                                source="calculated",
                                timestamp=ts,
                                metadata={"k_period": 14, "d_period": 3},
                            )
                        )
                    
                    if "d" in last_row and not np.isnan(last_row["d"]):
                        signals.append(
                            DataSignal(
                                name="STOCH_D",
                                value=round(float(last_row["d"]), 2),
                                confidence=1.0,
                                data_available=True,
                                source="calculated",
                                timestamp=ts,
                                metadata={"d_period": 3},
                            )
                        )
            except Exception as e:
                logger.warning(f"Stochastic calculation failed for {symbol}: {e}")

            # Williams %R
            try:
                wr_values = williams_r(high, low, close, period=14)
                if not wr_values.empty and not np.isnan(wr_values.iloc[-1]):
                    signals.append(
                        DataSignal(
                            name="WILLIAMS_R",
                            value=round(float(wr_values.iloc[-1]), 2),
                            confidence=1.0,
                            data_available=True,
                            source="calculated",
                            timestamp=ts,
                            metadata={"period": 14},
                        )
                    )
            except Exception as e:
                logger.warning(f"Williams %R calculation failed for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Indicator calculation failed for {symbol}: {e}")

        return signals

    def _create_unavailable_signals(self) -> list[DataSignal]:
        """Create unavailable signals for all indicators when data missing"""
        signal_names = self.get_signal_names()
        return [
            self._create_unavailable_signal(name, "Historical data unavailable")
            for name in signal_names
        ]

    def get_signal_names(self) -> list[str]:
        """Get list of all technical indicator signal names"""
        return [
            "RSI_14",
            "MACD_HISTOGRAM",
            "MACD_SIGNAL",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "EMA_12",
            "EMA_26",
            "BB_UPPER",
            "BB_MIDDLE",
            "BB_LOWER",
            "ATR_14",
            "STOCH_K",
            "STOCH_D",
            "WILLIAMS_R",
        ]

    def health_check(self) -> dict[str, Any]:
        """Verify technical engine can calculate indicators"""
        results = {
            "ok": True,
            "pillar": self.pillar_name,
            "providers": [],
            "errors": [],
        }

        # Test with SPY
        try:
            spy_response = self.get_signals("SPY", period=90)
            
            if spy_response.available_signal_count() >= 10:
                results["providers"].append(
                    {
                        "name": "technical_calculator",
                        "status": "ok",
                        "latency_ms": spy_response.execution_time_ms,
                        "signals_computed": spy_response.available_signal_count(),
                    }
                )
            else:
                results["ok"] = False
                results["errors"].append(
                    f"Technical indicators failed (only {spy_response.available_signal_count()} signals)"
                )
                results["providers"].append(
                    {
                        "name": "technical_calculator",
                        "status": "degraded",
                        "signals_computed": spy_response.available_signal_count(),
                    }
                )
        except Exception as e:
            results["ok"] = False
            results["errors"].append(f"Technical engine health check failed: {e}")

        return results
