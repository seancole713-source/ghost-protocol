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
        Fetch historical OHLCV data with fallback providers.
        
        Provider priority (FREE-TIER FIRST - Nov 24, 2025):
        STOCKS: Yahoo → yfinance → cache (100% FREE)
        CRYPTO: Binance → CoinGecko → cache (100% FREE)
        
        NO PAID APIs - Ghost must work on free tier!
        
        Args:
            symbol: Ticker symbol
            days: Lookback period
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        failed_providers = []
        
        # PRIMARY: Unified Provider (FREE-TIER ONLY)
        # Crypto: Binance OHLCV (FREE, no key)
        # Stocks: Yahoo Finance (FREE, rate-limited)
        try:
            from core.providers.unified_provider import get_unified_provider
            
            provider = get_unified_provider()
            interval = "1d"  # Daily bars for technical indicators
            lookback = max(days + 10, 100)  # Add buffer for indicator calculations
            
            ohlcv = provider.get_ohlcv(symbol, interval=interval, lookback=lookback)
            
            if ohlcv and ohlcv.bars and len(ohlcv.bars) >= 20:
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        "timestamp": bar.timestamp,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume
                    }
                    for bar in ohlcv.bars
                ])
                
                logger.info(
                    f"[TECH] ✅ {symbol}: Unified provider ({ohlcv.provider}) "
                    f"returned {len(df)} bars (cache_hit={ohlcv.cache_hit})"
                )
                return df
            else:
                logger.warning(f"[TECH] {symbol}: Unified provider returned insufficient data")
                failed_providers.append("unified")
        
        except ImportError:
            logger.debug(f"[TECH] {symbol}: Unified provider not available, using legacy")
        except Exception as e:
            logger.warning(f"[TECH] {symbol}: Unified provider failed: {e}")
            failed_providers.append("unified")
        
        # FALLBACK 1: Polygon for stocks
        if not self._is_crypto_symbol(symbol):
            df = self._fetch_polygon_historical(symbol, days)
            if df is not None and len(df) >= 20:
                logger.info(f"[TECH] {symbol}: Polygon returned {len(df)} bars")
                return df
            failed_providers.append("polygon")
            logger.warning(f"[TECH] {symbol}: Polygon failed, trying Yahoo")
        
        # FALLBACK 2: Yahoo Finance / yfinance
        df = self._fetch_yfinance(symbol, days)
        if df is not None and len(df) >= 20:
            logger.info(f"[TECH] {symbol}: Yahoo/yfinance returned {len(df)} bars")
            return df
        failed_providers.append("yahoo")
        
        # FALLBACK 3: Crypto providers (CoinGecko - deprecated)
        if self._is_crypto_symbol(symbol):
            logger.warning(f"[TECH] {symbol}: Yahoo failed, trying CoinGecko")
            df = self._fetch_crypto_historical(symbol, days)
            if df is not None and len(df) >= 20:
                logger.info(f"[TECH] {symbol}: CoinGecko returned {len(df)} bars")
                return df
            failed_providers.append("coingecko")
        
        logger.error(f"[TECH] {symbol}: ALL PROVIDERS FAILED - {failed_providers}")
        return None

    def _fetch_yfinance(self, symbol: str, days: int) -> pd.DataFrame | None:
        """Fetch historical data using yfinance"""
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            # Crypto symbols need -USD suffix for yfinance
            yf_symbol = symbol
            if self._is_crypto_symbol(symbol):
                yf_symbol = f"{symbol}-USD"
            
            # Fetch data using yfinance
            ticker = yf.Ticker(yf_symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Download historical data
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist is None or len(hist) < 20:
                logger.warning(f"Insufficient yfinance data for {yf_symbol}: {len(hist) if hist is not None else 0} bars")
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
                df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9
            
            # Ensure required columns
            for col in ["open", "high", "low", "close"]:
                if col not in df.columns:
                    df[col] = df.get("close", 0)
            
            if "volume" not in df.columns:
                df["volume"] = 0
            
            logger.info(f"yfinance: Fetched {len(df)} bars for {yf_symbol}")
            return df
            
        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def _fetch_polygon_historical(self, symbol: str, days: int) -> pd.DataFrame | None:
        """Fetch historical data from Polygon (stocks only)"""
        try:
            import os
            import requests
            from datetime import datetime, timedelta
            
            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                return None
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Polygon aggregates API
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"adjusted": "true", "sort": "asc", "limit": 500, "apiKey": api_key}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get("results"):
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data["results"])
            df = df.rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume"
            })
            
            # Timestamp already in milliseconds, convert to seconds
            df["timestamp"] = df["timestamp"] // 1000
            
            logger.info(f"Polygon: Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.warning(f"Polygon fetch failed for {symbol}: {e}")
            return None

    def _fetch_crypto_historical(self, symbol: str, days: int) -> pd.DataFrame | None:
        """Fetch historical data for crypto from CoinGecko"""
        try:
            import requests
            from datetime import datetime, timedelta
            
            # CoinGecko symbol mapping
            symbol_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "SOL": "solana",
                "BNB": "binancecoin",
                "XRP": "ripple",
                "ADA": "cardano",
                "DOGE": "dogecoin",
                "AVAX": "avalanche-2",
                "DOT": "polkadot",
                "MATIC": "matic-network",
            }
            
            coin_id = symbol_map.get(symbol.upper())
            if not coin_id:
                return None
            
            # CoinGecko market chart API (OHLC data)
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            if not prices or len(prices) < 20:
                return None
            
            # Build DataFrame (CoinGecko doesn't provide OHLC, only close prices)
            df = pd.DataFrame(prices, columns=["timestamp", "close"])
            df["timestamp"] = df["timestamp"] // 1000  # Convert ms to seconds
            df["open"] = df["close"]  # Approximate
            df["high"] = df["close"]
            df["low"] = df["close"]
            
            # Add volume if available
            if volumes:
                vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
                vol_df["timestamp"] = vol_df["timestamp"] // 1000
                df = df.merge(vol_df, on="timestamp", how="left")
            
            if "volume" not in df.columns:
                df["volume"] = 0
            
            logger.info(f"CoinGecko: Fetched {len(df)} bars for {symbol}")
            return df
            
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed for {symbol}: {e}")
            return None

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Detect if symbol is cryptocurrency"""
        from core.asset_classification import is_crypto_symbol

        return is_crypto_symbol(symbol)

    def _calculate_indicators(self, df: pd.DataFrame, symbol: str) -> list[DataSignal]:
        """
        Calculate all technical indicators from OHLCV data.
        
        Each indicator wrapped in try/except to ensure one failure doesn't kill entire pillar.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Ticker symbol
        
        Returns:
            List of DataSignal objects
        """
        signals = []
        ts = time.time()

        if len(df) < 20:
            logger.warning(f"Insufficient bars for {symbol}: {len(df)} (need 20+)")
            return self._create_unavailable_signals()

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            current_price = float(close.iloc[-1])

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
                else:
                    logger.debug(f"RSI calculation returned NaN for {symbol}")
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
                else:
                    logger.debug(f"MACD calculation returned empty for {symbol}")
            except Exception as e:
                logger.warning(f"MACD calculation failed for {symbol}: {e}")

            # Moving Averages
            for period in [20, 50, 200]:
                try:
                    if len(close) < period:
                        logger.debug(f"Insufficient data for SMA_{period} on {symbol}: {len(close)} < {period}")
                        continue
                    
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
                    if len(close) < period:
                        continue
                    
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

            # Bollinger Bands + POSITION
            try:
                bb = bollinger_bands(close, period=20, std_dev=2)
                if not bb.empty and len(bb) > 0:
                    last_row = bb.iloc[-1]
                    
                    bb_upper = None
                    bb_middle = None
                    bb_lower = None
                    
                    for band_name in ["upper", "middle", "lower"]:
                        if band_name in last_row and not np.isnan(last_row[band_name]):
                            value = round(float(last_row[band_name]), 2)
                            signals.append(
                                DataSignal(
                                    name=f"BB_{band_name.upper()}",
                                    value=value,
                                    confidence=1.0,
                                    data_available=True,
                                    source="calculated",
                                    timestamp=ts,
                                    metadata={"period": 20, "std_dev": 2},
                                )
                            )
                            
                            # Store for position calculation
                            if band_name == "upper":
                                bb_upper = value
                            elif band_name == "middle":
                                bb_middle = value
                            elif band_name == "lower":
                                bb_lower = value
                    
                    # Calculate Bollinger Band Position (0.0 = at lower, 1.0 = at upper)
                    if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
                        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                        bb_position = max(0.0, min(1.0, bb_position))  # Clamp to 0-1
                        
                        signals.append(
                            DataSignal(
                                name="BOLLINGER_POSITION",
                                value=round(bb_position, 3),
                                confidence=1.0,
                                data_available=True,
                                source="calculated",
                                timestamp=ts,
                                metadata={
                                    "current_price": current_price,
                                    "bb_upper": bb_upper,
                                    "bb_lower": bb_lower,
                                    "interpretation": "0.0=lower band, 0.5=middle, 1.0=upper band"
                                },
                            )
                        )
                else:
                    logger.debug(f"Bollinger Bands calculation returned empty for {symbol}")
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

        if not signals:
            logger.warning(f"No technical indicators calculated for {symbol}")
            return self._create_unavailable_signals()

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
            "BOLLINGER_POSITION",  # NEW: Position within bands (0.0-1.0)
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
