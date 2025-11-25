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
        """
        Fetch historical price/volume data with fallback providers.
        
        Shares fallback strategy with Technical Engine for consistency.
        """
        # Try yfinance first
        df = self._fetch_yfinance(symbol, days)
        if df is not None and len(df) >= 20:
            return df
        
        logger.warning(f"yfinance failed for {symbol}, trying fallbacks...")
        
        # PRIMARY: Polygon for stocks
        if not self._is_crypto_symbol(symbol):
            df = self._fetch_polygon_historical(symbol, days)
            if df is not None and len(df) >= 20:
                logger.info(f"[VOL] {symbol}: Polygon returned {len(df)} bars")
                return df
            logger.warning(f"[VOL] {symbol}: Polygon failed, trying Yahoo")
        
        # SECONDARY: Yahoo Finance / yfinance
        df = self._fetch_yfinance(symbol, days)
        if df is not None and len(df) >= 20:
            logger.info(f"[VOL] {symbol}: Yahoo/yfinance returned {len(df)} bars")
            return df
        
        # TERTIARY: Crypto providers
        if self._is_crypto_symbol(symbol):
            logger.warning(f"[VOL] {symbol}: Yahoo failed, trying CoinGecko")
            df = self._fetch_crypto_historical(symbol, days)
            if df is not None and len(df) >= 20:
                logger.info(f"[VOL] {symbol}: CoinGecko returned {len(df)} bars")
                return df
        
        logger.error(f"[VOL] {symbol}: ALL PROVIDERS FAILED")
        return None

    def _fetch_yfinance(self, symbol: str, days: int) -> pd.DataFrame | None:
        """Fetch using yfinance"""
        try:
            import yfinance as yf
            from datetime import datetime, timedelta
            
            # Crypto symbols need -USD suffix
            yf_symbol = symbol
            if self._is_crypto_symbol(symbol):
                yf_symbol = f"{symbol}-USD"
            
            ticker = yf.Ticker(yf_symbol)
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
                df["timestamp"] = pd.to_datetime(df["timestamp"]).astype(int) // 10**9
            
            if "close" not in df.columns:
                df["close"] = 0
            if "volume" not in df.columns:
                df["volume"] = 0

            return df

        except Exception as e:
            logger.warning(f"yfinance fetch failed for {symbol}: {e}")
            return None

    def _fetch_polygon_historical(self, symbol: str, days: int) -> pd.DataFrame | None:
        """Fetch from Polygon (stocks only)"""
        try:
            import os
            import requests
            from datetime import datetime, timedelta
            
            api_key = os.getenv("POLYGON_API_KEY")
            if not api_key:
                return None
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"adjusted": "true", "sort": "asc", "limit": 500, "apiKey": api_key}
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data.get("results"):
                return None
            
            df = pd.DataFrame(data["results"])
            df = df.rename(columns={
                "t": "timestamp",
                "c": "close",
                "v": "volume"
            })
            
            df["timestamp"] = df["timestamp"] // 1000
            
            return df
            
        except Exception as e:
            logger.warning(f"Polygon fetch failed for {symbol}: {e}")
            return None

    def _fetch_crypto_historical(self, symbol: str, days: int) -> pd.DataFrame | None:
        """Fetch crypto data from CoinGecko"""
        try:
            import requests
            
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
            
            df = pd.DataFrame(prices, columns=["timestamp", "close"])
            df["timestamp"] = df["timestamp"] // 1000
            
            if volumes:
                vol_df = pd.DataFrame(volumes, columns=["timestamp", "volume"])
                vol_df["timestamp"] = vol_df["timestamp"] // 1000
                df = df.merge(vol_df, on="timestamp", how="left")
            
            if "volume" not in df.columns:
                df["volume"] = 0
            
            return df
            
        except Exception as e:
            logger.warning(f"CoinGecko fetch failed for {symbol}: {e}")
            return None

    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Detect if symbol is cryptocurrency"""
        CRYPTO_SYMBOLS = {
            "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX",
            "DOT", "MATIC", "LINK", "UNI", "AAVE", "MKR", "CRV"
        }
        return symbol.upper() in CRYPTO_SYMBOLS

    def _calculate_volume_signals(
        self, df: pd.DataFrame, symbol: str
    ) -> list[DataSignal]:
        """
        Calculate volume and volatility signals.
        
        Each calculation wrapped in try/except for resilience.
        """
        signals = []
        ts = time.time()

        if len(df) < 20:
            logger.warning(f"Insufficient bars for {symbol}: {len(df)} (need 20+)")
            return []

        try:
            close = df["close"]
            volume = df["volume"]

            # Volume spike detection (current vs 20-day average)
            if len(volume) >= 20:
                try:
                    vol_ma_20 = volume.rolling(window=20).mean()
                    current_vol = volume.iloc[-1]
                    avg_vol = vol_ma_20.iloc[-1]

                    if avg_vol > 0 and not np.isnan(avg_vol) and not np.isnan(current_vol):
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
                except Exception as e:
                    logger.warning(f"VOLUME_SPIKE calculation failed for {symbol}: {e}")

            # 20-day realized volatility
            if len(close) >= 20:
                try:
                    returns = close.pct_change().dropna()
                    if len(returns) >= 20:
                        vol_20d = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized %
                        if not np.isnan(vol_20d):
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
                except Exception as e:
                    logger.warning(f"VOLATILITY_20D calculation failed for {symbol}: {e}")

            # 60-day realized volatility
            if len(close) >= 60:
                try:
                    returns = close.pct_change().dropna()
                    if len(returns) >= 60:
                        vol_60d = returns.tail(60).std() * np.sqrt(252) * 100
                        if not np.isnan(vol_60d):
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
                except Exception as e:
                    logger.warning(f"VOLATILITY_60D calculation failed for {symbol}: {e}")

            # Volume moving average
            if len(volume) >= 20:
                try:
                    vol_ma = volume.rolling(window=20).mean().iloc[-1]
                    if not np.isnan(vol_ma):
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
                except Exception as e:
                    logger.warning(f"VOLUME_MA_20 calculation failed for {symbol}: {e}")

            # Volume rate of change (10-day)
            if len(volume) >= 10:
                try:
                    vol_10_ago = volume.iloc[-10]
                    vol_current = volume.iloc[-1]
                    if vol_10_ago > 0 and not np.isnan(vol_10_ago) and not np.isnan(vol_current):
                        vol_roc = (vol_current / vol_10_ago - 1.0) * 100
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
                    logger.warning(f"VOLUME_ROC calculation failed for {symbol}: {e}")

        except Exception as e:
            logger.error(f"Volume signal calculation failed for {symbol}: {e}")

        if not signals:
            logger.warning(f"No volume signals calculated for {symbol}")

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
