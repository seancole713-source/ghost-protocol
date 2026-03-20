"""
Core Technical Indicators Library
=================================
Pure numpy/pandas calculation functions for technical analysis.
Used by core.data_pillars.technical_engine.

Author: Ghost AI
"""

import numpy as np
import pandas as pd
from typing import Optional


def sma(series: pd.Series, period: int = 14) -> pd.Series:
    """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()


        def ema(series: pd.Series, period: int = 14) -> pd.Series:
            """Exponential Moving Average."""
                return series.ewm(span=period, adjust=False).mean()


                def rsi(series: pd.Series, period: int = 14) -> pd.Series:
                    """Relative Strength Index (0-100)."""
                        delta = series.diff()
                            gain = delta.where(delta > 0, 0.0)
                                loss = -delta.where(delta < 0, 0.0)
                                    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
                                        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
                                            rs = avg_gain / avg_loss.replace(0, np.nan)
                                                return 100.0 - (100.0 / (1.0 + rs))


                                                def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
                                                    """MACD indicator. Returns dict with 'macd', 'signal', 'histogram'."""
                                                        fast_ema = ema(series, fast)
                                                            slow_ema = ema(series, slow)
                                                                macd_line = fast_ema - slow_ema
                                                                    signal_line = ema(macd_line, signal)
                                                                        histogram = macd_line - signal_line
                                                                            return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


                                                                            def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
                                                                                """Bollinger Bands. Returns dict with 'upper', 'middle', 'lower', 'width'."""
                                                                                    middle = sma(series, period)
                                                                                        rolling_std = series.rolling(window=period, min_periods=1).std()
                                                                                            upper = middle + (rolling_std * std_dev)
                                                                                                lower = middle - (rolling_std * std_dev)
                                                                                                    width = (upper - lower) / middle.replace(0, np.nan)
                                                                                                        return {"upper": upper, "middle": middle, "lower": lower, "width": width}
                                                                                                        
                                                                                                        
                                                                                                        def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
                                                                                                            """Average True Range."""
                                                                                                                prev_close = close.shift(1)
                                                                                                                    tr1 = high - low
                                                                                                                        tr2 = (high - prev_close).abs()
                                                                                                                            tr3 = (low - prev_close).abs()
                                                                                                                                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                                                                                                                                    return true_range.rolling(window=period, min_periods=1).mean()
                                                                                                                                    
                                                                                                                                    
                                                                                                                                    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                                                                                                                                                   k_period: int = 14, d_period: int = 3) -> dict:
                                                                                                                                                       """Stochastic Oscillator. Returns dict with 'k' and 'd'."""
                                                                                                                                                           lowest_low = low.rolling(window=k_period, min_periods=1).min()
                                                                                                                                                               highest_high = high.rolling(window=k_period, min_periods=1).max()
                                                                                                                                                                   denom = highest_high - lowest_low
                                                                                                                                                                       k = 100.0 * (close - lowest_low) / denom.replace(0, np.nan)
                                                                                                                                                                           d = k.rolling(window=d_period, min_periods=1).mean()
                                                                                                                                                                               return {"k": k, "d": d}
                                                                                                                                                                               
                                                                                                                                                                               
                                                                                                                                                                               def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
                                                                                                                                                                                              period: int = 14) -> pd.Series:
                                                                                                                                                                                                  """Williams %R (-100 to 0)."""
                                                                                                                                                                                                      highest_high = high.rolling(window=period, min_periods=1).max()
                                                                                                                                                                                                          lowest_low = low.rolling(window=period, min_periods=1).min()
                                                                                                                                                                                                              denom = highest_high - lowest_low
                                                                                                                                                                                                                  return -100.0 * (highest_high - close) / denom.replace(0, np.nan)
