"""
GHOST Technical Indicators Library
50+ indicators for technical analysis - completely free!
Uses pandas and numpy only - no paid APIs or services.
"""

from typing import Any

import numpy as np
import pandas as pd

# ============================================================================
# TREND INDICATORS
# ============================================================================


def sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window=period).mean()


def ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return prices.ewm(span=period, adjust=False).mean()


def wma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return prices.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def dema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Double Exponential Moving Average."""
    ema1 = ema(prices, period)
    ema2 = ema(ema1, period)
    return 2 * ema1 - ema2


def tema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Triple Exponential Moving Average."""
    ema1 = ema(prices, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    return 3 * ema1 - 3 * ema2 + ema3


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence.
    Returns DataFrame with macd, signal, and histogram.
    """
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index.
    Returns DataFrame with ADX, +DI, and -DI.
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    up_move = high - high.shift()
    down_move = low.shift() - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_line = dx.rolling(window=period).mean()

    return pd.DataFrame({"adx": adx_line, "plus_di": plus_di, "minus_di": minus_di})


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """
    Aroon Indicator.
    Returns DataFrame with aroon_up, aroon_down, and aroon_oscillator.
    """
    aroon_up = high.rolling(window=period + 1).apply(
        lambda x: float(np.argmax(x)) / period * 100, raw=True
    )
    aroon_down = low.rolling(window=period + 1).apply(
        lambda x: float(np.argmin(x)) / period * 100, raw=True
    )
    aroon_osc = aroon_up - aroon_down

    return pd.DataFrame(
        {"aroon_up": aroon_up, "aroon_down": aroon_down, "aroon_oscillator": aroon_osc}
    )


# ============================================================================
# MOMENTUM INDICATORS
# ============================================================================


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # type: ignore[operator]
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # type: ignore[operator]
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """
    Stochastic Oscillator.
    Returns DataFrame with %K and %D.
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period).mean()

    return pd.DataFrame({"k": k, "d": d})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def roc(prices: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change."""
    return 100 * (prices - prices.shift(period)) / prices.shift(period)


def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Momentum indicator."""
    return prices - prices.shift(period)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)


def ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
) -> pd.Series:
    """Ultimate Oscillator."""
    bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(
        axis=1
    )

    avg1 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg2 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg3 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()

    return 100 * (4 * avg1 + 2 * avg2 + avg3) / 7


# ============================================================================
# VOLATILITY INDICATORS
# ============================================================================


def bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with upper, middle, and lower bands.
    """
    middle = sma(prices, period)
    std = prices.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return pd.DataFrame(
        {
            "upper": upper,
            "middle": middle,
            "lower": lower,
            "bandwidth": (upper - lower) / middle,
            "%b": (prices - lower) / (upper - lower),
        }
    )


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def keltner_channels(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20, multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Keltner Channels.
    Returns DataFrame with upper, middle, and lower channels.
    """
    middle = ema(close, period)
    atr_val = atr(high, low, close, period)
    upper = middle + (multiplier * atr_val)
    lower = middle - (multiplier * atr_val)

    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})


def donchian_channels(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    """
    Donchian Channels.
    Returns DataFrame with upper, middle, and lower channels.
    """
    upper = high.rolling(window=period).max()
    lower = low.rolling(window=period).min()
    middle = (upper + lower) / 2

    return pd.DataFrame({"upper": upper, "middle": middle, "lower": lower})


def historical_volatility(
    prices: pd.Series, period: int = 20, trading_days: int = 252
) -> pd.Series:
    """Historical Volatility (annualized)."""
    log_returns = np.log(prices / prices.shift())
    # Ensure we're working with Series (not ndarray)
    if not isinstance(log_returns, pd.Series):
        log_returns = pd.Series(log_returns)
    return log_returns.rolling(window=period).std() * np.sqrt(trading_days)


# ============================================================================
# VOLUME INDICATORS
# ============================================================================


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff())
    direction[direction == 0] = 1  # No change = accumulation
    result = (direction * volume).cumsum()
    return pd.Series(result) if isinstance(result, np.ndarray) else result


def ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Accumulation/Distribution Line."""
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0)
    return (clv * volume).cumsum()


def cmf(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """Chaikin Money Flow."""
    mfv = ((close - low) - (high - close)) / (high - low) * volume
    mfv = mfv.fillna(0)
    return mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3
    mf = tp * volume

    positive_mf = mf.where(tp > tp.shift(), 0).rolling(window=period).sum()
    negative_mf = mf.where(tp < tp.shift(), 0).rolling(window=period).sum()

    mfr = positive_mf / negative_mf
    return 100 - (100 / (1 + mfr))


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price."""
    tp = (high + low + close) / 3
    return (tp * volume).cumsum() / volume.cumsum()


def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Force Index."""
    fi = close.diff() * volume
    return ema(fi, period)


def ease_of_movement(
    high: pd.Series, low: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """Ease of Movement."""
    distance = (high + low) / 2 - (high.shift() + low.shift()) / 2
    box_ratio = (volume / 1000000) / (high - low)
    emv = distance / box_ratio
    return emv.rolling(window=period).mean()


# ============================================================================
# SUPPORT/RESISTANCE INDICATORS
# ============================================================================


def pivot_points(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """
    Pivot Points (Standard).
    Returns DataFrame with pivot, resistance levels (r1, r2, r3), and support levels (s1, s2, s3).
    """
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)

    return pd.DataFrame(
        {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}
    )


def fibonacci_retracement(high_price: float, low_price: float) -> dict:
    """
    Fibonacci Retracement Levels.
    Returns dict with retracement levels.
    """
    diff = high_price - low_price
    return {
        "0.0": high_price,
        "0.236": high_price - 0.236 * diff,
        "0.382": high_price - 0.382 * diff,
        "0.500": high_price - 0.500 * diff,
        "0.618": high_price - 0.618 * diff,
        "0.786": high_price - 0.786 * diff,
        "1.0": low_price,
    }


def support_resistance(
    prices: pd.Series, window: int = 20, threshold: float = 0.02
) -> pd.DataFrame:
    """
    Find support and resistance levels using local extrema.
    Returns DataFrame with support and resistance levels.
    """
    # Find local maxima (resistance)
    resistance = prices.rolling(window=window, center=True).apply(
        lambda x: x[window // 2] if x[window // 2] == x.max() else np.nan, raw=True
    )

    # Find local minima (support)
    support = prices.rolling(window=window, center=True).apply(
        lambda x: x[window // 2] if x[window // 2] == x.min() else np.nan, raw=True
    )

    return pd.DataFrame({"support": support, "resistance": resistance})


# ============================================================================
# PATTERN RECOGNITION
# ============================================================================


def detect_trend(prices: pd.Series, short_period: int = 20, long_period: int = 50) -> pd.Series:
    """
    Detect trend direction.
    Returns 1 (uptrend), -1 (downtrend), or 0 (sideways).
    """
    short_ma = sma(prices, short_period)
    long_ma = sma(prices, long_period)

    trend = pd.Series(0, index=prices.index)
    trend[short_ma > long_ma] = 1
    trend[short_ma < long_ma] = -1

    return trend


def detect_golden_cross(
    prices: pd.Series, short_period: int = 50, long_period: int = 200
) -> pd.Series:
    """
    Detect Golden Cross (bullish) and Death Cross (bearish).
    Returns 1 (golden cross), -1 (death cross), or 0 (none).
    """
    short_ma = sma(prices, short_period)
    long_ma = sma(prices, long_period)

    crossover = pd.Series(0, index=prices.index)

    # Golden cross: short MA crosses above long MA
    golden = (short_ma > long_ma) & (short_ma.shift() <= long_ma.shift())
    crossover[golden] = 1

    # Death cross: short MA crosses below long MA
    death = (short_ma < long_ma) & (short_ma.shift() >= long_ma.shift())
    crossover[death] = -1

    return crossover


def detect_divergence(prices: pd.Series, indicator: pd.Series, window: int = 5) -> pd.Series:
    """
    Detect bullish/bearish divergence between price and indicator.
    Returns 1 (bullish divergence), -1 (bearish divergence), or 0 (none).
    """
    price_trend = prices - prices.shift(window)
    indicator_trend = indicator - indicator.shift(window)

    divergence = pd.Series(0, index=prices.index)

    # Bullish divergence: price down, indicator up
    divergence[(price_trend < 0) & (indicator_trend > 0)] = 1

    # Bearish divergence: price up, indicator down
    divergence[(price_trend > 0) & (indicator_trend < 0)] = -1

    return divergence


# ============================================================================
# COMPOSITE INDICATORS
# ============================================================================


def ichimoku_cloud(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    conversion_period: int = 9,
    base_period: int = 26,
    span_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Ichimoku Cloud.
    Returns DataFrame with all Ichimoku components.
    """
    # Conversion Line (Tenkan-sen)
    conv_high = high.rolling(window=conversion_period).max()
    conv_low = low.rolling(window=conversion_period).min()
    conversion_line = (conv_high + conv_low) / 2

    # Base Line (Kijun-sen)
    base_high = high.rolling(window=base_period).max()
    base_low = low.rolling(window=base_period).min()
    base_line = (base_high + base_low) / 2

    # Leading Span A (Senkou Span A)
    span_a = ((conversion_line + base_line) / 2).shift(displacement)

    # Leading Span B (Senkou Span B)
    span_b_high = high.rolling(window=span_b_period).max()
    span_b_low = low.rolling(window=span_b_period).min()
    span_b = ((span_b_high + span_b_low) / 2).shift(displacement)

    # Lagging Span (Chikou Span)
    lagging_span = close.shift(-displacement)

    return pd.DataFrame(
        {
            "conversion_line": conversion_line,
            "base_line": base_line,
            "span_a": span_a,
            "span_b": span_b,
            "lagging_span": lagging_span,
        }
    )


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """
    SuperTrend indicator.
    Returns DataFrame with supertrend line and trend direction.
    """
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + (multiplier * atr_val)
    lower_band = hl2 - (multiplier * atr_val)

    supertrend = pd.Series(index=close.index, dtype=float)
    direction = pd.Series(1, index=close.index)  # 1 = uptrend, -1 = downtrend

    for i in range(period, len(close)):
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i - 1]

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]

    return pd.DataFrame({"supertrend": supertrend, "direction": direction})


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def calculate_all_indicators(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    Calculate all major indicators for a given OHLCV DataFrame.
    Returns DataFrame with all indicators added as columns.
    """
    result = df.copy()

    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    volume = df[volume_col] if volume_col in df.columns else None

    # Trend indicators
    result["sma_20"] = sma(close, 20)
    result["sma_50"] = sma(close, 50)
    result["sma_200"] = sma(close, 200)
    result["ema_12"] = ema(close, 12)
    result["ema_26"] = ema(close, 26)

    macd_data = macd(close)
    result["macd"] = macd_data["macd"]
    result["macd_signal"] = macd_data["signal"]
    result["macd_histogram"] = macd_data["histogram"]

    adx_data = adx(high, low, close)
    result["adx"] = adx_data["adx"]
    result["plus_di"] = adx_data["plus_di"]
    result["minus_di"] = adx_data["minus_di"]

    # Momentum indicators
    result["rsi_14"] = rsi(close, 14)

    stoch_data = stochastic(high, low, close)
    result["stoch_k"] = stoch_data["k"]
    result["stoch_d"] = stoch_data["d"]

    result["williams_r"] = williams_r(high, low, close)
    result["roc_12"] = roc(close, 12)
    result["cci_20"] = cci(high, low, close, 20)

    # Volatility indicators
    bb_data = bollinger_bands(close)
    result["bb_upper"] = bb_data["upper"]
    result["bb_middle"] = bb_data["middle"]
    result["bb_lower"] = bb_data["lower"]
    result["bb_bandwidth"] = bb_data["bandwidth"]

    result["atr_14"] = atr(high, low, close, 14)
    result["historical_vol"] = historical_volatility(close)

    # Volume indicators (if volume data available)
    if volume is not None:
        result["obv"] = obv(close, volume)
        result["ad_line"] = ad_line(high, low, close, volume)
        result["cmf"] = cmf(high, low, close, volume)
        result["mfi"] = mfi(high, low, close, volume)
        result["vwap"] = vwap(high, low, close, volume)

    # Pattern detection
    result["trend"] = detect_trend(close)
    result["golden_cross"] = detect_golden_cross(close)

    return result


def get_indicator_summary(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a summary of current indicator signals.
    Returns dict with signal strength for buy/sell/hold.
    """
    latest = df.iloc[-1]

    signals: dict[str, Any] = {"buy": 0, "sell": 0, "neutral": 0}

    # RSI signals
    if "rsi_14" in latest:
        if latest["rsi_14"] < 30:
            signals["buy"] += 1
        elif latest["rsi_14"] > 70:
            signals["sell"] += 1
        else:
            signals["neutral"] += 1

    # MACD signals
    if "macd" in latest and "macd_signal" in latest:
        if latest["macd"] > latest["macd_signal"]:
            signals["buy"] += 1
        else:
            signals["sell"] += 1

    # Bollinger Bands signals
    if "bb_lower" in latest and "bb_upper" in latest:
        close = latest.get("close", latest.get("Close", 0))
        if close < latest["bb_lower"]:
            signals["buy"] += 1
        elif close > latest["bb_upper"]:
            signals["sell"] += 1
        else:
            signals["neutral"] += 1

    # ADX trend strength
    if "adx" in latest:
        if latest["adx"] > 25:
            signals["trend_strong"] = True
        else:
            signals["trend_weak"] = True

    # Calculate overall signal
    total = float(signals["buy"] + signals["sell"] + signals["neutral"])
    if total > 0:
        signals["buy_pct"] = float(signals["buy"]) / total
        signals["sell_pct"] = float(signals["sell"]) / total
        signals["neutral_pct"] = float(signals["neutral"]) / total

    # Overall recommendation
    if signals["buy"] > signals["sell"] and signals["buy"] > signals["neutral"]:
        signals["recommendation"] = "BUY"
    elif signals["sell"] > signals["buy"] and signals["sell"] > signals["neutral"]:
        signals["recommendation"] = "SELL"
    else:
        signals["recommendation"] = "HOLD"

    return signals


# ============================================================================
# INDICATOR LIST
# ============================================================================

AVAILABLE_INDICATORS = {
    "trend": ["sma", "ema", "wma", "dema", "tema", "macd", "adx", "aroon"],
    "momentum": [
        "rsi",
        "stochastic",
        "williams_r",
        "roc",
        "momentum",
        "cci",
        "ultimate_oscillator",
    ],
    "volatility": [
        "bollinger_bands",
        "atr",
        "keltner_channels",
        "donchian_channels",
        "historical_volatility",
    ],
    "volume": ["obv", "ad_line", "cmf", "mfi", "vwap", "force_index", "ease_of_movement"],
    "support_resistance": ["pivot_points", "fibonacci_retracement", "support_resistance"],
    "pattern": ["detect_trend", "detect_golden_cross", "detect_divergence"],
    "composite": ["ichimoku_cloud", "supertrend"],
}


def list_indicators() -> dict:
    """Return dictionary of all available indicators by category."""
    return AVAILABLE_INDICATORS
