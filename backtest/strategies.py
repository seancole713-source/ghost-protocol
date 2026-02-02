"""
Simple Trading Strategies for Backtesting
Each strategy takes historical data and returns 'UP', 'DOWN', or 'FLAT'
"""

import pandas as pd
import numpy as np
from typing import Optional


# =============================================================================
# STRATEGY 1: MOMENTUM (Trend Following)
# =============================================================================

def momentum_strategy(df: pd.DataFrame, lookback_hours: int = 24) -> str:
    """
    Trend following: If price went UP, predict it continues UP.
    
    Logic: Trends tend to persist in the short term.
    
    Args:
        df: Historical OHLCV data (strategy only sees this, no future)
        lookback_hours: How far back to measure trend
        
    Returns:
        'UP', 'DOWN', or 'FLAT'
    """
    if len(df) < lookback_hours + 1:
        return 'FLAT'
    
    current_price = df['Close'].iloc[-1]
    past_price = df['Close'].iloc[-lookback_hours]
    
    recent_return = (current_price - past_price) / past_price
    
    if recent_return > 0.01:  # +1% threshold
        return 'UP'
    elif recent_return < -0.01:  # -1% threshold
        return 'DOWN'
    
    return 'FLAT'


def momentum_strong_strategy(df: pd.DataFrame, lookback_hours: int = 24) -> str:
    """
    Strong momentum: Only trade when move is 3%+
    """
    if len(df) < lookback_hours + 1:
        return 'FLAT'
    
    current_price = df['Close'].iloc[-1]
    past_price = df['Close'].iloc[-lookback_hours]
    
    recent_return = (current_price - past_price) / past_price
    
    if recent_return > 0.03:  # +3% threshold
        return 'UP'
    elif recent_return < -0.03:  # -3% threshold
        return 'DOWN'
    
    return 'FLAT'


# =============================================================================
# STRATEGY 2: MEAN REVERSION (Buy Dips)
# =============================================================================

def mean_reversion_strategy(df: pd.DataFrame, lookback_hours: int = 24) -> str:
    """
    Mean reversion: If price went DOWN, predict bounce back UP.
    
    Logic: After big moves, price tends to revert to mean.
    Opposite of momentum.
    
    Args:
        df: Historical OHLCV data
        lookback_hours: How far back to measure move
        
    Returns:
        'UP', 'DOWN', or 'FLAT'
    """
    if len(df) < lookback_hours + 1:
        return 'FLAT'
    
    current_price = df['Close'].iloc[-1]
    past_price = df['Close'].iloc[-lookback_hours]
    
    recent_return = (current_price - past_price) / past_price
    
    if recent_return < -0.02:  # Down 2%+ → expect bounce
        return 'UP'
    elif recent_return > 0.02:  # Up 2%+ → expect pullback
        return 'DOWN'
    
    return 'FLAT'


def mean_reversion_aggressive_strategy(df: pd.DataFrame, lookback_hours: int = 24) -> str:
    """
    Aggressive mean reversion: Trade smaller moves (1%)
    """
    if len(df) < lookback_hours + 1:
        return 'FLAT'
    
    current_price = df['Close'].iloc[-1]
    past_price = df['Close'].iloc[-lookback_hours]
    
    recent_return = (current_price - past_price) / past_price
    
    if recent_return < -0.01:  # Down 1%+ → expect bounce
        return 'UP'
    elif recent_return > 0.01:  # Up 1%+ → expect pullback
        return 'DOWN'
    
    return 'FLAT'


# =============================================================================
# STRATEGY 3: GHOST INVERSE (Simulates V3 Logic)
# =============================================================================

def ghost_inverse_strategy(df: pd.DataFrame, lookback_hours: int = 24) -> str:
    """
    Simulate Ghost predicting DOWN, then inverting to UP.
    
    Logic: Ghost historically got DOWN predictions wrong on major crypto.
    If momentum suggests DOWN → we predict UP (inverse).
    
    This tests if the inverse logic has historical edge.
    
    Args:
        df: Historical OHLCV data
        lookback_hours: How far back to measure
        
    Returns:
        'UP', 'DOWN', or 'FLAT'
    """
    if len(df) < lookback_hours + 1:
        return 'FLAT'
    
    current_price = df['Close'].iloc[-1]
    past_price = df['Close'].iloc[-lookback_hours]
    
    recent_return = (current_price - past_price) / past_price
    
    # If downtrend (Ghost would predict DOWN), we inverse to UP
    if recent_return < -0.01:  # Downtrend detected
        return 'UP'  # Inverse: predict UP
    
    # Don't trade uptrends (Ghost wouldn't inverse those)
    return 'FLAT'


def ghost_inverse_strong_strategy(df: pd.DataFrame, lookback_hours: int = 48) -> str:
    """
    Ghost inverse on stronger downtrends (3%+, 48hr lookback)
    """
    if len(df) < lookback_hours + 1:
        return 'FLAT'
    
    current_price = df['Close'].iloc[-1]
    past_price = df['Close'].iloc[-lookback_hours]
    
    recent_return = (current_price - past_price) / past_price
    
    # Only inverse on strong downtrends
    if recent_return < -0.03:  # Down 3%+
        return 'UP'  # Inverse
    
    return 'FLAT'


# =============================================================================
# STRATEGY 4: RSI (Relative Strength Index)
# =============================================================================

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def rsi_strategy(df: pd.DataFrame, period: int = 14) -> str:
    """
    RSI oversold/overbought strategy.
    
    Logic:
    - RSI < 30 → Oversold → predict UP (bounce)
    - RSI > 70 → Overbought → predict DOWN (correction)
    
    Args:
        df: Historical OHLCV data
        period: RSI calculation period
        
    Returns:
        'UP', 'DOWN', or 'FLAT'
    """
    if len(df) < period + 10:
        return 'FLAT'
    
    rsi = _calculate_rsi(df['Close'], period)
    current_rsi = rsi.iloc[-1]
    
    if pd.isna(current_rsi):
        return 'FLAT'
    
    if current_rsi < 30:
        return 'UP'
    elif current_rsi > 70:
        return 'DOWN'
    
    return 'FLAT'


def rsi_extreme_strategy(df: pd.DataFrame, period: int = 14) -> str:
    """
    RSI with extreme thresholds (20/80 instead of 30/70)
    """
    if len(df) < period + 10:
        return 'FLAT'
    
    rsi = _calculate_rsi(df['Close'], period)
    current_rsi = rsi.iloc[-1]
    
    if pd.isna(current_rsi):
        return 'FLAT'
    
    if current_rsi < 20:
        return 'UP'
    elif current_rsi > 80:
        return 'DOWN'
    
    return 'FLAT'


# =============================================================================
# STRATEGY 5: VOLUME SPIKE
# =============================================================================

def volume_spike_strategy(df: pd.DataFrame) -> str:
    """
    Volume spike detection: High volume + direction = trend continues.
    
    Logic: Unusual volume often confirms price direction.
    
    Args:
        df: Historical OHLCV data
        
    Returns:
        'UP', 'DOWN', or 'FLAT'
    """
    if len(df) < 48:
        return 'FLAT'
    
    # Calculate average volume (last 24 hours)
    avg_volume = df['Volume'].iloc[-24:].mean()
    current_volume = df['Volume'].iloc[-1]
    
    # Recent price change (last 4 hours)
    price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-4]) / df['Close'].iloc[-4]
    
    # Check for volume spike (2x average)
    if current_volume > avg_volume * 2:
        if price_change > 0.005:  # Up 0.5%+ with volume
            return 'UP'
        elif price_change < -0.005:  # Down 0.5%+ with volume
            return 'DOWN'
    
    return 'FLAT'


def volume_breakout_strategy(df: pd.DataFrame) -> str:
    """
    Volume breakout: High volume breaking recent high/low
    """
    if len(df) < 72:
        return 'FLAT'
    
    # Calculate average volume
    avg_volume = df['Volume'].iloc[-24:].mean()
    current_volume = df['Volume'].iloc[-1]
    
    # Recent high/low (last 48 hours)
    recent_high = df['High'].iloc[-48:-1].max()
    recent_low = df['Low'].iloc[-48:-1].min()
    
    current_price = df['Close'].iloc[-1]
    
    # Volume spike required
    if current_volume < avg_volume * 1.5:
        return 'FLAT'
    
    # Breakout detection
    if current_price > recent_high:
        return 'UP'  # Bullish breakout
    elif current_price < recent_low:
        return 'DOWN'  # Bearish breakdown
    
    return 'FLAT'


# =============================================================================
# STRATEGY 6: MOVING AVERAGE CROSSOVER
# =============================================================================

def ma_crossover_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> str:
    """
    Moving average crossover strategy.
    
    Logic:
    - Fast MA crosses above slow MA → UP
    - Fast MA crosses below slow MA → DOWN
    
    Args:
        df: Historical OHLCV data
        fast: Fast MA period (hours)
        slow: Slow MA period (hours)
        
    Returns:
        'UP', 'DOWN', or 'FLAT'
    """
    if len(df) < slow + 5:
        return 'FLAT'
    
    fast_ma = df['Close'].rolling(fast).mean()
    slow_ma = df['Close'].rolling(slow).mean()
    
    # Current and previous positions
    current_fast = fast_ma.iloc[-1]
    current_slow = slow_ma.iloc[-1]
    prev_fast = fast_ma.iloc[-2]
    prev_slow = slow_ma.iloc[-2]
    
    if pd.isna(current_fast) or pd.isna(current_slow):
        return 'FLAT'
    
    # Crossover detection
    if prev_fast <= prev_slow and current_fast > current_slow:
        return 'UP'  # Bullish crossover
    elif prev_fast >= prev_slow and current_fast < current_slow:
        return 'DOWN'  # Bearish crossover
    
    return 'FLAT'


# =============================================================================
# STRATEGY 7: BOLLINGER BAND BOUNCE
# =============================================================================

def bollinger_strategy(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> str:
    """
    Bollinger Band strategy: Trade bounces off bands.
    
    Logic:
    - Price touches lower band → predict UP (bounce)
    - Price touches upper band → predict DOWN (pullback)
    
    Args:
        df: Historical OHLCV data
        period: MA period for bands
        num_std: Number of standard deviations
        
    Returns:
        'UP', 'DOWN', or 'FLAT'
    """
    if len(df) < period + 5:
        return 'FLAT'
    
    # Calculate Bollinger Bands
    ma = df['Close'].rolling(period).mean()
    std = df['Close'].rolling(period).std()
    
    upper_band = ma + (std * num_std)
    lower_band = ma - (std * num_std)
    
    current_price = df['Close'].iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    
    if pd.isna(current_upper) or pd.isna(current_lower):
        return 'FLAT'
    
    # Check if price is at bands
    if current_price <= current_lower:
        return 'UP'  # At lower band → bounce up
    elif current_price >= current_upper:
        return 'DOWN'  # At upper band → pullback
    
    return 'FLAT'


# =============================================================================
# STRATEGY 8: ALWAYS UP (Baseline - Crypto Bias)
# =============================================================================

def always_up_strategy(df: pd.DataFrame) -> str:
    """
    Always predict UP.
    
    Logic: Crypto has long-term bullish bias. What if we always go long?
    This is a baseline to compare against.
    """
    return 'UP'


def always_down_strategy(df: pd.DataFrame) -> str:
    """
    Always predict DOWN.
    
    Inverse of always_up. Should lose in bull markets.
    """
    return 'DOWN'


# =============================================================================
# STRATEGY 9: RANDOM (True Baseline)
# =============================================================================

def random_strategy(df: pd.DataFrame) -> str:
    """
    Random prediction.
    
    True baseline - should achieve ~50% over enough trades.
    If any strategy can't beat this, it has no edge.
    """
    return np.random.choice(['UP', 'DOWN'])


# =============================================================================
# ALL STRATEGIES (for easy import)
# =============================================================================

ALL_STRATEGIES = {
    'momentum': momentum_strategy,
    'momentum_strong': momentum_strong_strategy,
    'mean_reversion': mean_reversion_strategy,
    'mean_reversion_aggressive': mean_reversion_aggressive_strategy,
    'ghost_inverse': ghost_inverse_strategy,
    'ghost_inverse_strong': ghost_inverse_strong_strategy,
    'rsi': rsi_strategy,
    'rsi_extreme': rsi_extreme_strategy,
    'volume_spike': volume_spike_strategy,
    'volume_breakout': volume_breakout_strategy,
    'ma_crossover': ma_crossover_strategy,
    'bollinger': bollinger_strategy,
    'always_up': always_up_strategy,
    'always_down': always_down_strategy,
    'random': random_strategy,
}

# Core strategies to test
CORE_STRATEGIES = {
    'momentum': momentum_strategy,
    'mean_reversion': mean_reversion_strategy,
    'ghost_inverse': ghost_inverse_strategy,
    'rsi': rsi_strategy,
    'volume_spike': volume_spike_strategy,
}


if __name__ == "__main__":
    print("Available strategies:")
    for name in ALL_STRATEGIES.keys():
        print(f"  - {name}")
