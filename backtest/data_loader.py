"""
Data Loader for Backtesting Framework
Downloads and caches hourly OHLCV data from multiple sources
"""

import os
import pandas as pd
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
import json

# Symbols to download
CRYPTO_SYMBOLS = {
    'BTC': 'bitcoin',
    'ETH': 'ethereum',
    'SOL': 'solana',
    'XRP': 'ripple',
    'AVAX': 'avalanche-2',
    'LINK': 'chainlink',
}

# Data directory
DATA_DIR = Path(__file__).parent / "data"


def ensure_data_dir():
    """Create data directory if it doesn't exist"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_from_coingecko(symbol: str, coin_id: str, days: int = 365) -> pd.DataFrame:
    """
    Download hourly data from CoinGecko API.
    
    CoinGecko free tier limits:
    - Hourly data available for up to 90 days
    - We'll fetch multiple 90-day chunks and combine
    
    Args:
        symbol: Short name (BTC, ETH, etc.)
        coin_id: CoinGecko coin ID
        days: Number of days of data to fetch
        
    Returns:
        DataFrame with OHLCV-like data (CoinGecko provides price/volume)
    """
    print(f"Downloading {symbol} from CoinGecko...")
    
    all_data = []
    
    # CoinGecko gives hourly data for 1-90 days
    # For longer periods, we need to fetch in chunks
    chunk_days = 89  # Stay under 90 day limit
    
    end_timestamp = int(datetime.now().timestamp())
    start_timestamp = end_timestamp - (days * 24 * 3600)
    
    current_end = end_timestamp
    
    while current_end > start_timestamp:
        current_start = max(start_timestamp, current_end - (chunk_days * 24 * 3600))
        
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        params = {
            'vs_currency': 'usd',
            'from': current_start,
            'to': current_end
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                print("  Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            if 'prices' in data and data['prices']:
                all_data.extend(data['prices'])
                print(f"  Fetched {len(data['prices'])} price points")
            
            current_end = current_start
            time.sleep(1.5)  # Rate limit: ~30 calls/min
            
        except Exception as e:
            print(f"  Error fetching chunk: {e}")
            break
    
    if not all_data:
        print(f"  No data retrieved for {symbol}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'Close'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    df = df.sort_index()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    # Resample to hourly (CoinGecko returns ~5min intervals for recent data)
    df_hourly = df.resample('1h').last().dropna()
    
    # Create OHLCV-like columns (we only have close from CoinGecko)
    # For backtesting purposes, we'll use Close for all
    df_hourly['Open'] = df_hourly['Close'].shift(1)
    df_hourly['High'] = df_hourly['Close']  # Approximation
    df_hourly['Low'] = df_hourly['Close']   # Approximation
    df_hourly['Volume'] = 1e9  # Placeholder - CoinGecko market_chart doesn't give volume
    
    # Forward fill the first Open
    df_hourly['Open'] = df_hourly['Open'].bfill()
    
    print(f"  Downloaded {len(df_hourly)} hourly bars for {symbol}")
    if len(df_hourly) > 0:
        print(f"  Date range: {df_hourly.index[0]} to {df_hourly.index[-1]}")
    
    return df_hourly


def download_from_binance(symbol: str, pair: str = None, days: int = 365) -> pd.DataFrame:
    """
    Download hourly OHLCV data from Binance public API (no auth needed).
    
    Args:
        symbol: Short name (BTC, ETH, etc.)
        pair: Trading pair (default: {symbol}USDT)
        days: Number of days
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Downloading {symbol} from Binance...")
    
    if pair is None:
        pair = f"{symbol}USDT"
    
    all_data = []
    
    # Binance klines endpoint
    url = "https://api.binance.com/api/v3/klines"
    
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = end_time - (days * 24 * 3600 * 1000)
    
    current_start = start_time
    limit = 1000  # Max per request
    
    while current_start < end_time:
        params = {
            'symbol': pair,
            'interval': '1h',
            'startTime': current_start,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 429:
                print("  Rate limited, waiting 60s...")
                time.sleep(60)
                continue
            
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_data.extend(data)
            
            # Move to next chunk
            last_time = data[-1][0]
            current_start = last_time + 1
            
            print(f"  Fetched {len(data)} candles ({len(all_data)} total)")
            
            time.sleep(0.2)  # Rate limit
            
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    # Binance klines format: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    
    # Keep only OHLCV columns and convert to float
    ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[ohlcv_cols].astype(float)
    
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"  Downloaded {len(df)} hourly bars for {symbol}")
    if len(df) > 0:
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df


def download_symbol(symbol: str, coin_id: str = None, days: int = 365) -> pd.DataFrame:
    """
    Download data for a symbol, trying multiple sources.
    
    Args:
        symbol: Short name (BTC, ETH, etc.)
        coin_id: CoinGecko ID (optional)
        days: Number of days
        
    Returns:
        DataFrame with OHLCV data
    """
    # Try Binance first (better quality OHLCV data)
    df = download_from_binance(symbol, days=days)
    
    if df.empty and coin_id:
        # Fall back to CoinGecko
        df = download_from_coingecko(symbol, coin_id, days=days)
    
    return df


def save_data(df: pd.DataFrame, symbol: str) -> str:
    """Save DataFrame to CSV"""
    ensure_data_dir()
    filepath = DATA_DIR / f"{symbol}_hourly.csv"
    df.to_csv(filepath)
    print(f"  Saved to {filepath}")
    return str(filepath)


def load_data(symbol: str) -> pd.DataFrame:
    """
    Load cached data for a symbol.
    
    Args:
        symbol: Short name (BTC, ETH, etc.)
        
    Returns:
        DataFrame with OHLCV data, or empty DataFrame if not found
    """
    filepath = DATA_DIR / f"{symbol}_hourly.csv"
    
    if not filepath.exists():
        print(f"No cached data for {symbol}. Run download_all_data() first.")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def download_all_data(days: int = 365, force: bool = False):
    """
    Download data for all symbols.
    
    Args:
        days: Number of days of history
        force: If True, re-download even if cached
    """
    ensure_data_dir()
    
    print("=" * 60)
    print("DOWNLOADING CRYPTO DATA")
    print(f"Period: Last {days} days")
    print("=" * 60)
    
    results = {}
    
    for symbol, coin_id in CRYPTO_SYMBOLS.items():
        filepath = DATA_DIR / f"{symbol}_hourly.csv"
        
        # Skip if cached (unless force=True)
        if filepath.exists() and not force:
            df = load_data(symbol)
            print(f"{symbol}: Using cached data ({len(df)} bars)")
            results[symbol] = len(df)
            continue
        
        # Download
        df = download_symbol(symbol, coin_id, days=days)
        
        if not df.empty:
            save_data(df, symbol)
            results[symbol] = len(df)
        else:
            results[symbol] = 0
        
        # Small delay between symbols
        time.sleep(1)
    
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    for symbol, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {symbol}: {count} bars")
    
    return results


def get_data_summary() -> dict:
    """Get summary of all cached data"""
    ensure_data_dir()
    
    summary = {}
    for symbol in CRYPTO_SYMBOLS.keys():
        filepath = DATA_DIR / f"{symbol}_hourly.csv"
        if filepath.exists():
            df = load_data(symbol)
            summary[symbol] = {
                'bars': len(df),
                'start': df.index[0] if len(df) > 0 else None,
                'end': df.index[-1] if len(df) > 0 else None,
            }
        else:
            summary[symbol] = {'bars': 0, 'start': None, 'end': None}
    
    return summary


if __name__ == "__main__":
    # Download all data
    download_all_data(days=365, force=True)
    
    # Show summary
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    summary = get_data_summary()
    for symbol, info in summary.items():
        if info['bars'] > 0:
            print(f"{symbol}: {info['bars']} bars from {info['start']} to {info['end']}")
        else:
            print(f"{symbol}: No data")
