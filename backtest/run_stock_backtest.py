#!/usr/bin/env python3
"""
Run Stock Backtests - Execute all strategies across stock symbols
Uses Yahoo Finance for hourly stock data
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import yfinance as yf

from backtest.engine import BacktestEngine, calculate_significance
from backtest.strategies import ALL_STRATEGIES, CORE_STRATEGIES


# Stock symbols to test
STOCK_SYMBOLS = [
    "SNOW",   # Snowflake - cloud data
    "DDOG",   # Datadog - monitoring
    "NET",    # Cloudflare - CDN/security
    "PANW",   # Palo Alto Networks - cybersecurity
    "FTNT",   # Fortinet - cybersecurity
]

# Also test some popular ones
POPULAR_STOCKS = [
    "AAPL",   # Apple
    "NVDA",   # NVIDIA
    "TSLA",   # Tesla
    "MSFT",   # Microsoft
    "META",   # Meta
]

HOLDING_PERIODS = [24, 48, 72, 168]  # hours
MIN_TRADES = 30
DATA_DIR = Path(__file__).parent / "data" / "stocks"


def download_stock_data(symbol: str, days: int = 730) -> pd.DataFrame:
    """
    Download hourly stock data from Yahoo Finance.
    
    Yahoo provides:
    - 1-hour data for up to 730 days
    - Only market hours (no 24/7 like crypto)
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DATA_DIR / f"{symbol}_hourly.csv"
    
    # Check cache (refresh if older than 1 day)
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=1):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"  {symbol}: Loaded {len(df)} bars from cache")
            return df
    
    print(f"  Downloading {symbol} from Yahoo Finance...")
    
    try:
        # Use download() method instead of Ticker().history() - more reliable
        df = yf.download(symbol, period=f"{days}d", interval="1h", progress=False)
        
        if df.empty:
            print(f"  {symbol}: No data returned")
            return pd.DataFrame()
        
        # Handle multi-level columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Keep only OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Save to cache
        df.to_csv(cache_file)
        print(f"  {symbol}: Downloaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")
        
        return df
        
    except Exception as e:
        print(f"  {symbol}: Error - {e}")
        return pd.DataFrame()


def run_single_backtest(
    symbol: str, 
    strategy_name: str, 
    strategy_fn, 
    holding_hours: int,
    data: pd.DataFrame
) -> Dict:
    """Run a single backtest and return results"""
    
    engine = BacktestEngine(
        data=data,
        strategy_fn=strategy_fn,
        holding_hours=holding_hours,
        min_lookback_hours=168,
        step_hours=24,  # One prediction per day
    )
    
    result = engine.run()
    sig = calculate_significance(result['wins'], result['total_trades'])
    
    return {
        'symbol': symbol,
        'strategy': strategy_name,
        'holding_hours': holding_hours,
        'total_trades': result['total_trades'],
        'wins': result['wins'],
        'losses': result['losses'],
        'flats_skipped': result['flats_skipped'],
        'win_rate': result['win_rate'],
        'avg_win_pct': result['avg_win_pct'],
        'avg_loss_pct': result['avg_loss_pct'],
        'is_significant': sig['is_significant'],
        'p_value': sig['p_value'],
        'ci_lower': sig['confidence_interval'][0],
        'ci_upper': sig['confidence_interval'][1],
    }


def run_all_stock_backtests(symbols: List[str] = None, strategies: Dict = None) -> pd.DataFrame:
    """Run backtests for all stock combinations"""
    
    if symbols is None:
        symbols = STOCK_SYMBOLS + POPULAR_STOCKS
    if strategies is None:
        strategies = CORE_STRATEGIES
    
    print("=" * 70)
    print("STOCK BACKTEST - Finding Validated Strategies")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Strategies: {list(strategies.keys())}")
    print(f"Holding periods: {HOLDING_PERIODS} hours")
    print()
    
    # Download all data first
    print("Downloading stock data...")
    stock_data = {}
    for symbol in symbols:
        df = download_stock_data(symbol)
        if not df.empty and len(df) >= 500:
            stock_data[symbol] = df
        else:
            print(f"  Skipping {symbol}: insufficient data")
    
    print(f"\nLoaded data for {len(stock_data)} stocks")
    print()
    
    results = []
    
    for symbol, df in stock_data.items():
        print(f"\n{symbol}: {len(df)} bars")
        
        for strategy_name, strategy_fn in strategies.items():
            for holding_hours in HOLDING_PERIODS:
                try:
                    result = run_single_backtest(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        strategy_fn=strategy_fn,
                        holding_hours=holding_hours,
                        data=df
                    )
                    
                    if result['total_trades'] >= MIN_TRADES:
                        results.append(result)
                        status = "✓" if result['win_rate'] > 0.52 else "·"
                        sig = " ***" if result['is_significant'] else ""
                        print(f"  {status} {strategy_name}/{holding_hours}h: {result['win_rate']:.1%} ({result['total_trades']} trades){sig}")
                    
                except Exception as e:
                    print(f"  ERROR {strategy_name}/{holding_hours}h: {e}")
    
    return pd.DataFrame(results)


def generate_stock_report(results_df: pd.DataFrame) -> str:
    """Generate report for stock backtests"""
    
    if results_df.empty:
        return "No results to report"
    
    lines = []
    lines.append("=" * 70)
    lines.append("STOCK BACKTEST RESULTS")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Data: Hourly OHLCV from Yahoo Finance")
    lines.append(f"Total combinations tested: {len(results_df)}")
    lines.append("")
    
    # Overall stats
    total_trades = results_df['total_trades'].sum()
    total_wins = results_df['wins'].sum()
    overall_wr = total_wins / total_trades if total_trades > 0 else 0
    
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total trades: {total_trades:,}")
    lines.append(f"Overall win rate: {overall_wr:.1%}")
    lines.append(f"Stocks tested: {results_df['symbol'].nunique()}")
    lines.append("")
    
    # STATISTICALLY SIGNIFICANT RESULTS
    sig_df = results_df[results_df['is_significant'] == True].sort_values('win_rate', ascending=False)
    
    lines.append("=" * 70)
    lines.append(f"STATISTICALLY SIGNIFICANT RESULTS (p < 0.05): {len(sig_df)} found")
    lines.append("=" * 70)
    
    if len(sig_df) > 0:
        lines.append(f"{'Symbol':<8} {'Strategy':<25} {'Hold':<6} {'Win Rate':<10} {'Trades':<8} {'p-value':<10} {'95% CI'}")
        lines.append("-" * 90)
        
        for _, row in sig_df.iterrows():
            ci = f"[{row['ci_lower']:.1%}-{row['ci_upper']:.1%}]"
            lines.append(f"{row['symbol']:<8} {row['strategy']:<25} {row['holding_hours']:<6}h {row['win_rate']:.1%}      {row['total_trades']:<8} {row['p_value']:.4f}     {ci}")
    else:
        lines.append("No statistically significant results found.")
        lines.append("This means none of the strategies showed edge above random (p < 0.05)")
    
    lines.append("")
    
    # TOP 20 by win rate
    lines.append("=" * 70)
    lines.append("TOP 20 PERFORMERS (by win rate, min 50 trades)")
    lines.append("=" * 70)
    
    top_df = results_df[results_df['total_trades'] >= 50].sort_values('win_rate', ascending=False).head(20)
    
    lines.append(f"{'Rank':<5} {'Symbol':<8} {'Strategy':<25} {'Hold':<6} {'Trades':<8} {'Win Rate':<10} {'Sig?'}")
    lines.append("-" * 80)
    
    for i, (_, row) in enumerate(top_df.iterrows(), 1):
        sig = "YES*" if row['is_significant'] else "no"
        lines.append(f"{i:<5} {row['symbol']:<8} {row['strategy']:<25} {row['holding_hours']:<6}h {row['total_trades']:<8} {row['win_rate']:.1%}      {sig}")
    
    lines.append("")
    
    # BY SYMBOL
    lines.append("=" * 70)
    lines.append("WIN RATE BY SYMBOL (averaged)")
    lines.append("=" * 70)
    
    symbol_avg = results_df.groupby('symbol').agg({
        'win_rate': 'mean',
        'total_trades': 'sum',
        'is_significant': 'sum'
    }).sort_values('win_rate', ascending=False)
    
    lines.append(f"{'Symbol':<10} {'Avg Win Rate':<15} {'Total Trades':<15} {'Sig Tests'}")
    lines.append("-" * 50)
    
    for symbol, row in symbol_avg.iterrows():
        lines.append(f"{symbol:<10} {row['win_rate']:.1%}           {int(row['total_trades']):<15} {int(row['is_significant'])}")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("V3 CANDIDATES (p < 0.05 with win rate > 55%)")
    lines.append("=" * 70)
    
    v3_candidates = sig_df[sig_df['win_rate'] > 0.55]
    if len(v3_candidates) > 0:
        for _, row in v3_candidates.iterrows():
            lines.append(f"'{row['symbol']}': {{")
            lines.append(f"    'strategy': '{row['strategy']}',")
            lines.append(f"    'hold_hours': {row['holding_hours']},")
            lines.append(f"    'win_rate': {row['win_rate']:.3f},")
            lines.append(f"    'sample_size': {row['total_trades']},")
            lines.append(f"    'p_value': {row['p_value']:.4f},")
            lines.append(f"}},")
            lines.append("")
    else:
        lines.append("No V3 candidates found (need p<0.05 AND win_rate>55%)")
    
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run stock backtests')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to test')
    parser.add_argument('--cloud-only', action='store_true', help='Only test cloud/cyber stocks')
    parser.add_argument('--popular-only', action='store_true', help='Only test popular stocks')
    args = parser.parse_args()
    
    if args.symbols:
        symbols = args.symbols
    elif args.cloud_only:
        symbols = STOCK_SYMBOLS
    elif args.popular_only:
        symbols = POPULAR_STOCKS
    else:
        symbols = STOCK_SYMBOLS + POPULAR_STOCKS
    
    # Run backtests
    results = run_all_stock_backtests(symbols=symbols)
    
    # Generate report
    report = generate_stock_report(results)
    print("\n" + report)
    
    # Save report
    output_path = Path(__file__).parent / "results" / "stock_backtest_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")
    
    # Save CSV
    csv_path = Path(__file__).parent / "results" / "stock_backtest_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
