#!/usr/bin/env python3
"""
Run Backtests - Execute all strategies across all symbols and holding periods
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

from backtest.data_loader import load_data, download_all_data, CRYPTO_SYMBOLS
from backtest.engine import BacktestEngine, calculate_significance
from backtest.strategies import ALL_STRATEGIES, CORE_STRATEGIES


# Configuration
SYMBOLS = list(CRYPTO_SYMBOLS.keys())  # ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'LINK']
HOLDING_PERIODS = [24, 48, 72, 168]  # hours (1 day, 2 days, 3 days, 1 week)
MIN_TRADES = 30  # Minimum trades to include in results


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
        min_lookback_hours=168,  # 7 days minimum history
        step_hours=24,  # Make prediction every 24 hours
    )
    
    result = engine.run()
    
    # Calculate significance
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


def run_all_backtests(strategies: Dict = None, symbols: List[str] = None) -> pd.DataFrame:
    """
    Run all backtests for all combinations of strategies, symbols, and holding periods.
    
    Returns:
        DataFrame with all backtest results
    """
    if strategies is None:
        strategies = CORE_STRATEGIES
    if symbols is None:
        symbols = SYMBOLS
    
    print("=" * 70)
    print("RUNNING BACKTESTS")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Strategies: {list(strategies.keys())}")
    print(f"Holding periods: {HOLDING_PERIODS} hours")
    print()
    
    results = []
    total_combos = len(symbols) * len(strategies) * len(HOLDING_PERIODS)
    completed = 0
    
    for symbol in symbols:
        # Load data
        df = load_data(symbol)
        if df.empty or len(df) < 500:
            print(f"Skipping {symbol}: insufficient data ({len(df)} bars)")
            continue
        
        print(f"\n{symbol}: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        for strategy_name, strategy_fn in strategies.items():
            for holding_hours in HOLDING_PERIODS:
                completed += 1
                
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
                        status = "✓" if result['win_rate'] > 0.50 else "✗"
                        sig = "*" if result['is_significant'] else ""
                        print(f"  {status} {strategy_name}/{holding_hours}h: {result['win_rate']:.1%} ({result['total_trades']} trades){sig}")
                    
                except Exception as e:
                    print(f"  ERROR {strategy_name}/{holding_hours}h: {e}")
    
    print(f"\nCompleted {completed}/{total_combos} backtests")
    
    return pd.DataFrame(results)


def generate_report(results_df: pd.DataFrame, output_path: str = None) -> str:
    """Generate text report from backtest results"""
    
    if results_df.empty:
        return "No results to report"
    
    lines = []
    lines.append("=" * 70)
    lines.append("BACKTEST RESULTS - Simple Trading Strategies")
    lines.append("=" * 70)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Period: Jan 2025 - Jan 2026")
    lines.append(f"Data: Hourly OHLCV from Yahoo Finance")
    lines.append(f"Total strategy/symbol/period combinations: {len(results_df)}")
    lines.append("")
    
    # Overall statistics
    total_trades = results_df['total_trades'].sum()
    total_wins = results_df['wins'].sum()
    overall_win_rate = total_wins / total_trades if total_trades > 0 else 0
    
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total trades across all backtests: {total_trades:,}")
    lines.append(f"Overall win rate: {overall_win_rate:.1%}")
    lines.append(f"Strategies tested: {results_df['strategy'].nunique()}")
    lines.append(f"Symbols tested: {results_df['symbol'].nunique()}")
    lines.append("")
    
    # TOP PERFORMERS (by win rate, min 50 trades)
    lines.append("=" * 70)
    lines.append("TOP 20 PERFORMERS (by win rate)")
    lines.append("=" * 70)
    
    top_df = results_df[results_df['total_trades'] >= 50].sort_values('win_rate', ascending=False).head(20)
    
    lines.append(f"{'Rank':<4} {'Symbol':<6} {'Strategy':<20} {'Hold':<6} {'Trades':<7} {'Win Rate':<10} {'Significant?'}")
    lines.append("-" * 70)
    
    for i, (_, row) in enumerate(top_df.iterrows(), 1):
        sig = "YES*" if row['is_significant'] else "no"
        lines.append(f"{i:<4} {row['symbol']:<6} {row['strategy']:<20} {row['holding_hours']:<6}h {row['total_trades']:<7} {row['win_rate']:.1%}       {sig}")
    
    lines.append("")
    
    # WORST PERFORMERS
    lines.append("=" * 70)
    lines.append("WORST 10 PERFORMERS (strategies that DON'T work)")
    lines.append("=" * 70)
    
    bottom_df = results_df[results_df['total_trades'] >= 50].sort_values('win_rate', ascending=True).head(10)
    
    lines.append(f"{'Rank':<4} {'Symbol':<6} {'Strategy':<20} {'Hold':<6} {'Trades':<7} {'Win Rate':<10}")
    lines.append("-" * 70)
    
    for i, (_, row) in enumerate(bottom_df.iterrows(), 1):
        lines.append(f"{i:<4} {row['symbol']:<6} {row['strategy']:<20} {row['holding_hours']:<6}h {row['total_trades']:<7} {row['win_rate']:.1%}")
    
    lines.append("")
    
    # BY STRATEGY (average across all symbols)
    lines.append("=" * 70)
    lines.append("WIN RATE BY STRATEGY (averaged across symbols)")
    lines.append("=" * 70)
    
    strategy_avg = results_df.groupby('strategy').agg({
        'win_rate': 'mean',
        'total_trades': 'sum',
        'is_significant': 'sum'
    }).sort_values('win_rate', ascending=False)
    
    lines.append(f"{'Strategy':<25} {'Avg Win Rate':<12} {'Total Trades':<12} {'Significant Tests'}")
    lines.append("-" * 70)
    
    for strategy, row in strategy_avg.iterrows():
        lines.append(f"{strategy:<25} {row['win_rate']:.1%}        {int(row['total_trades']):<12} {int(row['is_significant'])}")
    
    lines.append("")
    
    # BY SYMBOL (average across all strategies)
    lines.append("=" * 70)
    lines.append("WIN RATE BY SYMBOL (averaged across strategies)")
    lines.append("=" * 70)
    
    symbol_avg = results_df.groupby('symbol').agg({
        'win_rate': 'mean',
        'total_trades': 'sum',
        'is_significant': 'sum'
    }).sort_values('win_rate', ascending=False)
    
    lines.append(f"{'Symbol':<10} {'Avg Win Rate':<12} {'Total Trades':<12} {'Significant Tests'}")
    lines.append("-" * 70)
    
    for symbol, row in symbol_avg.iterrows():
        lines.append(f"{symbol:<10} {row['win_rate']:.1%}        {int(row['total_trades']):<12} {int(row['is_significant'])}")
    
    lines.append("")
    
    # BY HOLDING PERIOD
    lines.append("=" * 70)
    lines.append("WIN RATE BY HOLDING PERIOD")
    lines.append("=" * 70)
    
    hold_avg = results_df.groupby('holding_hours').agg({
        'win_rate': 'mean',
        'total_trades': 'sum',
        'is_significant': 'sum'
    }).sort_values('win_rate', ascending=False)
    
    lines.append(f"{'Hold Period':<15} {'Avg Win Rate':<12} {'Total Trades':<12} {'Significant Tests'}")
    lines.append("-" * 70)
    
    for hold, row in hold_avg.iterrows():
        lines.append(f"{hold}h           {row['win_rate']:.1%}        {int(row['total_trades']):<12} {int(row['is_significant'])}")
    
    lines.append("")
    
    # GHOST INVERSE SPECIFIC
    lines.append("=" * 70)
    lines.append("GHOST INVERSE STRATEGY - DETAILED RESULTS")
    lines.append("=" * 70)
    
    ghost_df = results_df[results_df['strategy'].str.contains('ghost', case=False)]
    
    if not ghost_df.empty:
        ghost_total = ghost_df['total_trades'].sum()
        ghost_wins = ghost_df['wins'].sum()
        ghost_rate = ghost_wins / ghost_total if ghost_total > 0 else 0
        ghost_significant = (ghost_df['is_significant'].sum() > 0)
        
        lines.append(f"Total Ghost Inverse Trades: {ghost_total}")
        lines.append(f"Total Wins: {ghost_wins}")
        lines.append(f"Overall Win Rate: {ghost_rate:.1%}")
        lines.append(f"Any Significant Results: {'YES' if ghost_significant else 'NO'}")
        lines.append("")
        
        lines.append("By Symbol/Period:")
        for _, row in ghost_df.sort_values('win_rate', ascending=False).iterrows():
            sig = "*" if row['is_significant'] else ""
            lines.append(f"  {row['symbol']} {row['holding_hours']}h: {row['win_rate']:.1%} ({row['total_trades']} trades){sig}")
    else:
        lines.append("No ghost inverse results found")
    
    lines.append("")
    
    # STATISTICAL SIGNIFICANCE SUMMARY
    lines.append("=" * 70)
    lines.append("STATISTICALLY SIGNIFICANT RESULTS (p < 0.05)")
    lines.append("=" * 70)
    
    sig_df = results_df[results_df['is_significant'] == True].sort_values('win_rate', ascending=False)
    
    if not sig_df.empty:
        lines.append(f"Found {len(sig_df)} significant results out of {len(results_df)} tests")
        lines.append("")
        lines.append(f"{'Symbol':<6} {'Strategy':<20} {'Hold':<6} {'Win Rate':<10} {'Trades':<7} {'p-value':<10} {'95% CI'}")
        lines.append("-" * 80)
        
        for _, row in sig_df.iterrows():
            ci = f"[{row['ci_lower']:.1%}-{row['ci_upper']:.1%}]"
            lines.append(f"{row['symbol']:<6} {row['strategy']:<20} {row['holding_hours']:<6}h {row['win_rate']:.1%}      {row['total_trades']:<7} {row['p_value']:.4f}     {ci}")
    else:
        lines.append("NO statistically significant results found.")
        lines.append("This suggests the market may not be predictable with simple strategies.")
    
    lines.append("")
    
    # CONCLUSIONS
    lines.append("=" * 70)
    lines.append("CONCLUSIONS")
    lines.append("=" * 70)
    
    best_strategy = strategy_avg.index[0] if not strategy_avg.empty else "N/A"
    best_strategy_rate = strategy_avg['win_rate'].iloc[0] if not strategy_avg.empty else 0
    
    best_symbol = symbol_avg.index[0] if not symbol_avg.empty else "N/A"
    best_symbol_rate = symbol_avg['win_rate'].iloc[0] if not symbol_avg.empty else 0
    
    best_hold = hold_avg.index[0] if not hold_avg.empty else "N/A"
    best_hold_rate = hold_avg['win_rate'].iloc[0] if not hold_avg.empty else 0
    
    lines.append(f"Best Strategy: {best_strategy} ({best_strategy_rate:.1%} avg)")
    lines.append(f"Best Symbol: {best_symbol} ({best_symbol_rate:.1%} avg)")
    lines.append(f"Best Hold Period: {best_hold}h ({best_hold_rate:.1%} avg)")
    lines.append("")
    
    # Ghost inverse validation
    if not ghost_df.empty:
        ghost_validated = ghost_rate > 0.52 and ghost_total >= 500
        lines.append(f"Ghost Inverse Validated: {'YES ✓' if ghost_validated else 'NO ✗'}")
        lines.append(f"  - Win rate: {ghost_rate:.1%} (need >52%)")
        lines.append(f"  - Total trades: {ghost_total} (need 500+)")
    
    lines.append("")
    
    # Final verdict
    any_significant = results_df['is_significant'].any()
    best_overall_rate = results_df['win_rate'].max()
    
    if any_significant and best_overall_rate > 0.55:
        lines.append("VERDICT: Some strategies show edge. Worth pursuing.")
    elif best_overall_rate > 0.52:
        lines.append("VERDICT: Marginal edge detected. Needs more data/validation.")
    else:
        lines.append("VERDICT: No clear edge found. Market may not be predictable.")
    
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    
    return report


def main():
    """Main entry point"""
    
    # Ensure we have data
    print("Checking for data...")
    download_all_data(force=False)  # Only download if not cached
    
    # Run all backtests with all strategies
    print("\n" + "=" * 70)
    print("RUNNING FULL BACKTEST SUITE")
    print("=" * 70)
    
    results_df = run_all_backtests(
        strategies=ALL_STRATEGIES,  # All 15 strategies
        symbols=SYMBOLS  # All 6 crypto symbols
    )
    
    # Save raw results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_csv = results_dir / "backtest_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nRaw results saved to: {results_csv}")
    
    # Generate report
    report_path = results_dir / "backtest_report.txt"
    report = generate_report(results_df, str(report_path))
    
    # Print report
    print("\n" + report)
    
    return results_df


if __name__ == "__main__":
    main()
