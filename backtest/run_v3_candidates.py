#!/usr/bin/env python3
"""
Run V3 Candidate Backtests - Test RNDR, TURBO, CHZ for V3 validation
Based on strong paper trade performance
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
from scipy import stats

from backtest.data_loader import download_from_binance, save_data, load_data, DATA_DIR
from backtest.engine import BacktestEngine, calculate_significance
from backtest.strategies import CORE_STRATEGIES, ALL_STRATEGIES


# V3 Candidates based on paper trade performance
V3_CANDIDATES = {
    'RNDR': {'paper_wr': 0.68, 'paper_trades': 347},
    'TURBO': {'paper_wr': 0.485, 'paper_trades': 103},
    'CHZ': {'paper_wr': 0.446, 'paper_trades': 298},
}

HOLDING_PERIODS = [24, 48, 72, 168]  # hours
MIN_TRADES = 50


def download_candidate_data():
    """Download historical data for V3 candidates"""
    print("=" * 70)
    print("DOWNLOADING V3 CANDIDATE DATA")
    print("=" * 70)
    
    for symbol in V3_CANDIDATES.keys():
        filepath = DATA_DIR / f"{symbol}_hourly.csv"
        if filepath.exists():
            # Check if data is recent
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            if len(df) > 500:
                print(f"{symbol}: Already have {len(df)} bars")
                continue
        
        print(f"\nDownloading {symbol}...")
        df = download_from_binance(symbol, days=365)
        
        if not df.empty:
            save_data(df, symbol)
        else:
            print(f"  WARNING: Failed to download {symbol}")


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
        step_hours=24,
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
        'win_rate': result['win_rate'],
        'is_significant': sig['is_significant'],
        'p_value': sig['p_value'],
        'ci_lower': sig['confidence_interval'][0],
        'ci_upper': sig['confidence_interval'][1],
    }


def run_v3_candidate_backtests():
    """Run all backtests for V3 candidates"""
    
    print("=" * 70)
    print("V3 CANDIDATE BACKTESTS")
    print("=" * 70)
    print(f"Symbols: {list(V3_CANDIDATES.keys())}")
    print(f"Strategies: {list(CORE_STRATEGIES.keys())}")
    print(f"Holding periods: {HOLDING_PERIODS} hours")
    print()
    
    results = []
    
    for symbol, info in V3_CANDIDATES.items():
        # Load data
        df = load_data(symbol)
        if df.empty or len(df) < 500:
            print(f"Skipping {symbol}: insufficient data ({len(df)} bars)")
            continue
        
        print(f"\n{'='*70}")
        print(f"{symbol} - Paper WR: {info['paper_wr']:.1%} ({info['paper_trades']} trades)")
        print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        print('='*70)
        
        for strategy_name, strategy_fn in CORE_STRATEGIES.items():
            for holding_hours in HOLDING_PERIODS:
                result = run_single_backtest(
                    symbol, strategy_name, strategy_fn, 
                    holding_hours, df
                )
                results.append(result)
                
                if result['win_rate'] > 0.52 and result['total_trades'] >= MIN_TRADES:
                    sig_marker = "*" if result['is_significant'] else ""
                    print(f"  {strategy_name:25} {holding_hours:3}h: {result['win_rate']:5.1%} ({result['total_trades']} trades) {sig_marker}")
    
    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Analyze backtest results and recommend V3 strategies"""
    
    print("\n" + "=" * 70)
    print("V3 VALIDATION ANALYSIS")
    print("=" * 70)
    
    # Filter for significant results with edge
    winners = df[
        (df['total_trades'] >= MIN_TRADES) & 
        (df['win_rate'] > 0.52)
    ].sort_values('win_rate', ascending=False)
    
    print("\n📊 TOP PERFORMERS (52%+ WR, 50+ trades):")
    print("-" * 70)
    
    for _, row in winners.head(20).iterrows():
        sig = "✓ SIG" if row['is_significant'] else ""
        print(f"{row['symbol']:6} {row['strategy']:25} {row['holding_hours']:3}h  "
              f"{row['win_rate']:5.1%} ({row['total_trades']:3} trades)  p={row['p_value']:.4f} {sig}")
    
    # V3 Recommendations
    print("\n" + "=" * 70)
    print("🎯 V3 VALIDATION RECOMMENDATIONS")
    print("=" * 70)
    
    for symbol in V3_CANDIDATES.keys():
        symbol_results = winners[winners['symbol'] == symbol]
        
        if len(symbol_results) == 0:
            print(f"\n{symbol}: ❌ No strategies meet criteria")
            continue
        
        # Get best result
        best = symbol_results.iloc[0]
        significant_results = symbol_results[symbol_results['is_significant']]
        
        print(f"\n{symbol}:")
        if len(significant_results) > 0:
            best_sig = significant_results.iloc[0]
            print(f"  ✅ VALIDATE with: {best_sig['strategy']} @ {best_sig['holding_hours']}h")
            print(f"     Win Rate: {best_sig['win_rate']:.1%} ({best_sig['total_trades']} trades)")
            print(f"     p-value: {best_sig['p_value']:.4f} (statistically significant)")
            print(f"     95% CI: [{best_sig['ci_lower']:.1%} - {best_sig['ci_upper']:.1%}]")
        else:
            print(f"  ⚠️  Best: {best['strategy']} @ {best['holding_hours']}h = {best['win_rate']:.1%}")
            print(f"     Not statistically significant yet (p={best['p_value']:.4f})")
    
    return winners


def main():
    # Download data if needed
    download_candidate_data()
    
    # Run backtests
    results_df = run_v3_candidate_backtests()
    
    # Analyze and recommend
    winners = analyze_results(results_df)
    
    # Save results
    results_path = Path(__file__).parent / "results" / "v3_candidates_backtest.csv"
    results_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")
    
    return results_df, winners


if __name__ == "__main__":
    main()
