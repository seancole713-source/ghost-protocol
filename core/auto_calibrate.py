#!/usr/bin/env python3
"""
Ghost Auto-Calibration System
Automatically finds optimal strategies for each symbol based on recent data.

Runs weekly to:
1. Download latest 6 months of data (crypto + stocks)
2. Backtest all strategy/symbol/timeframe combinations
3. Find statistically significant winners (p < 0.05)
4. Update V3_VALIDATED_STRATEGIES automatically
5. Send alerts about changes
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import logging
from config.symbols import DIRECTION_FLIP

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from backtest.engine import BacktestEngine, calculate_significance
from backtest.strategies import ALL_STRATEGIES, CORE_STRATEGIES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Crypto symbols to test (via CoinGecko or similar)
CRYPTO_SYMBOLS = ['ETH', 'XRP', 'LINK', 'CHZ', 'BTC', 'SOL', 'AVAX', 'DOGE', 'ADA']

# Stock symbols to test (via Yahoo Finance)
STOCK_SYMBOLS = [
    # Cybersecurity (current winners)
    'PANW', 'NET', 'FTNT', 'CRWD', 'ZS',
    # Cloud/SaaS
    'SNOW', 'DDOG', 'NOW', 'WDAY', 'HUBS',
    # Tech giants
    'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META',
    # High-beta
    'TSLA', 'AMD', 'MSTR', 'COIN',
]

# Hold periods to test (hours)
HOLD_PERIODS = [24, 48, 72, 168]

# Minimum requirements for validation
MIN_TRADES = 50          # Need at least 50 trades
MIN_WIN_RATE = 0.54      # Need >54% win rate
MAX_P_VALUE = 0.05       # Need p < 0.05 (statistically significant)

# Strategies excluded from calibration candidates:
# - random: coin flip baseline, "wins" are statistical noise not repeatable edge
# - always_up/always_down: directional bias, not a strategy
# These are useful as baselines to BEAT, not strategies to DEPLOY.
EXCLUDED_STRATEGIES = {'random', 'always_up', 'always_down'}

# Use all real strategies for testing (exclude baselines)
DEFAULT_STRATEGIES = {k: v for k, v in ALL_STRATEGIES.items() if k not in EXCLUDED_STRATEGIES}

# Data settings
DATA_DIR = Path(__file__).parent.parent / "backtest" / "data"
CALIBRATION_DIR = Path(__file__).parent.parent / "data" / "calibration"


# =============================================================================
# DATA LOADING
# =============================================================================

def download_stock_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """Download hourly stock data from Yahoo Finance"""
    if not YFINANCE_AVAILABLE:
        logger.warning("yfinance not available")
        return pd.DataFrame()
    
    cache_dir = DATA_DIR / "stocks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_hourly.csv"
    
    # Check cache (refresh if older than 1 day)
    if cache_file.exists():
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=1):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
    
    try:
        df = yf.download(symbol, period=f"{days}d", interval="1h", progress=False)
        
        if df.empty:
            return pd.DataFrame()
        
        # Handle multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.to_csv(cache_file)
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        return pd.DataFrame()


def load_crypto_data(symbol: str) -> pd.DataFrame:
    """Load crypto data from cache"""
    cache_file = DATA_DIR / f"{symbol}_hourly.csv"
    
    if cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
    
    # Try alternate location
    cache_file2 = DATA_DIR / "crypto" / f"{symbol}_hourly.csv"
    if cache_file2.exists():
        return pd.read_csv(cache_file2, index_col=0, parse_dates=True)
    
    return pd.DataFrame()


# =============================================================================
# BACKTESTING
# =============================================================================

def run_single_backtest(
    symbol: str,
    strategy_name: str,
    strategy_fn,
    holding_hours: int,
    data: pd.DataFrame
) -> Optional[Dict]:
    """Run a single backtest and return results"""
    
    if len(data) < 500:
        return None
    
    try:
        engine = BacktestEngine(
            data=data,
            strategy_fn=strategy_fn,
            holding_hours=holding_hours,
            min_lookback_hours=168,
            step_hours=24,
        )
        
        result = engine.run()
        
        if result['total_trades'] < MIN_TRADES:
            return None
        
        sig = calculate_significance(result['wins'], result['total_trades'])
        
        return {
            'symbol': symbol,
            'strategy': strategy_name,
            'holding_hours': holding_hours,
            'total_trades': result['total_trades'],
            'wins': result['wins'],
            'win_rate': result['win_rate'],
            'is_significant': sig['is_significant'],
            'p_value': sig['p_value'],
            'ci_lower': sig['confidence_interval'][0],
            'ci_upper': sig['confidence_interval'][1],
        }
        
    except Exception as e:
        logger.error(f"Backtest error {symbol}/{strategy_name}/{holding_hours}h: {e}")
        return None


def run_all_backtests(
    symbols: List[str],
    asset_type: str,
    strategies: Dict = None
) -> pd.DataFrame:
    """Run all backtests for given symbols"""
    
    if strategies is None:
        strategies = DEFAULT_STRATEGIES
    
    results = []
    
    for symbol in symbols:
        # Load data based on asset type
        if asset_type == 'crypto':
            df = load_crypto_data(symbol)
        else:
            df = download_stock_data(symbol)
        
        if df.empty or len(df) < 500:
            logger.info(f"Skipping {symbol}: insufficient data")
            continue
        
        logger.info(f"Testing {symbol}: {len(df)} bars")
        
        for strategy_name, strategy_fn in strategies.items():
            for holding_hours in HOLD_PERIODS:
                result = run_single_backtest(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    strategy_fn=strategy_fn,
                    holding_hours=holding_hours,
                    data=df
                )
                
                if result:
                    results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# FIND WINNERS
# =============================================================================

def find_validated_strategies(results_df: pd.DataFrame) -> Dict:
    """
    Find the best strategy for each symbol that meets validation criteria.
    
    Returns dict in V3_VALIDATED_STRATEGIES format.
    """
    if results_df.empty:
        return {}
    
    validated = {}
    
    # Filter to significant results with good win rate
    winners = results_df[
        (results_df['is_significant'] == True) &
        (results_df['win_rate'] >= MIN_WIN_RATE) &
        (results_df['p_value'] <= MAX_P_VALUE)
    ].copy()
    
    if winners.empty:
        logger.info("No statistically significant winners found")
        return {}
    
    # For each symbol, pick the best strategy (highest win rate)
    for symbol in winners['symbol'].unique():
        symbol_winners = winners[winners['symbol'] == symbol]
        best = symbol_winners.loc[symbol_winners['win_rate'].idxmax()]
        
        # Determine direction override
        if best['strategy'] == 'ghost_inverse':
            direction_override = DIRECTION_FLIP
        else:
            direction_override = None
        
        validated[symbol] = {
            'strategy': best['strategy'],
            'direction_override': direction_override,
            'hold_hours': int(best['holding_hours']),
            'win_rate': round(best['win_rate'], 3),
            'sample_size': int(best['total_trades']),
            'p_value': round(best['p_value'], 4),
            'confidence_interval': (round(best['ci_lower'], 3), round(best['ci_upper'], 3)),
        }
    
    return validated


# =============================================================================
# COMPARE & UPDATE
# =============================================================================

def load_current_strategies() -> Dict:
    """Load current V3_VALIDATED_STRATEGIES from config"""
    try:
        from config.symbols import v3_strategies_as_dicts
        return v3_strategies_as_dicts()
    except Exception:
        return {}


def compare_strategies(current: Dict, new: Dict) -> Dict:
    """
    Compare current vs new strategies and identify changes.
    
    Returns:
        {
            'added': {symbol: config},
            'removed': {symbol: reason},
            'changed': {symbol: {'old': ..., 'new': ...}},
            'unchanged': [symbols]
        }
    """
    changes = {
        'added': {},
        'removed': {},
        'changed': {},
        'unchanged': []
    }
    
    current_symbols = set(current.keys())
    new_symbols = set(new.keys())
    
    # Added symbols
    for symbol in new_symbols - current_symbols:
        changes['added'][symbol] = new[symbol]
    
    # Removed symbols - BUT preserve crypto if we only tested stocks
    # This prevents crypto symbols from being removed when we can't backtest them
    CRYPTO_PRESERVE = {'ETH', 'XRP', 'LINK', 'CHZ', 'BTC', 'SOL', 'AVAX', 'DOGE', 'ADA'}
    
    for symbol in current_symbols - new_symbols:
        # Don't remove crypto symbols just because they weren't in the new backtest
        if symbol in CRYPTO_PRESERVE:
            changes['unchanged'].append(symbol)
            logger.info(f"[CALIBRATE] Preserving crypto symbol {symbol} (not tested)")
        else:
            changes['removed'][symbol] = "No longer statistically significant"
    
    # Changed or unchanged
    for symbol in current_symbols & new_symbols:
        old = current[symbol]
        new_config = new[symbol]
        
        # Compare key fields
        old_hold = old.get('hold_hours', old.get('holding_hours', 0))
        new_hold = new_config.get('hold_hours', 0)
        
        old_strategy = old.get('strategy', '')
        new_strategy = new_config.get('strategy', '')
        
        old_wr = old.get('win_rate', 0)
        new_wr = new_config.get('win_rate', 0)
        
        if old_hold != new_hold or old_strategy != new_strategy or abs(old_wr - new_wr) > 0.03:
            changes['changed'][symbol] = {
                'old': {'strategy': old_strategy, 'hold_hours': old_hold, 'win_rate': old_wr},
                'new': {'strategy': new_strategy, 'hold_hours': new_hold, 'win_rate': new_wr}
            }
        else:
            changes['unchanged'].append(symbol)
    
    return changes


def format_changes_alert(changes: Dict) -> str:
    """Format changes as a readable alert message"""
    lines = []
    lines.append("🔄 GHOST AUTO-CALIBRATION REPORT")
    lines.append(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    if changes['added']:
        lines.append("🆕 NEW VALIDATED STRATEGIES:")
        for symbol, config in changes['added'].items():
            lines.append(f"  + {symbol}: {config['strategy']} @ {config['hold_hours']}h ({config['win_rate']:.1%} win rate, p={config['p_value']})")
        lines.append("")
    
    if changes['removed']:
        lines.append("❌ REMOVED (no longer significant):")
        for symbol, reason in changes['removed'].items():
            lines.append(f"  - {symbol}: {reason}")
        lines.append("")
    
    if changes['changed']:
        lines.append("📈 UPDATED PARAMETERS:")
        for symbol, info in changes['changed'].items():
            old = info['old']
            new = info['new']
            lines.append(f"  ~ {symbol}: {old['strategy']}@{old['hold_hours']}h ({old['win_rate']:.1%}) → {new['strategy']}@{new['hold_hours']}h ({new['win_rate']:.1%})")
        lines.append("")
    
    if changes['unchanged']:
        lines.append(f"✓ UNCHANGED: {', '.join(changes['unchanged'])}")
        lines.append("")
    
    total_changes = len(changes['added']) + len(changes['removed']) + len(changes['changed'])
    if total_changes == 0:
        lines.append("No changes needed - all strategies still optimal!")
    else:
        lines.append(f"Total changes: {total_changes}")
    
    return "\n".join(lines)


def save_calibration_results(
    crypto_results: pd.DataFrame,
    stock_results: pd.DataFrame,
    validated: Dict,
    changes: Dict
):
    """Save calibration results to disk"""
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save raw results
    if not crypto_results.empty:
        crypto_results.to_csv(CALIBRATION_DIR / f"crypto_backtest_{timestamp}.csv", index=False)
    
    if not stock_results.empty:
        stock_results.to_csv(CALIBRATION_DIR / f"stock_backtest_{timestamp}.csv", index=False)
    
    # Save validated strategies
    with open(CALIBRATION_DIR / f"validated_strategies_{timestamp}.json", 'w') as f:
        json.dump(validated, f, indent=2)
    
    # Save changes
    with open(CALIBRATION_DIR / f"changes_{timestamp}.json", 'w') as f:
        json.dump(changes, f, indent=2, default=str)
    
    # Save latest (overwrite)
    with open(CALIBRATION_DIR / "latest_validated.json", 'w') as f:
        json.dump(validated, f, indent=2)
    
    logger.info(f"Saved calibration results to {CALIBRATION_DIR}")


def generate_config_code(validated: Dict) -> str:
    """Generate Python code for V3_VALIDATED_STRATEGIES"""
    lines = []
    lines.append("# Auto-generated by auto_calibrate.py")
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("V3_VALIDATED_STRATEGIES = {")
    
    # Sort by asset type (crypto first, then stocks)
    crypto = {k: v for k, v in validated.items() if k in CRYPTO_SYMBOLS}
    stocks = {k: v for k, v in validated.items() if k in STOCK_SYMBOLS}
    
    if crypto:
        lines.append("    # CRYPTO")
        for symbol, config in sorted(crypto.items()):
            lines.append(f"    '{symbol}': {{")
            lines.append(f"        'strategy': '{config['strategy']}',")
            lines.append(f"        'direction_override': {repr(config['direction_override'])},")
            lines.append(f"        'hold_hours': {config['hold_hours']},")
            lines.append(f"        'win_rate': {config['win_rate']},")
            lines.append(f"        'sample_size': {config['sample_size']},")
            lines.append(f"        'p_value': {config['p_value']},")
            lines.append(f"        'confidence_interval': {config['confidence_interval']},")
            lines.append(f"    }},")
    
    if stocks:
        lines.append("    # STOCKS")
        for symbol, config in sorted(stocks.items()):
            lines.append(f"    '{symbol}': {{")
            lines.append(f"        'strategy': '{config['strategy']}',")
            lines.append(f"        'direction_override': {repr(config['direction_override'])},")
            lines.append(f"        'hold_hours': {config['hold_hours']},")
            lines.append(f"        'win_rate': {config['win_rate']},")
            lines.append(f"        'sample_size': {config['sample_size']},")
            lines.append(f"        'p_value': {config['p_value']},")
            lines.append(f"        'asset_type': 'stock',")
            lines.append(f"    }},")
    
    lines.append("}")
    
    return "\n".join(lines)


# =============================================================================
# MAIN CALIBRATION
# =============================================================================

def run_calibration(
    test_crypto: bool = True,
    test_stocks: bool = True,
    auto_update: bool = False,
    dry_run: bool = True
) -> Dict:
    """
    Run full calibration process.
    
    Args:
        test_crypto: Whether to test crypto symbols
        test_stocks: Whether to test stock symbols
        auto_update: Whether to automatically update config files
        dry_run: If True, don't make any changes (just report)
    
    Returns:
        Dict with calibration results
    """
    logger.info("=" * 60)
    logger.info("GHOST AUTO-CALIBRATION STARTING")
    logger.info("=" * 60)
    
    all_results = []
    
    # Test crypto
    if test_crypto:
        logger.info("\n📊 Testing CRYPTO symbols...")
        crypto_results = run_all_backtests(CRYPTO_SYMBOLS, 'crypto')
        all_results.append(crypto_results)
        logger.info(f"Crypto: {len(crypto_results)} results")
    else:
        crypto_results = pd.DataFrame()
    
    # Test stocks
    if test_stocks:
        logger.info("\n📊 Testing STOCK symbols...")
        stock_results = run_all_backtests(STOCK_SYMBOLS, 'stock')
        all_results.append(stock_results)
        logger.info(f"Stocks: {len(stock_results)} results")
    else:
        stock_results = pd.DataFrame()
    
    # Combine results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
    else:
        combined = pd.DataFrame()
    
    # Find validated strategies
    logger.info("\n🔍 Finding validated strategies...")
    validated = find_validated_strategies(combined)
    logger.info(f"Found {len(validated)} validated strategies")
    
    # Compare with current
    current = load_current_strategies()
    changes = compare_strategies(current, validated)
    
    # Generate alert
    alert = format_changes_alert(changes)
    print("\n" + alert)
    
    # Save results
    save_calibration_results(crypto_results, stock_results, validated, changes)
    
    # Generate config code
    config_code = generate_config_code(validated)
    config_path = CALIBRATION_DIR / "generated_config.py"
    config_path.write_text(config_code)
    logger.info(f"\nGenerated config saved to: {config_path}")
    
    # Always send Telegram alert with findings for human review
    try:
        send_calibration_alert(alert, changes)
        logger.info("📩 Calibration alert sent to Telegram for review")
    except Exception as e:
        logger.warning(f"Failed to send calibration alert: {e}")
    
    if not dry_run and auto_update:
        logger.info("\n⚠️ AUTO-UPDATE is enabled - updating config files...")
        update_success = auto_update_config(validated, changes)
        
        if update_success:
            # Auto-commit and push (requires git credentials)
            auto_deploy(changes)
    
    return {
        'validated': validated,
        'changes': changes,
        'alert': alert,
        'crypto_results': crypto_results,
        'stock_results': stock_results,
    }


# =============================================================================
# AUTO-UPDATE FUNCTIONS
# =============================================================================

def auto_update_config(validated: Dict, changes: Dict) -> bool:
    """
    Automatically update V3_VALIDATED_STRATEGIES in config/symbols.py.

    The canonical source of truth is config/symbols.py which uses frozen
    ValidatedStrategy dataclasses.  ghost_notifications.py auto-converts
    these at import time via v3_strategies_as_dicts().
    """
    import re

    config_file = Path(__file__).parent.parent / "config" / "symbols.py"

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        return False

    # Generate new V3_VALIDATED_STRATEGIES block (dataclass format)
    new_block = _generate_dataclass_block(validated)

    # Read current file
    content = config_file.read_text()

    # Match the V3_VALIDATED_STRATEGIES dict from opening brace to its
    # closing brace + newline.  The dict spans many lines with nested
    # ValidatedStrategy(...) calls.
    pattern = (
        r'(V3_VALIDATED_STRATEGIES:\s*Dict\[str,\s*ValidatedStrategy\]\s*=\s*\{)'
        r'(.*?)'
        r'(^\})'
    )
    m = re.search(pattern, content, re.DOTALL | re.MULTILINE)
    if not m:
        logger.error("Could not find V3_VALIDATED_STRATEGIES dict in config/symbols.py")
        return False

    # Replace with new block
    new_content = content[:m.start()] + new_block + content[m.end():]

    # Backup original
    backup_path = CALIBRATION_DIR / f"symbols_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    backup_path.write_text(content)
    logger.info(f"Backup saved to: {backup_path}")

    # Write updated file
    config_file.write_text(new_content)
    logger.info(f"Updated canonical config: {config_file}")

    return True


def generate_v3_block(validated: Dict) -> str:
    """Generate the legacy dict-of-dicts V3_VALIDATED_STRATEGIES string.

    Kept for backwards compatibility with generate_config_code().
    For the canonical dataclass format, use _generate_dataclass_block().
    """
    return _generate_dataclass_block(validated)


def _generate_dataclass_block(validated: Dict) -> str:
    """Generate ValidatedStrategy dataclass entries for config/symbols.py."""
    lines = []
    lines.append("V3_VALIDATED_STRATEGIES: Dict[str, ValidatedStrategy] = {")

    # Separate crypto and stocks
    crypto = {k: v for k, v in validated.items() if v.get('asset_type') != 'stock' and k in CRYPTO_SYMBOLS}
    stocks = {k: v for k, v in validated.items() if v.get('asset_type') == 'stock' or k in STOCK_SYMBOLS}

    def _direction(config: Dict) -> str:
        strat = config['strategy']
        if strat in ('ghost_inverse', 'ghost_inverse_strong'):
            return "DIRECTION_FLIP"
        elif strat == 'always_down':
            return "'DOWN'"
        elif strat == 'always_up':
            return "'UP'"
        return 'None'

    def _emit(symbol: str, config: Dict, is_stock: bool = False) -> None:
        ci = config.get('confidence_interval', (0.50, 0.60))
        wr = config['win_rate']
        lines.append(f"    # {symbol} {config['strategy']} @ {config['hold_hours']}h: "
                     f"{wr*100:.1f}% win rate, {config['sample_size']} trades, p={config['p_value']}")
        lines.append(f"    '{symbol}': ValidatedStrategy(")
        lines.append(f"        symbol='{symbol}',")
        lines.append(f"        strategy='{config['strategy']}',")
        lines.append(f"        direction_override={_direction(config)},")
        lines.append(f"        hold_hours={config['hold_hours']},")
        lines.append(f"        backtest_win_rate={wr},")
        lines.append(f"        backtest_trades={config['sample_size']},")
        lines.append(f"        p_value={config['p_value']},")
        lines.append(f"        confidence_interval={ci},")
        if is_stock:
            lines.append(f"        asset_type='stock',")
        lines.append(f"    ),")

    if crypto:
        lines.append("    # =========================================================================")
        lines.append("    # CRYPTO - Auto-calibrated " + datetime.now().strftime('%Y-%m-%d'))
        lines.append("    # =========================================================================")
        for symbol in sorted(crypto.keys()):
            _emit(symbol, crypto[symbol], is_stock=False)

    if stocks:
        lines.append("    # =========================================================================")
        lines.append("    # STOCKS - Auto-calibrated " + datetime.now().strftime('%Y-%m-%d'))
        lines.append("    # =========================================================================")
        for symbol in sorted(stocks.keys()):
            _emit(symbol, stocks[symbol], is_stock=True)

    lines.append("}")
    return "\n".join(lines)


def send_calibration_alert(alert: str, changes: Dict) -> bool:
    """Send calibration results to Telegram"""
    try:
        import requests
        
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        chat_id = os.environ.get('TELEGRAM_CHAT_ID')
        
        if not bot_token or not chat_id:
            logger.warning("Telegram credentials not set, skipping alert")
            return False
        
        # Truncate if too long
        if len(alert) > 4000:
            alert = alert[:3900] + "\n\n... (truncated)"
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': alert,
            'parse_mode': 'HTML'
        }
        
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Telegram alert sent")
            return True
        else:
            logger.error(f"Telegram error: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send Telegram alert: {e}")
        return False


def auto_deploy(changes: Dict) -> bool:
    """Auto-commit and push changes to trigger Railway deploy"""
    import subprocess
    
    total_changes = len(changes['added']) + len(changes['removed']) + len(changes['changed'])
    
    if total_changes == 0:
        logger.info("No changes to deploy")
        return True
    
    try:
        # Stage changes
        subprocess.run(['git', 'add', '-A'], check=True, cwd=Path(__file__).parent.parent)
        
        # Create commit message
        msg_parts = []
        if changes['added']:
            msg_parts.append(f"Added: {', '.join(changes['added'].keys())}")
        if changes['removed']:
            msg_parts.append(f"Removed: {', '.join(changes['removed'].keys())}")
        if changes['changed']:
            msg_parts.append(f"Updated: {', '.join(changes['changed'].keys())}")
        
        commit_msg = f"Auto-calibration: {'; '.join(msg_parts)}"
        
        # Commit
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True, cwd=Path(__file__).parent.parent)
        
        # Push
        subprocess.run(['git', 'push', 'origin', 'main'], check=True, cwd=Path(__file__).parent.parent)
        
        logger.info(f"✅ Auto-deployed: {commit_msg}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Auto-deploy failed: {e}")
        return False


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ghost Auto-Calibration System')
    parser.add_argument('--crypto-only', action='store_true', help='Only test crypto')
    parser.add_argument('--stocks-only', action='store_true', help='Only test stocks')
    parser.add_argument('--auto-update', action='store_true', help='Auto-update config files')
    parser.add_argument('--apply', action='store_true', help='Apply changes (not dry-run)')
    args = parser.parse_args()
    
    test_crypto = not args.stocks_only
    test_stocks = not args.crypto_only
    
    result = run_calibration(
        test_crypto=test_crypto,
        test_stocks=test_stocks,
        auto_update=args.auto_update,
        dry_run=not args.apply
    )
    
    print("\n" + "=" * 60)
    print("VALIDATED STRATEGIES:")
    print("=" * 60)
    for symbol, config in sorted(result['validated'].items()):
        print(f"  {symbol}: {config['strategy']} @ {config['hold_hours']}h ({config['win_rate']:.1%}, p={config['p_value']})")
