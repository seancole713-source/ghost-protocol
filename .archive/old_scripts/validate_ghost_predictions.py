"""
GHOST PROTOCOL - PREDICTION VALIDATION AGAINST REAL MARKET DATA
================================================================
This script pulls all Ghost predictions and validates them against
actual market prices to calculate TRUE accuracy.

NO HACKS. NO ASSUMPTIONS. JUST REAL DATA.

RUN THIS IN YOUR GHOST CODESPACE:
    cd /workspaces/ghost-protocol
    python3 validate_ghost_predictions.py
"""

import os
import sys

# Add ghost-protocol to path if running from codespace
sys.path.insert(0, '/workspaces/ghost-protocol')
sys.path.insert(0, '.')

import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import requests

# Try to import database libraries
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    print("❌ psycopg2 not installed - cannot connect to PostgreSQL")
    sys.exit(1)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL not set!")
    print("Set it with: export DATABASE_URL='postgresql://...'")
    sys.exit(1)

print("=" * 80)
print("🔬 GHOST PROTOCOL - REAL ACCURACY VALIDATION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# PRICE FETCHING FUNCTIONS
# ============================================================================

def get_crypto_price(symbol: str) -> Optional[float]:
    """Get current crypto price from multiple sources"""
    symbol = symbol.upper()
    
    # Map common symbols to CoinGecko IDs
    coingecko_ids = {
        'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
        'DOGE': 'dogecoin', 'XRP': 'ripple', 'ADA': 'cardano',
        'AVAX': 'avalanche-2', 'DOT': 'polkadot', 'MATIC': 'matic-network',
        'LINK': 'chainlink', 'UNI': 'uniswap', 'ATOM': 'cosmos',
        'LTC': 'litecoin', 'BCH': 'bitcoin-cash', 'XLM': 'stellar',
        'ALGO': 'algorand', 'VET': 'vechain', 'FIL': 'filecoin',
        'AAVE': 'aave', 'EOS': 'eos', 'XTZ': 'tezos',
        'THETA': 'theta-token', 'XMR': 'monero', 'NEO': 'neo',
        'MKR': 'maker', 'COMP': 'compound-governance-token',
        'SNX': 'synthetix-network-token', 'YFI': 'yearn-finance',
        'SUSHI': 'sushi', 'CRV': 'curve-dao-token',
    }
    
    cg_id = coingecko_ids.get(symbol, symbol.lower())
    
    # Try CoinGecko
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if cg_id in data:
                return float(data[cg_id]['usd'])
    except:
        pass
    
    # Try Coinbase
    try:
        url = f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return float(data['data']['amount'])
    except:
        pass
    
    # Try Binance
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return float(data['price'])
    except:
        pass
    
    return None


def get_stock_price(symbol: str) -> Optional[float]:
    """Get current stock price"""
    symbol = symbol.upper()
    
    # Try Alpha Vantage
    av_key = os.getenv("ALPHAVANTAGE_API_KEY", "3WNNLA81KS7BG4AK")
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={av_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'Global Quote' in data and '05. price' in data['Global Quote']:
                return float(data['Global Quote']['05. price'])
    except:
        pass
    
    # Try Polygon
    polygon_key = os.getenv("POLYGON_API_KEY", "8VIvELVXiLG30K2l1348RzSurffLM0jR")
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={polygon_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if 'results' in data and len(data['results']) > 0:
                return float(data['results'][0]['c'])
    except:
        pass
    
    return None


def get_current_price(symbol: str) -> Optional[float]:
    """Get current price for any symbol (crypto or stock)"""
    # Common crypto symbols
    crypto_symbols = {
        'BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'AVAX', 'DOT', 'MATIC',
        'LINK', 'UNI', 'ATOM', 'LTC', 'BCH', 'XLM', 'ALGO', 'VET', 'FIL',
        'AAVE', 'EOS', 'XTZ', 'THETA', 'XMR', 'NEO', 'MKR', 'COMP', 'SNX',
        'YFI', 'SUSHI', 'CRV', 'SAND', 'MANA', 'AXS', 'ENJ', 'GALA', 'APE',
        'SHIB', 'PEPE', 'FLOKI', 'BONK', 'WIF', 'INJ', 'TIA', 'SEI', 'SUI',
        'ARB', 'OP', 'IMX', 'RNDR', 'FET', 'AGIX', 'OCEAN', 'GRT', 'AR',
    }
    
    symbol = symbol.upper()
    
    if symbol in crypto_symbols or symbol.endswith('USD') or symbol.endswith('USDT'):
        return get_crypto_price(symbol.replace('USD', '').replace('T', ''))
    else:
        return get_stock_price(symbol)


def get_historical_price(symbol: str, timestamp: float) -> Optional[float]:
    """Get historical price at a specific timestamp"""
    # For crypto, use CryptoCompare
    crypto_symbols = {'BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'AVAX', 'DOT', 'MATIC', 'LINK'}
    
    if symbol.upper() in crypto_symbols:
        try:
            url = f"https://min-api.cryptocompare.com/data/pricehistorical"
            params = {'fsym': symbol.upper(), 'tsyms': 'USD', 'ts': int(timestamp)}
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return float(data[symbol.upper()]['USD'])
        except:
            pass
    
    return None


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_db_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(DATABASE_URL)


def get_all_predictions(days: int = 30, limit: int = 1000) -> List[Dict]:
    """Get all predictions from the database"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT 
            id,
            symbol,
            direction,
            confidence,
            price_at_prediction,
            prediction_time,
            horizon_hours,
            created_at
        FROM ghost_predictions
        WHERE prediction_time > NOW() - INTERVAL '%s days'
        ORDER BY prediction_time DESC
        LIMIT %s
    """, (days, limit))
    
    predictions = [dict(row) for row in cur.fetchall()]
    conn.close()
    
    return predictions


def get_prediction_outcomes() -> Dict[int, Dict]:
    """Get existing outcomes"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT 
            prediction_id,
            was_correct,
            actual_change_pct,
            exit_price
        FROM ghost_prediction_outcomes
    """)
    
    outcomes = {row['prediction_id']: dict(row) for row in cur.fetchall()}
    conn.close()
    
    return outcomes


# ============================================================================
# VALIDATION LOGIC
# ============================================================================

def validate_prediction(pred: Dict, current_price: float) -> Dict:
    """Validate a single prediction against current price"""
    entry_price = float(pred['price_at_prediction'])
    direction = pred['direction'].upper()
    
    # Calculate actual change
    if entry_price > 0:
        change_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        change_pct = 0
    
    # Determine if prediction was correct
    if direction == 'UP':
        was_correct = current_price > entry_price
    elif direction == 'DOWN':
        was_correct = current_price < entry_price
    else:  # FLAT/HOLD
        was_correct = abs(change_pct) < 2.0  # Within 2%
    
    return {
        'prediction_id': pred['id'],
        'symbol': pred['symbol'],
        'direction': direction,
        'confidence': float(pred['confidence']),
        'entry_price': entry_price,
        'current_price': current_price,
        'change_pct': change_pct,
        'was_correct': was_correct,
        'prediction_time': pred['prediction_time'],
    }


def run_validation():
    """Run full validation of all predictions"""
    print("\n" + "=" * 80)
    print("FETCHING PREDICTIONS FROM DATABASE")
    print("=" * 80)
    
    # Get predictions
    predictions = get_all_predictions(days=30, limit=500)
    print(f"Found {len(predictions)} predictions in last 30 days")
    
    if len(predictions) == 0:
        print("❌ No predictions found!")
        return
    
    # Get existing outcomes
    existing_outcomes = get_prediction_outcomes()
    print(f"Found {len(existing_outcomes)} existing outcomes")
    
    # Group predictions by symbol for efficient price fetching
    symbols = set(p['symbol'] for p in predictions)
    print(f"Unique symbols: {len(symbols)}")
    
    # Fetch current prices
    print("\n" + "=" * 80)
    print("FETCHING CURRENT PRICES")
    print("=" * 80)
    
    current_prices = {}
    for symbol in symbols:
        price = get_current_price(symbol)
        if price:
            current_prices[symbol] = price
            print(f"  ✅ {symbol}: ${price:,.2f}")
        else:
            print(f"  ❌ {symbol}: Failed to get price")
        time.sleep(0.1)  # Rate limiting
    
    print(f"\nGot prices for {len(current_prices)}/{len(symbols)} symbols")
    
    # Validate predictions
    print("\n" + "=" * 80)
    print("VALIDATING PREDICTIONS")
    print("=" * 80)
    
    results = []
    for pred in predictions:
        symbol = pred['symbol']
        if symbol not in current_prices:
            continue
        
        # Skip predictions less than 48 hours old (not yet resolved)
        pred_time = pred['prediction_time']
        if isinstance(pred_time, str):
            pred_time = datetime.fromisoformat(pred_time.replace('Z', '+00:00'))
        
        age_hours = (datetime.now(pred_time.tzinfo) - pred_time).total_seconds() / 3600
        if age_hours < 48:
            continue  # Too recent
        
        result = validate_prediction(pred, current_prices[symbol])
        results.append(result)
    
    print(f"Validated {len(results)} predictions (48h+ old)")
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("ACCURACY RESULTS")
    print("=" * 80)
    
    if len(results) == 0:
        print("❌ No predictions old enough to validate (need 48h+)")
        return
    
    total = len(results)
    correct = sum(1 for r in results if r['was_correct'])
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print(f"\n📊 OVERALL ACCURACY: {accuracy:.1f}%")
    print(f"   Correct: {correct}/{total}")
    print()
    
    # By direction
    print("📈 BY DIRECTION:")
    for direction in ['UP', 'DOWN', 'FLAT']:
        dir_results = [r for r in results if r['direction'] == direction]
        if len(dir_results) > 0:
            dir_correct = sum(1 for r in dir_results if r['was_correct'])
            dir_acc = (dir_correct / len(dir_results)) * 100
            print(f"   {direction}: {dir_acc:.1f}% ({dir_correct}/{len(dir_results)})")
    
    # By confidence level
    print("\n📊 BY CONFIDENCE:")
    conf_buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for low, high in conf_buckets:
        bucket_results = [r for r in results if low <= r['confidence'] < high]
        if len(bucket_results) > 0:
            bucket_correct = sum(1 for r in bucket_results if r['was_correct'])
            bucket_acc = (bucket_correct / len(bucket_results)) * 100
            print(f"   {low:.0%}-{high:.0%}: {bucket_acc:.1f}% ({bucket_correct}/{len(bucket_results)})")
    
    # By symbol (top 10)
    print("\n📈 TOP SYMBOLS BY TRADE COUNT:")
    from collections import Counter
    symbol_counts = Counter(r['symbol'] for r in results)
    for symbol, count in symbol_counts.most_common(15):
        sym_results = [r for r in results if r['symbol'] == symbol]
        sym_correct = sum(1 for r in sym_results if r['was_correct'])
        sym_acc = (sym_correct / len(sym_results)) * 100
        print(f"   {symbol:8}: {sym_acc:5.1f}% ({sym_correct}/{len(sym_results)})")
    
    # Recent vs older predictions
    print("\n📅 BY TIME PERIOD:")
    now = datetime.now()
    for days_ago, label in [(7, "Last 7 days"), (14, "8-14 days ago"), (30, "15-30 days ago")]:
        if days_ago == 7:
            period_results = [r for r in results if r['prediction_time'] and 
                           (now - r['prediction_time'].replace(tzinfo=None)).days <= 7]
        elif days_ago == 14:
            period_results = [r for r in results if r['prediction_time'] and 
                           7 < (now - r['prediction_time'].replace(tzinfo=None)).days <= 14]
        else:
            period_results = [r for r in results if r['prediction_time'] and 
                           14 < (now - r['prediction_time'].replace(tzinfo=None)).days <= 30]
        
        if len(period_results) > 0:
            period_correct = sum(1 for r in period_results if r['was_correct'])
            period_acc = (period_correct / len(period_results)) * 100
            print(f"   {label}: {period_acc:.1f}% ({period_correct}/{len(period_results)})")
    
    # Sample of recent predictions
    print("\n📋 SAMPLE PREDICTIONS (most recent):")
    print("-" * 100)
    print(f"{'Symbol':<8} {'Direction':<8} {'Conf':<6} {'Entry':>12} {'Current':>12} {'Change':>8} {'Result':<8}")
    print("-" * 100)
    
    for r in sorted(results, key=lambda x: x['prediction_time'] or datetime.min, reverse=True)[:20]:
        result_str = "✅ WIN" if r['was_correct'] else "❌ LOSS"
        print(f"{r['symbol']:<8} {r['direction']:<8} {r['confidence']:>5.0%} "
              f"${r['entry_price']:>10,.2f} ${r['current_price']:>10,.2f} "
              f"{r['change_pct']:>+7.2f}% {result_str}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
📊 GHOST PROTOCOL REAL ACCURACY:

   Total Predictions Validated: {total}
   Correct Predictions: {correct}
   
   ═══════════════════════════════════
   ║  REAL ACCURACY: {accuracy:.1f}%  ║
   ═══════════════════════════════════
   
   Average Confidence: {sum(r['confidence'] for r in results)/len(results)*100:.1f}%
   Average Move: {sum(r['change_pct'] for r in results)/len(results):+.2f}%
   
   UP Predictions: {sum(1 for r in results if r['direction'] == 'UP')}
   DOWN Predictions: {sum(1 for r in results if r['direction'] == 'DOWN')}
""")
    
    # Interpretation
    print("\n🎯 INTERPRETATION:")
    if accuracy >= 60:
        print("   ✅ Model has significant edge (>60%)")
        print("   → Ghost is working correctly!")
    elif accuracy >= 55:
        print("   ✅ Model has slight edge (55-60%)")
        print("   → Ghost is profitable but could improve")
    elif accuracy >= 48:
        print("   ⚠️ Model is roughly random (48-55%)")
        print("   → Need better features or more training data")
    elif accuracy >= 35:
        print("   ❌ Model is anti-correlated (35-48%)")
        print("   → Consider INVERSE_GHOST=1 or retrain")
    else:
        print("   ❌ Model is severely anti-correlated (<35%)")
        print("   → INVERSE_GHOST=1 would give", f"{100-accuracy:.1f}% accuracy")
    
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        run_validation()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
