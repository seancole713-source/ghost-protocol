#!/usr/bin/env python3
"""
Reconcile historical predictions using CoinGecko API.
This bypasses Yahoo Finance issues and works for crypto.
"""
import sqlite3
import time
import requests
from datetime import datetime
import sys

# CoinGecko symbol mapping
COINGECKO_IDS = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
    'ADA': 'cardano', 'XRP': 'ripple', 'SOL': 'solana',
    'DOT': 'polkadot', 'DOGE': 'dogecoin', 'AVAX': 'avalanche-2',
    'MATIC': 'matic-network', 'SHIB': 'shiba-inu', 'UNI': 'uniswap',
    'LTC': 'litecoin', 'ATOM': 'cosmos', 'AAVE': 'aave',
    'ALGO': 'algorand', 'FIL': 'filecoin', 'CRV': 'curve-dao-token',
    'SUSHI': 'sushi', 'YFI': 'yearn-finance', 'PEPE': 'pepe',
    'APT': 'aptos', 'ARB': 'arbitrum', 'AXS': 'axie-infinity',
    'FLOW': 'flow', 'SAND': 'the-sandbox', 'GALA': 'gala',
    'CHZ': 'chiliz', 'IMX': 'immutable-x', 'ICP': 'internet-computer',
    'HBAR': 'hedera-hashgraph', 'QNT': 'quant-network',
    'RUNE': 'thorchain', 'INJ': 'injective-protocol',
    'TIA': 'celestia', 'FTM': 'fantom', 'ETC': 'ethereum-classic',
    '1INCH': '1inch', 'BAL': 'balancer', 'LDO': 'lido-dao',
    'RPL': 'rocket-pool'
}

print("=" * 70)
print("GHOST ACCURACY AUDIT - HISTORICAL RECONCILIATION")
print("Using CoinGecko API for current crypto prices")
print("=" * 70)

# Connect to local database
db_path = "data/ghost_predictions.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Get predictions older than 48h
now = time.time()
cutoff = now - (48 * 3600)

cur.execute("""
    SELECT id, symbol, run_at, horizon_h, direction, confidence
    FROM predictions
    WHERE run_at < ?
    ORDER BY run_at DESC
""", (cutoff,))

predictions = cur.fetchall()

if not predictions:
    print("\n❌ No predictions found older than 48h")
    sys.exit(1)

print(f"\n📊 Found {len(predictions)} predictions older than 48h")
print(f"Date range: {datetime.fromtimestamp(predictions[-1][2])} → {datetime.fromtimestamp(predictions[0][2])}")

# Get unique symbols
symbols = set(p[1] for p in predictions)
crypto_symbols = [s for s in symbols if s in COINGECKO_IDS]
stock_symbols = [s for s in symbols if s not in COINGECKO_IDS]

print(f"\nCrypto: {len(crypto_symbols)} symbols")
print(f"Stocks: {len(stock_symbols)} symbols")

# Fetch current prices for crypto via CoinGecko
print("\n🔍 Fetching current crypto prices from CoinGecko...")
current_prices = {}

if crypto_symbols:
    try:
        ids = [COINGECKO_IDS[s] for s in crypto_symbols]
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={','.join(ids)}&vs_currencies=usd"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for symbol in crypto_symbols:
                cg_id = COINGECKO_IDS[symbol]
                if cg_id in data and 'usd' in data[cg_id]:
                    current_prices[symbol] = data[cg_id]['usd']
                    print(f"  ✓ {symbol}: ${current_prices[symbol]:.4f}")
        else:
            print(f"⚠️  CoinGecko API error: {response.status_code}")
    except Exception as e:
        print(f"⚠️  CoinGecko fetch failed: {e}")

if not current_prices:
    print("\n❌ No prices fetched - cannot reconcile")
    sys.exit(1)

print(f"\n✅ Fetched {len(current_prices)} prices")

# Reconcile predictions
print("\n" + "=" * 70)
print("RECONCILIATION")
print("=" * 70)

correct = 0
wrong = 0
no_data = 0

for pred_id, symbol, run_at, horizon_h, predicted_direction, confidence in predictions:
    # Skip if no current price
    if symbol not in current_prices:
        no_data += 1
        continue
    
    # Get price at prediction time from forecast points
    cur.execute("""
        SELECT price FROM prediction_points
        WHERE prediction_id = ? AND kind = 'forecast'
        ORDER BY ts ASC LIMIT 1
    """, (pred_id,))
    
    price_t0_row = cur.fetchone()
    if not price_t0_row:
        no_data += 1
        continue
    
    price_t0 = price_t0_row[0]
    price_t1 = current_prices[symbol]
    
    # Calculate movement
    realized_move_pct = ((price_t1 - price_t0) / price_t0) * 100
    
    # Determine actual direction (0.25% threshold)
    if abs(realized_move_pct) < 0.25:
        actual_direction = "FLAT"
    elif realized_move_pct > 0:
        actual_direction = "UP"
    else:
        actual_direction = "DOWN"
    
    # Check correctness
    is_correct = (predicted_direction == actual_direction)
    
    if is_correct:
        correct += 1
        status = "✅"
    else:
        wrong += 1
        status = "❌"
    
    # Show first 20
    if (correct + wrong) <= 20:
        age_days = (now - run_at) / 86400
        print(f"{status} {symbol:6} Pred:{predicted_direction:4} Act:{actual_direction:4} "
              f"Move:{realized_move_pct:+6.2f}% (t0=${price_t0:.4f} t1=${price_t1:.4f}) {age_days:.0f}d ago")

conn.close()

# Calculate accuracy
total = correct + wrong
if total > 0:
    accuracy = (correct / total) * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\n✅ Correct:  {correct:3}")
    print(f"❌ Wrong:    {wrong:3}")
    print(f"⚠️  No Data:  {no_data:3}")
    print(f"─" * 40)
    print(f"📊 ACCURACY: {accuracy:.1f}% ({correct}/{total})")
    
    # Confidence interval
    if total >= 30:
        import math
        p = correct / total
        z = 1.96
        se = math.sqrt(p * (1 - p) / total)
        margin = z * se
        ci_lower = max(0, (p - margin) * 100)
        ci_upper = min(100, (p + margin) * 100)
        
        print(f"📈 95% CI:   [{ci_lower:.1f}%, {ci_upper:.1f}%]")
        print(f"📏 Margin:   ±{margin*100:.1f}%")
        
        if ci_lower >= 70:
            print("\n🎯 GHOST MEETS 70% TARGET WITH STATISTICAL CONFIDENCE! ✅")
        elif accuracy >= 70:
            print(f"\n⚠️  Point estimate {accuracy:.1f}% ≥ 70%, but CI lower bound {ci_lower:.1f}% < 70%")
            gap = 70 - ci_lower
            needed = int((gap / (z * math.sqrt(p * (1 - p)))) ** 2)
            print(f"    Need ~{needed} more predictions at this accuracy for 70% CI")
        else:
            gap = 70 - accuracy
            print(f"\n❌ Below 70% target by {gap:.1f}%")
    else:
        print(f"\n⚠️  Only {total} samples - need 30+ for reliable confidence interval")
    
    print("\n" + "=" * 70)
    print("NOTES")
    print("=" * 70)
    print("• Using current prices (not historical t+48h prices)")
    print("• This is a proxy - actual accuracy may differ slightly")
    print("• Stocks excluded (need different price source)")
    print("• For production: use outcome_reconciler_v2.py with historical data")
    print("=" * 70)
else:
    print("\n❌ No predictions could be reconciled")
