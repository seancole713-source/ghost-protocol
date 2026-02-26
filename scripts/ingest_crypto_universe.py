#!/usr/bin/env python3
"""
Ingest top 500 crypto assets into PostgreSQL symbol_universe table.
Uses CoinGecko API (free tier - no API key needed).
Takes ~10-15 minutes with rate limiting.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import time

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

def get_top_crypto_from_coingecko(limit=500):
    """Fetch top crypto assets from CoinGecko API"""
    print(f"📡 Fetching top {limit} crypto assets from CoinGecko...")
    
    all_coins = []
    page = 1
    per_page = 250  # CoinGecko max per page
    
    while len(all_coins) < limit:
        try:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': per_page,
                'page': page,
                'sparkline': False
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            coins = response.json()
            
            if not coins:
                break
            
            all_coins.extend(coins)
            print(f"  ✓ Fetched page {page}: {len(coins)} coins (total: {len(all_coins)})")
            
            page += 1
            time.sleep(1.5)  # CoinGecko rate limit: ~50 calls/min free tier
            
        except Exception as e:
            print(f"  ✗ Error fetching page {page}: {e}")
            break
    
    return all_coins[:limit]

def insert_crypto_to_postgres(conn, coins):
    """Insert crypto assets into PostgreSQL symbol_universe table"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print(f"\n💾 Inserting {len(coins)} crypto assets into PostgreSQL...")
    inserted = 0
    updated = 0
    skipped = 0
    
    for i, coin in enumerate(coins, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(coins)} - Inserted: {inserted}, Updated: {updated}, Skipped: {skipped}")
            conn.commit()
        
        try:
            symbol = coin['symbol'].upper()
            name = coin['name']
            market_cap = coin.get('market_cap', 0) or 0
            current_price = coin.get('current_price', 0) or 0
            
            # Check if exists
            cursor.execute("SELECT id FROM symbol_universe WHERE symbol = %s", (symbol,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing crypto
                cursor.execute("""
                    UPDATE symbol_universe
                    SET name = %s,
                        market_cap = %s,
                        last_price = %s,
                        last_updated = %s,
                        metadata = %s
                    WHERE symbol = %s
                """, (
                    name,
                    market_cap,
                    current_price,
                    int(time.time()),
                    f"coingecko_id:{coin['id']}",
                    symbol
                ))
                updated += 1
            else:
                # Insert new crypto
                cursor.execute("""
                    INSERT INTO symbol_universe (
                        symbol, name, asset_type, sector, industry,
                        market_cap, exchange, is_active, last_price, last_updated, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    symbol,
                    name,
                    'crypto',
                    'Cryptocurrency',
                    coin.get('category', 'Unknown'),
                    market_cap,
                    'CoinGecko',
                    1,
                    current_price,
                    int(time.time()),
                    f"coingecko_id:{coin['id']}"
                ))
                inserted += 1
                
        except Exception as e:
            skipped += 1
            if skipped <= 5:  # Only show first 5 errors
                print(f"    ⚠️  Error on {coin.get('symbol', 'unknown')}: {str(e)[:100]}")
            continue
    
    conn.commit()
    cursor.close()
    
    print(f"\n✅ Crypto ingestion complete!")
    print(f"   Inserted: {inserted}")
    print(f"   Updated: {updated}")
    print(f"   Skipped: {skipped}")
    
    return inserted + updated

def main():
    print("🚀 Ghost Protocol - Top 500 Crypto Ingestion")
    print("=" * 60)
    
    # Fetch from CoinGecko
    coins = get_top_crypto_from_coingecko(limit=500)
    
    if not coins:
        print("❌ No crypto assets fetched!")
        return
    
    print(f"\n✓ Fetched {len(coins)} crypto assets")
    print(f"  Top 5: {', '.join([c['symbol'].upper() for c in coins[:5]])}")
    
    # Connect to PostgreSQL
    print(f"\n🔌 Connecting to PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    print("   ✓ Connected!")
    
    # Insert crypto
    total = insert_crypto_to_postgres(conn, coins)
    
    # Final stats
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT COUNT(*) as count FROM symbol_universe WHERE asset_type = 'stock'")
    total_stocks = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM symbol_universe WHERE asset_type = 'crypto'")
    total_crypto = cursor.fetchone()['count']
    
    # Get top 10 crypto by market cap
    cursor.execute("""
        SELECT symbol, name, market_cap
        FROM symbol_universe
        WHERE asset_type = 'crypto'
        ORDER BY market_cap DESC
        LIMIT 10
    """)
    top_crypto = cursor.fetchall()
    
    print("\n" + "=" * 60)
    print("📊 FINAL DATABASE STATE:")
    print(f"   Total Stocks: {total_stocks}")
    print(f"   Total Crypto: {total_crypto}")
    print(f"   Grand Total: {total_stocks + total_crypto}")
    print("\n🏆 Top 10 Crypto by Market Cap:")
    for coin in top_crypto:
        mc = coin['market_cap'] / 1_000_000_000  # Convert to billions
        print(f"   {coin['symbol']:8s} - {coin['name']:20s} ${mc:,.1f}B")
    print("=" * 60)
    
    cursor.close()
    conn.close()
    
    print("\n✅ Top 500 crypto ingestion complete!")
    print(f"💡 Ghost Protocol now tracks {total_stocks + total_crypto} symbols!")
    print(f"   📈 {total_stocks} stocks + 🪙 {total_crypto} crypto")

if __name__ == "__main__":
    main()
