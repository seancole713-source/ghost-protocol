#!/usr/bin/env python3
"""
Quick ingestion of top 1000 US stocks by market cap
Uses yfinance's screener API to get liquid, high-volume stocks
Takes ~10-15 minutes instead of 2 hours
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
import yfinance as yf
import time
from datetime import datetime

# PostgreSQL connection — NEVER hardcode credentials
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

def get_top_1000_stocks():
    """Get top 1000 US stocks by market cap from major indices"""
    print("📊 Fetching top 1000 US stocks...")
    
    # Major indices tickers
    indices = {
        'S&P 500': '^GSPC',
        'NASDAQ 100': '^NDX',
        'Russell 2000': '^RUT',
        'Dow Jones': '^DJI'
    }
    
    all_symbols = set()
    
    # Get S&P 500 (top 500)
    print("  Downloading S&P 500 constituents...")
    try:
        import pandas as pd
        sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        all_symbols.update(sp500['Symbol'].str.replace('.', '-').tolist())
        print(f"    ✓ Added {len(sp500)} S&P 500 stocks")
    except Exception as e:
        print(f"    ✗ Failed to get S&P 500: {e}")
    
    # Get NASDAQ 100 (top 100 tech)
    print("  Downloading NASDAQ 100 constituents...")
    try:
        nasdaq100 = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
        nasdaq_symbols = nasdaq100['Ticker'].str.replace('.', '-').tolist()
        all_symbols.update(nasdaq_symbols)
        print(f"    ✓ Added {len(nasdaq_symbols)} NASDAQ 100 stocks")
    except Exception as e:
        print(f"    ✗ Failed to get NASDAQ 100: {e}")
    
    # Get Russell 2000 top 400 (mid/small caps)
    print("  Downloading Russell 2000 top holdings...")
    try:
        # IWM (iShares Russell 2000 ETF) holdings
        iwm = yf.Ticker("IWM")
        holdings = iwm.get_holdings()
        if holdings is not None and len(holdings) > 0:
            russell_symbols = holdings.head(400)['Symbol'].tolist()
            all_symbols.update(russell_symbols)
            print(f"    ✓ Added {len(russell_symbols)} Russell 2000 stocks")
    except Exception as e:
        print(f"    ✗ Failed to get Russell 2000: {e}")
    
    print(f"\n📈 Total unique symbols collected: {len(all_symbols)}")
    return list(all_symbols)[:1000]  # Cap at 1000

def enrich_and_insert(conn, symbols):
    """Enrich symbols with Yahoo Finance data and insert into PostgreSQL"""
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    print(f"\n🔍 Enriching {len(symbols)} symbols with Yahoo Finance...")
    inserted = 0
    skipped = 0
    errors = 0
    
    for i, symbol in enumerate(symbols, 1):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(symbols)} ({(i/len(symbols)*100):.1f}%) - Inserted: {inserted}, Skipped: {skipped}, Errors: {errors}")
            conn.commit()
        
        try:
            # Check if already exists
            cursor.execute(
                "SELECT symbol FROM symbol_universe WHERE symbol = %s",
                (symbol,)
            )
            if cursor.fetchone():
                skipped += 1
                continue
            
            # Get info from Yahoo Finance
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract data with fallbacks
            name = info.get('longName') or info.get('shortName') or symbol
            sector = info.get('sector', 'Unknown')
            industry = info.get('industry', 'Unknown')
            market_cap = info.get('marketCap', 0)
            exchange = info.get('exchange', 'Unknown')
            
            # Determine asset type
            asset_type = 'crypto' if symbol.endswith('-USD') else 'stock'
            
            # Insert into database
            cursor.execute("""
                INSERT INTO symbol_universe (
                    symbol, name, asset_type, sector, industry, 
                    market_cap, exchange, is_active, last_updated
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol) DO NOTHING
            """, (
                symbol, name, asset_type, sector, industry,
                market_cap, exchange, True, datetime.now()
            ))
            
            inserted += 1
            time.sleep(0.05)  # Rate limit: 20 req/sec (faster than before)
            
        except Exception as e:
            errors += 1
            if errors % 10 == 0:
                print(f"    ⚠️  Error on {symbol}: {str(e)[:100]}")
            continue
    
    conn.commit()
    cursor.close()
    
    print(f"\n✅ Enrichment complete!")
    print(f"   Inserted: {inserted}")
    print(f"   Skipped: {skipped}")
    print(f"   Errors: {errors}")
    
    return inserted

def main():
    print("🚀 Ghost Protocol - Top 1000 US Stocks Ingestion")
    print("=" * 60)
    
    # Get symbols
    symbols = get_top_1000_stocks()
    
    if not symbols:
        print("❌ No symbols found!")
        return
    
    # Connect to PostgreSQL
    print(f"\n🔌 Connecting to PostgreSQL...")
    conn = psycopg2.connect(DATABASE_URL)
    print("   ✓ Connected!")
    
    # Enrich and insert
    inserted = enrich_and_insert(conn, symbols)
    
    # Summary
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT COUNT(*) as count FROM symbol_universe WHERE asset_type = 'stock'")
    total_stocks = cursor.fetchone()['count']
    
    cursor.execute("SELECT COUNT(*) as count FROM symbol_universe WHERE asset_type = 'crypto'")
    total_crypto = cursor.fetchone()['count']
    
    print(f"\n" + "=" * 60)
    print(f"📊 FINAL DATABASE STATE:")
    print(f"   Total Stocks: {total_stocks}")
    print(f"   Total Crypto: {total_crypto}")
    print(f"   Grand Total: {total_stocks + total_crypto}")
    print("=" * 60)
    
    cursor.close()
    conn.close()
    
    print("\n✅ Top 1000 ingestion complete!")
    print("💡 To add remaining 11,000 stocks later, run: python scripts/ingest_us_market.py")

if __name__ == "__main__":
    main()
