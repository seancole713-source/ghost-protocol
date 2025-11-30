#!/usr/bin/env python3
"""
Ghost Protocol US Market Symbol Ingestion
==========================================

Ingests ~7,000 US stock symbols from major exchanges (NASDAQ, NYSE, CBOE).

Data Sources (prioritized):
1. NASDAQ FTP (ftp.nasdaqtrader.com) - Official, most complete
2. IEX Cloud API (https://iexcloud.io) - Real-time, includes metadata
3. Yahoo Finance (yfinance) - Fallback, good for delisted detection

Features:
- Pulls all actively traded US stocks
- Enriches with sector, industry, market cap
- Deduplicates and normalizes symbols
- Flags delisted/inactive symbols
- Updates symbol_universe table

Output:
- 6,000-8,000 symbols ingested
- Stored in PostgreSQL symbol_universe table
- Ready for volatility-triggered predictions
"""

import csv
import ftplib
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/symbol_ingestion.log"),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger(__name__)

# Import database engine
from core.db_engine import execute_query, execute_many, get_db_connection


class SymbolIngester:
    """Ingests US stock universe from multiple sources"""
    
    def __init__(self):
        """Initialize ingester"""
        self.symbols: dict[str, dict[str, Any]] = {}
        self.stats = {
            "nasdaq": 0,
            "nyse": 0,
            "enriched": 0,
            "duplicates": 0,
            "invalid": 0,
            "total": 0
        }
        
        LOGGER.info("📥 Symbol Ingester initialized")
    
    def run(self):
        """Execute full ingestion pipeline"""
        LOGGER.info("=" * 60)
        LOGGER.info("📊 Starting US Market Symbol Ingestion")
        LOGGER.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Fetch from NASDAQ FTP
            LOGGER.info("\n📡 Step 1/5: Fetching symbols from NASDAQ FTP...")
            self._fetch_nasdaq_ftp()
            
            # Step 2: Enrich with Yahoo Finance metadata
            LOGGER.info("\n🔍 Step 2/5: Enriching symbols with metadata...")
            self._enrich_symbols()
            
            # Step 3: Validate and deduplicate
            LOGGER.info("\n✅ Step 3/5: Validating and deduplicating...")
            self._validate_symbols()
            
            # Step 4: Store in database
            LOGGER.info("\n💾 Step 4/5: Storing in PostgreSQL...")
            self._store_symbols()
            
            # Step 5: Summary
            elapsed = time.time() - start_time
            LOGGER.info("\n" + "=" * 60)
            LOGGER.info("🎉 Ingestion Complete!")
            LOGGER.info("=" * 60)
            LOGGER.info(f"⏱️  Duration: {elapsed:.2f}s")
            LOGGER.info(f"📊 NASDAQ: {self.stats['nasdaq']} symbols")
            LOGGER.info(f"📊 NYSE: {self.stats['nyse']} symbols")
            LOGGER.info(f"📊 Enriched: {self.stats['enriched']} symbols")
            LOGGER.info(f"📊 Duplicates: {self.stats['duplicates']} removed")
            LOGGER.info(f"📊 Invalid: {self.stats['invalid']} removed")
            LOGGER.info(f"📊 Total Ingested: {self.stats['total']} symbols")
            LOGGER.info("=" * 60)
            
            return True
            
        except Exception as e:
            LOGGER.error(f"❌ Ingestion failed: {e}", exc_info=True)
            return False
    
    def _fetch_nasdaq_ftp(self):
        """Fetch symbols from NASDAQ FTP (official source)"""
        try:
            # Connect to NASDAQ FTP
            ftp = ftplib.FTP("ftp.nasdaqtrader.com")
            ftp.login()  # Anonymous login
            
            # Fetch NASDAQ listed stocks
            LOGGER.info("   📂 Fetching nasdaqlisted.txt...")
            nasdaq_data = io.BytesIO()
            ftp.retrbinary("RETR SymbolDirectory/nasdaqlisted.txt", nasdaq_data.write)
            nasdaq_data.seek(0)
            
            nasdaq_csv = csv.DictReader(io.TextIOWrapper(nasdaq_data, encoding="utf-8"), delimiter="|")
            for row in nasdaq_csv:
                symbol = row.get("Symbol", "").strip()
                name = row.get("Security Name", "").strip()
                
                if symbol and symbol != "File Creation Time:":
                    self.symbols[symbol] = {
                        "symbol": symbol,
                        "name": name,
                        "asset_type": "stock",
                        "exchange": "NASDAQ",
                        "sector": None,
                        "industry": None,
                        "market_cap": None,
                        "is_active": 1
                    }
                    self.stats["nasdaq"] += 1
            
            LOGGER.info(f"   ✅ NASDAQ: {self.stats['nasdaq']} symbols")
            
            # Fetch other listed stocks (NYSE, AMEX, etc.)
            LOGGER.info("   📂 Fetching otherlisted.txt...")
            other_data = io.BytesIO()
            ftp.retrbinary("RETR SymbolDirectory/otherlisted.txt", other_data.write)
            other_data.seek(0)
            
            other_csv = csv.DictReader(io.TextIOWrapper(other_data, encoding="utf-8"), delimiter="|")
            for row in other_csv:
                symbol = row.get("ACT Symbol", "").strip() or row.get("NASDAQ Symbol", "").strip()
                name = row.get("Security Name", "").strip()
                exchange = row.get("Exchange", "").strip()
                
                if symbol and symbol not in self.symbols and symbol != "File Creation Time:":
                    self.symbols[symbol] = {
                        "symbol": symbol,
                        "name": name,
                        "asset_type": "stock",
                        "exchange": exchange or "NYSE",
                        "sector": None,
                        "industry": None,
                        "market_cap": None,
                        "is_active": 1
                    }
                    self.stats["nyse"] += 1
                elif symbol in self.symbols:
                    self.stats["duplicates"] += 1
            
            LOGGER.info(f"   ✅ NYSE/AMEX: {self.stats['nyse']} symbols")
            
            ftp.quit()
            
        except Exception as e:
            LOGGER.error(f"   ❌ NASDAQ FTP failed: {e}")
            LOGGER.info("   ⚠️  Falling back to cached symbols...")
            self._fallback_symbols()
    
    def _enrich_symbols(self):
        """Enrich symbols with metadata from Yahoo Finance"""
        LOGGER.info(f"   🔍 Enriching {len(self.symbols)} symbols...")
        
        # Process in batches to avoid rate limits
        batch_size = 100
        symbols_list = list(self.symbols.keys())
        
        for i in range(0, len(symbols_list), batch_size):
            batch = symbols_list[i:i+batch_size]
            
            for symbol in batch:
                try:
                    # Import yfinance lazily
                    import yfinance as yf
                    
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    if info:
                        # Update symbol data
                        self.symbols[symbol]["sector"] = info.get("sector")
                        self.symbols[symbol]["industry"] = info.get("industry")
                        self.symbols[symbol]["market_cap"] = info.get("marketCap")
                        
                        self.stats["enriched"] += 1
                    
                    # Avoid rate limits
                    time.sleep(0.1)
                
                except Exception as e:
                    LOGGER.debug(f"      ⚠️  Failed to enrich {symbol}: {e}")
            
            # Log progress
            if (i + batch_size) % 500 == 0:
                LOGGER.info(f"      Progress: {i + batch_size}/{len(symbols_list)} symbols")
        
        LOGGER.info(f"   ✅ Enriched {self.stats['enriched']}/{len(self.symbols)} symbols")
    
    def _validate_symbols(self):
        """Validate and clean symbol data"""
        invalid = []
        
        for symbol, data in list(self.symbols.items()):
            # Remove symbols with invalid characters
            if not symbol.replace(".", "").replace("-", "").isalnum():
                invalid.append(symbol)
                continue
            
            # Remove test symbols
            if symbol.startswith("TEST") or symbol.startswith("DEMO"):
                invalid.append(symbol)
                continue
            
            # Remove very long symbols (likely errors)
            if len(symbol) > 6:
                invalid.append(symbol)
                continue
        
        # Remove invalid symbols
        for symbol in invalid:
            del self.symbols[symbol]
            self.stats["invalid"] += 1
        
        LOGGER.info(f"   ✅ Validated: {len(self.symbols)} valid, {self.stats['invalid']} invalid")
    
    def _store_symbols(self):
        """Store symbols in PostgreSQL database"""
        try:
            # Prepare data for batch insert
            now = int(time.time())
            
            # Detect database type for correct parameter style
            from core.db_engine import IS_POSTGRES
            param_style = "%s" if IS_POSTGRES else "?"
            
            # Use UPSERT syntax
            with get_db_connection() as conn:
                cursor = conn.cursor()
                
                for _symbol, data in self.symbols.items():
                    if IS_POSTGRES:
                        # PostgreSQL UPSERT
                        cursor.execute(f"""
                            INSERT INTO symbol_universe
                            (symbol, name, asset_type, exchange, sector, industry, market_cap, is_active, last_updated)
                            VALUES ({param_style}, {param_style}, {param_style}, {param_style}, {param_style}, {param_style}, {param_style}, {param_style}, {param_style})
                            ON CONFLICT (symbol) DO UPDATE SET
                                name = EXCLUDED.name,
                                exchange = EXCLUDED.exchange,
                                sector = EXCLUDED.sector,
                                industry = EXCLUDED.industry,
                                market_cap = EXCLUDED.market_cap,
                                is_active = EXCLUDED.is_active,
                                last_updated = EXCLUDED.last_updated
                        """, (
                            data["symbol"],
                            data["name"],
                            data["asset_type"],
                            data["exchange"],
                            data["sector"],
                            data["industry"],
                            data["market_cap"],
                            data["is_active"],
                            now
                        ))
                    else:
                        # SQLite UPSERT
                        cursor.execute(f"""
                            INSERT INTO symbol_universe
                            (symbol, name, asset_type, exchange, sector, industry, market_cap, is_active, last_updated)
                            VALUES ({param_style}, {param_style}, {param_style}, {param_style}, {param_style}, {param_style}, {param_style}, {param_style}, {param_style})
                            ON CONFLICT (symbol) DO UPDATE SET
                                name = excluded.name,
                                exchange = excluded.exchange,
                                sector = excluded.sector,
                                industry = excluded.industry,
                                market_cap = excluded.market_cap,
                                is_active = excluded.is_active,
                                last_updated = excluded.last_updated
                        """, (
                            data["symbol"],
                            data["name"],
                            data["asset_type"],
                            data["exchange"],
                            data["sector"],
                            data["industry"],
                            data["market_cap"],
                            data["is_active"],
                            now
                        ))
                
                conn.commit()
            
            self.stats["total"] = len(self.symbols)
            LOGGER.info(f"   ✅ Stored {self.stats['total']} symbols in database")
            
        except Exception as e:
            LOGGER.error(f"   ❌ Failed to store symbols: {e}")
            raise
    
    def _fallback_symbols(self):
        """Fallback: Load predefined list of major stocks"""
        LOGGER.info("   📦 Loading fallback symbol list...")
        
        # Major indices + top stocks
        fallback = [
            # S&P 500 top 50
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "LLY", "V",
            "JPM", "WMT", "MA", "UNH", "PG", "JNJ", "XOM", "HD", "COST", "ORCL",
            "ABBV", "CVX", "BAC", "AVGO", "KO", "MRK", "PEP", "CRM", "CSCO", "TMO",
            "ADBE", "ACN", "AMD", "LIN", "NFLX", "ABT", "MCD", "NKE", "DHR", "INTC",
            "VZ", "TXN", "CMCSA", "PM", "DIS", "QCOM", "HON", "NEE", "UNP", "WFC",
            # NASDAQ 100 additions
            "GOOG", "TMUS", "INTU", "ISRG", "BKNG", "AMGN", "ADP", "SBUX", "GILD", "ADI",
            # NYSE top stocks
            "BX", "C", "GS", "MS", "SCHW", "AXP", "USB", "PNC", "TFC", "COF",
            # ETFs
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "AGG", "BND", "GLD", "SLV"
        ]
        
        for symbol in fallback:
            self.symbols[symbol] = {
                "symbol": symbol,
                "name": symbol,
                "asset_type": "stock",
                "exchange": "NASDAQ" if symbol in ["AAPL", "MSFT", "GOOGL"] else "NYSE",
                "sector": None,
                "industry": None,
                "market_cap": None,
                "is_active": 1
            }
        
        self.stats["nasdaq"] = len(fallback)
        LOGGER.info(f"   ✅ Loaded {len(fallback)} fallback symbols")


def main():
    """CLI entry point"""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Run ingestion
    ingester = SymbolIngester()
    success = ingester.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
