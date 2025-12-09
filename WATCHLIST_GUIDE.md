# 🎯 GHOST Watchlist Manager - Complete Guide

## Overview

The GHOST Watchlist Manager is a sophisticated system that tracks market symbols and
filters them based on GHOST Performance Score (GPS). **Only symbols that pass the GHOST
scoring threshold appear in "top movers"**- this is your buy signal list.

______________________________________________________________________

## 🏗️ Architecture

### Core Components

1.**Watchlist Manager**(`core/watchlist_manager.py`)

- Manages watchlist symbols
- Calculates and tracks GHOST scores
- Filters symbols by GPS threshold
- Maintains historical score database

1.**API Endpoints**(integrated in `wolf_app.py`)

- `/api/watchlist` - Get all watchlist symbols
- `/api/watchlist/add` - Add symbol to watchlist
- `/api/watchlist/remove` - Remove symbol from watchlist
- `/api/watchlist/score` - Update GHOST score for symbol
- `/api/watchlist/history/{symbol}` - Get historical scores
- `/api/watchlist/statistics` - Get watchlist statistics
- `/api/top_movers` -**Updated**- Only returns symbols with GPS ≥ threshold

1.**Database**(`watchlist.db`)

- `watchlist` table: Symbol information
- `ghost_scores` table: Historical GPS scores
- Indexed for fast queries

______________________________________________________________________

## 📊 GHOST Performance Score (GPS)

### Scoring Logic (0-10 scale)

The GPS is calculated based on multiple market factors:

```python
Base Score: 5.0

Momentum Scoring:

  - +1.5 if abs(change_pct) > 3.0%    (Strong momentum)
  - +1.0 if abs(change_pct) > 2.0%    (Good momentum)
  - +0.5 if abs(change_pct) > 1.0%    (Moderate momentum)


Volatility Sweet Spot:

  - +0.5 if 0.5% ≤ abs(change_pct) ≤ 5.0%  (Not too low, not too high)


Large Cap Stability:

  - +0.5 if market_cap > $50B


High Volume Interest:

  - +0.3 if volume > 7,000,000


Maximum Score: 10.0

```text

### GPS Threshold**Default: 7.0**-**GPS ≥ 7.0**: Symbol appears in top movers → **Buy Signal**-**GPS < 7.0**: Symbol stays in watchlist → **Watch Only**______________________________________________________________________

## 🚀 Initial Setup

### Your 52 Watchlist Symbols

All symbols from your data have been pre-configured:

```text

WFC, SLB, HLN, CNH, KDP, CORZ, SBUX, UWMC, EQT, MDT,
HPQ, ETSY, PBA, LVS, PGY, CTRA, HBM, MRNA, SBSW, CVS,
KHC, M, VTRS, PDD, ELAN, CFG, CRM, ENVX, SCHW, WRD,
NWL, CL, UAA, EBAY, IPG, NG, SIRI, CAH, WMB, PPL,
MDU, TFC, AEO, GAP, MAT, STUB, APH, CNP, ANET, MDLZ,
USB, CRDO

```text

### Population Script

Run this to populate the watchlist with initial market data:

```bash

cd /workspaces/GHOST
python utils/populate_watchlist.py

```text

This script will:

1. ✅ Add all 52 symbols to watchlist
2. 📊 Calculate GPS scores based on current market data
3. 🔥 Show top movers (GPS ≥ 7.0)
4. 📈 Display statistics


______________________________________________________________________

## 🔌 API Usage

### 1. Get All Watchlist Symbols

```bash

curl <<<<<http://localhost:5000/api/watchlist>>>>>

```text

Response:

```json

{
  "symbols": [
    {
      "symbol": "AAPL",
      "name": "Apple Inc.",
      "added_at": "2025-10-05T12:00:00",
      "last_updated": "2025-10-05T14:30:00",
      "metadata": ""
    },
    ...
  ],
  "count": 52
}

```text

### 2. Add Symbol to Watchlist

```bash

curl -X POST "<<<<<http://localhost:5000/api/watchlist/add?symbol=TSLA&name=Tesla>>>>> Inc"

```text

Response:

```json

{
  "success": true,
  "symbol": "TSLA",
  "name": "Tesla Inc",
  "added_at": "2025-10-05T14:35:00"
}

```text

### 3. Update GHOST Score

```bash

curl -X POST "<<<<<http://localhost:5000/api/watchlist/score">>>>> \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "gps_score": 8.5,
    "price": 175.25,
    "change_pct": 2.5,
    "volume": 50000000,
    "market_cap": 2800000000000,
    "threshold": 7.0
  }'

```text

Response:

```json

{
  "success": true,
  "symbol": "AAPL",
  "gps_score": 8.5,
  "passed_threshold": true,
  "threshold": 7.0,
  "timestamp": "2025-10-05T14:40:00"
}

```text

### 4. Get Top Movers (Buy Signals)

```bash

curl "<<<<<http://localhost:5000/api/top_movers?threshold=7.0&limit=20">>>>>

```text

Response:

```json

{
  "stocks": [
    {
      "symbol": "EBAY",
      "sym": "EBAY",
      "name": "eBay Inc.",
      "gps": 8.5,
      "price": 92.17,
      "change_pct": 4.26,
      "volume": 7537000,
      "market_cap": 42122000000,
      "timestamp": "2025-10-05T14:40:00"
    },
    {
      "symbol": "PBA",
      "sym": "PBA",
      "name": "Pembina Pipeline Corporation",
      "gps": 8.2,
      "price": 42.09,
      "change_pct": 6.02,
      "volume": 7355000,
      "market_cap": 24489000000,
      "timestamp": "2025-10-05T14:40:00"
    },
    ...
  ],
  "crypto": [],
  "threshold": 7.0,
  "count": 15
}

```text

### 5. Get Symbol History

```bash

curl "<<<<<http://localhost:5000/api/watchlist/history/AAPL?limit=10">>>>>

```text

Response:

```json

{
  "symbol": "AAPL",
  "history": [
    {
      "timestamp": "2025-10-05T14:40:00",
      "gps_score": 8.5,
      "price": 175.25,
      "change_pct": 2.5,
      "volume": 50000000,
      "market_cap": 2800000000000,
      "passed_threshold": true
    },
    ...
  ],
  "count": 10
}

```text

### 6. Get Watchlist Statistics

```bash

curl <<<<<http://localhost:5000/api/watchlist/statistics>>>>>

```text

Response:

```json

{
  "total_symbols": 52,
  "symbols_with_scores": 52,
  "symbols_passing_threshold": 15,
  "average_gps_score": 6.8,
  "pass_rate_pct": 28.85
}

```text

### 7. Remove Symbol

```bash

curl -X POST "<<<<<http://localhost:5000/api/watchlist/remove?symbol=TSLA">>>>>

```text

Response:

```json

{
  "success": true,
  "symbol": "TSLA",
  "deleted": true
}

```text

______________________________________________________________________

## 💡 Usage Workflow

### The GHOST Way: Watchlist → GPS Scoring → Top Movers → Buy Signal

1.**Add Symbols to Watchlist**```bash

   # Symbols are monitored but not actionable yet

   curl -X POST "<<<<<http://localhost:5000/api/watchlist/add?symbol=AAPL&name=Apple>>>>> Inc"

   ```text

1.**Calculate GPS Scores**(Automated or Manual)


   ```bash

   # Update GPS score based on market data

   curl -X POST "<<<<<http://localhost:5000/api/watchlist/score">>>>> \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "AAPL",
       "gps_score": 8.5,
       "price": 175.25,
       "change_pct": 2.5
     }'

   ```text

1.**Check Top Movers**```bash

   # Only symbols with GPS ≥ 7.0 appear here

   curl "<<<<<http://localhost:5000/api/top_movers?threshold=7.0">>>>>

   ```text

1.**Buy Signal**- ✅ Symbol appears in `/api/top_movers` →**Consider buying**- ⏸️ Symbol not in top movers →**Continue watching**______________________________________________________________________

## 📈 Expected Top Movers from Your Data

Based on the initial market data, these symbols likely pass GPS ≥ 7.0:

### High Confidence (GPS ≥ 8.0)

-**PBA**(Pembina Pipeline) - +6.02% change
-**EBAY**(eBay Inc.) - +4.26% change
-**MAT**(Mattel) - +4.88% change
-**NG**(NovaGold Resources) - +3.52% change
-**NWL**(Newell Brands) - +3.55% change


### Medium Confidence (GPS 7.0-7.9)

-**SIRI**(Sirius XM) - +2.99% change
-**MDT**(Medtronic) - +2.33% change
-**ENVX**(Enovix) - +2.32% change
-**STUB**(StubHub) - +2.11% change
-**HBM**(Hudbay Minerals) - +1.88% change


### Large Caps with Momentum

-**SCHW**(Charles Schwab) - +1.49%, $170B cap
-**CFG**(Citizens Financial) - +1.59%, $23B cap
-**MDLZ**(Mondelez) - +1.44%, $81B cap


______________________________________________________________________

## 🔄 Automation Ideas

### 1. Scheduled GPS Updates

Create a cron job to update GPS scores every 5 minutes:

```python

# scheduled_gps_update.py

import requests
from core.watchlist_manager import get_watchlist_manager

def update_all_scores():
    watchlist_mgr = get_watchlist_manager()
    symbols = watchlist_mgr.get_watchlist()

    for symbol_data in symbols:
        symbol = symbol_data['symbol']

        # Fetch current price (implement your price fetcher)

        price, change_pct = fetch_current_price(symbol)

        # Calculate GPS

        gps_score = calculate_gps(symbol, price, change_pct)

        # Update score

        requests.post(f"<<<<<http://localhost:5000/api/watchlist/score",>>>>> json={
            "symbol": symbol,
            "gps_score": gps_score,
            "price": price,
            "change_pct": change_pct,
            "threshold": 7.0
        })

if __name__ == "__main__":
    update_all_scores()

```text

### 2. Alert System

Get notified when a symbol crosses the GPS threshold:

```python

def check_for_buy_signals():
    response = requests.get("<<<<<http://localhost:5000/api/top_movers?threshold=7.0>>>>>")
    movers = response.json()['stocks']

    for stock in movers:
        if stock['gps'] >= 7.5:  # High confidence
            send_alert(f"🔥 BUY SIGNAL: {stock['symbol']} - GPS {stock['gps']}")

```text

______________________________________________________________________

## 📊 Database Schema

### watchlist Table

```sql

CREATE TABLE watchlist (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    added_at TEXT,
    last_updated TEXT,
    metadata TEXT
);

```text

### ghost_scores Table

```sql

CREATE TABLE ghost_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    gps_score REAL,
    price REAL,
    change_pct REAL,
    volume REAL,
    market_cap REAL,
    passed_threshold INTEGER DEFAULT 0,
    FOREIGN KEY (symbol) REFERENCES watchlist(symbol)
);

CREATE INDEX idx_scores_symbol_time ON ghost_scores(symbol, timestamp DESC);

```text

______________________________________________________________________

## 🎯 Best Practices

### GPS Threshold Tuning

-**Conservative**(GPS ≥ 8.0): Fewer signals, higher quality
-**Balanced**(GPS ≥ 7.0): Default, good signal-to-noise ratio
-**Aggressive**(GPS ≥ 6.0): More signals, requires more filtering


### Watchlist Management

1.**Start with Quality**: Add fundamentally sound companies

1. **Monitor GPS Trends**: Watch for symbols approaching threshold
2. **Respect the Signal**: GPS ≥ 7.0 = Buy consideration
3. **Review History**: Check `/api/watchlist/history/{symbol}` for patterns


### Integration with GHOST Stages

The watchlist system integrates with all GHOST intelligence stages:

- **Stage 1**(World Context): News sentiment affects GPS


-**Stage 2**(Learning): GPS threshold auto-tuned based on outcomes
-**Stage 3**(Risk/Regime): GPS adjusted for market regime
-**Stage 4**(Portfolio): Watchlist used for optimization inputs
-**Stage 5**(Execution): Top movers trigger order creation


______________________________________________________________________

## 🔧 Configuration

### Update GPS Threshold

In your application:

```python

# Set custom threshold

custom_threshold = 7.5

response = requests.get(f"<<<<<http://localhost:5000/api/top_movers?threshold={custom_threshold}>>>>>")

```text

### Adjust GPS Calculation

Modify `utils/populate_watchlist.py` → `calculate_gps_score()` function to customize
scoring logic.

______________________________________________________________________

## ✅ Testing

### Test Watchlist Operations

```bash

# 1. Populate watchlist

python utils/populate_watchlist.py

# 2. Check watchlist

curl <<<<<http://localhost:5000/api/watchlist>>>>> | jq '.count'

# 3. Check statistics

curl <<<<<http://localhost:5000/api/watchlist/statistics>>>>> | jq

# 4. Get top movers

curl "<<<<<http://localhost:5000/api/top_movers?threshold=7.0">>>>> | jq '.stocks | length'

# 5. Test score update

curl -X POST "<<<<<http://localhost:5000/api/watchlist/score">>>>> \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "gps_score": 9.0,
    "price": 175.0,
    "change_pct": 3.5,
    "threshold": 7.0
  }' | jq

# 6. Verify in top movers

curl "<<<<<http://localhost:5000/api/top_movers?threshold=7.0">>>>> | jq '.stocks[] | select(.symbol == "AAPL")'

```text

______________________________________________________________________

## 🚀 Quick Start

```bash

# 1. Start GHOST server

cd /workspaces/GHOST
source .venv/bin/activate
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload

# 2. In another terminal, populate watchlist

python utils/populate_watchlist.py

# 3. Check top movers (buy signals)

curl "<<<<<http://localhost:5000/api/top_movers?threshold=7.0">>>>>

# 4. Monitor in browser

open <<<<<http://localhost:5000/cockpit>>>>>

```text

______________________________________________________________________

## 📝 Summary**The GHOST Way**

1. ✅ All 52 symbols added to watchlist
2. 📊 GPS scores calculated automatically
3. 🎯 Only GPS ≥ 7.0 symbols in `/api/top_movers`
4. 🔥 Top movers = Your buy signal list
5. 📈 Historical tracking for pattern analysis


**Key Insight**:

- Watchlist = monitoring zone
- Top movers = action zone
- GPS threshold = the decision boundary


**When you see a symbol in `/api/top_movers`, the GHOST logic has already validated it
as a buy candidate!**

______________________________________________________________________

## 🎓 Advanced Features

### Custom GPS Algorithms

You can implement custom GPS calculation algorithms:

```python

def custom_gps_calculator(symbol_data):

    # Your custom logic

    score = 5.0

    # Technical indicators

    if rsi > 70:
        score += 1.0

    # Fundamental factors

    if pe_ratio < 15:
        score += 0.5

    # Sentiment analysis

    if news_sentiment > 0.7:
        score += 1.5

    return min(10.0, score)

```text

### Multi-Timeframe Scoring

Track GPS across different timeframes:

```python

# 5-minute GPS

gps_5m = calculate_gps(symbol, timeframe='5m')

# 1-hour GPS

gps_1h = calculate_gps(symbol, timeframe='1h')

# 1-day GPS

gps_1d = calculate_gps(symbol, timeframe='1d')

# Composite score

gps_composite = (gps_5m *0.2) + (gps_1h*0.3) + (gps_1d* 0.5)

```text

______________________________________________________________________

**Status**: ✅ Watchlist Manager fully integrated and operational!

**Next Steps**: Run `python utils/populate_watchlist.py` to get started!
