# GHOST INTEL - INTEL RUNBOOK
## Deployment & Verification Guide

### Overview
Ghost Intel is the 8-layer institutional intelligence system that enables Ghost to enter at steps 1-3 instead of step 9 (retail timing).

### Quick Start

```bash
# Run smoke test to verify all components
cd /workspaces/ghost-protocol
python intel_smoke_test.py
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/intel/now` | Top live events sorted by impact |
| `GET /api/intel/timeline?hours=24` | Events over time window |
| `GET /api/intel/impact/{symbol}` | Aggregated impact for a symbol |
| `GET /api/intel/rates` | Live treasury yields, DXY, VIX |
| `GET /api/intel/positioning` | Market positioning analysis |
| `GET /api/intel/social/{symbol}` | Social sentiment (StockTwits, Reddit) |
| `GET /api/intel/macro` | FRED macro data (CPI, NFP, GDP) |
| `GET /api/intel/health` | Feed availability status |

### Required Environment Variables

#### Core APIs (Set these in Railway)

```bash
# FRED API (free) - For macro data
# Get from: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key

# Polygon (free tier works) - For news
# Get from: https://polygon.io/
POLYGON_API_KEY=your_polygon_key

# Optional: Reddit API (for better WSB data)
# Get from: https://www.reddit.com/prefs/apps
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
```

#### Free Sources (No API Key Needed)
- Yahoo Finance: Rates, VIX, DXY ✅
- StockTwits: Social sentiment ✅
- Reddit (public): WSB top posts ✅
- CBOE: Put/Call ratio (estimated from VIX) ✅

### The 8-Layer Model

```
Layer 1: MACRO DATA (CPI, Jobs, FOMC, GDP)        → FRED API
         ↓
Layer 2: RATES & LIQUIDITY (2Y, 10Y, DXY, VIX)    → Yahoo/FRED
         ↓
Layer 3: CORPORATE (Earnings, Guidance)            → Polygon
         ↓
Layer 4: POLITICS (Tariffs, Sanctions)             → News
         ↓
Layer 5: GEOPOLITICS (War, Energy)                 → News
         ↓
Layer 6: KEY INDIVIDUALS (Elon, Fed Chair)         → Twitter/News
         ↓
Layer 7: SOCIAL (Attention velocity)               → StockTwits/Reddit
         ↓
Layer 8: POSITIONING (Options, Gamma)              → CBOE/VIX
         ↓
OUTPUT: PRICE MOVEMENT
```

### Impact Scoring Components (0-100)

| Component | Max Score | What It Measures |
|-----------|-----------|------------------|
| Rate Sensitivity | 25 | Changes rate expectations |
| Liquidity | 20 | Market fragility (VIX, P/C) |
| Earnings | 15 | Forward earnings changes |
| Geopolitical | 15 | Energy, trade, conflict risk |
| Virality | 10 | Attention velocity |
| Positioning | 15 | Market positioned wrong |

### Credibility Multipliers

| Source Tier | Multiplier | Examples |
|-------------|------------|----------|
| Tier 1 | 1.0x | Fed, SEC, company filings |
| Tier 2 | 0.95x | Reuters, Bloomberg, WSJ |
| Tier 3 | 0.85x | Yahoo, business sites |
| Tier 4 | 0.70x | Twitter verified |
| Tier 5 | 0.55x | Social unverified |

### Action Signals

| Score | Signal | Action |
|-------|--------|--------|
| 70+ | ACT | High impact - immediate attention |
| 50-69 | PREPARE | Significant - prepare for impact |
| 30-49 | WATCH | Monitor situation |
| <30 | IGNORE | Low impact - noise |

### Example API Calls

```bash
# Get top live events
curl https://your-domain.railway.app/api/intel/now?limit=10&min_score=40

# Get impact for NVDA
curl https://your-domain.railway.app/api/intel/impact/NVDA

# Check current rates
curl https://your-domain.railway.app/api/intel/rates

# Get positioning analysis
curl https://your-domain.railway.app/api/intel/positioning

# Get social sentiment for TSLA
curl https://your-domain.railway.app/api/intel/social/TSLA

# Check feed health
curl https://your-domain.railway.app/api/intel/health
```

### Testing Locally

```bash
cd /workspaces/ghost-protocol

# Run smoke test
python intel_smoke_test.py

# Start server with intel routes
python wolf_app.py

# In another terminal, test endpoints
curl http://localhost:8000/api/intel/health
curl http://localhost:8000/api/intel/rates
curl http://localhost:8000/api/intel/now
```

### Troubleshooting

#### "FRED_API_KEY not set"
Set the environment variable in Railway or local .env:
```bash
export FRED_API_KEY=your_key_here
```

#### "Rate limit" errors
The system has built-in rate limiting. Wait and retry, or the cache will serve data.

#### No macro data
FRED API key is required for macro data. Get one free at:
https://fred.stlouisfed.org/docs/api/api_key.html

#### StockTwits 429 errors
StockTwits free tier: 200 requests/hour. The cache (2 min TTL) helps.

### Integration with Ghost Brain

Ghost Intel feeds into the main prediction system:

1. **Before prediction**: Call `/api/intel/impact/{symbol}` 
2. **Check event impact**: If score > 50, factor into prediction
3. **Check positioning**: Use `/api/intel/positioning` for size adjustment
4. **Social acceleration**: Check `/api/intel/social/{symbol}` for momentum

### Files Reference

| File | Purpose |
|------|---------|
| `ghost_intel/__init__.py` | Module exports |
| `ghost_intel/sources.py` | All data connectors |
| `ghost_intel/normalize.py` | Event schema & deduplication |
| `ghost_intel/impact_model.py` | Impact scoring engine |
| `ghost_intel/positioning.py` | Options/vol analysis |
| `ghost_intel/taxonomy.py` | Event classification |
| `ghost_intel/routes.py` | API endpoints |
| `intel_smoke_test.py` | Verification tests |

### Why This Matters

| Before (Technical Only) | After (Full Picture) |
|------------------------|---------------------|
| RSI < 30 = buy | FOMC dovish + rates falling + low VIX = buy |
| MACD crossover | Institutional accumulation + guidance raise |
| Volume spike | Positioning fragile + macro surprise |
| **40% accuracy ceiling** | **60-75% possible** |

Ghost was entering at step 9 (retail timing). With Ghost Intel, we enter at steps 1-3 (institutional timing).

---

**Last Updated**: 2026-01-26
**Module Version**: 1.0.0
