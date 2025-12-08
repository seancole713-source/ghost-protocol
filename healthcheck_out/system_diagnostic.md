# Ghost System Diagnostic Report

**Scan Time:**2025-10-15T10:06:14.873159**Base URL:**<<<<<http://localhost:8444>>>>>

## 🎯 Overall Status

-**Status:**✅ OPERATIONAL
-**Subsystems Healthy:**4/5
-**Endpoints Online:**21/26
-**Critical Issues:**0
-**Warnings:**0


## 🔍 Subsystem Details

### AI Core & Prediction Engine**Status:**✅ HEALTHY**Health:**6/6 endpoints online

| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|
| `/api/agent/stats` | ✅ ONLINE | 11ms | {'total_decisions': 0, 'win_rate': 0.0, 'avg_confi |
| `/api/agent/decisions` | ✅ ONLINE | 2ms | {'decisions': [], 'count': 0} |
| `/api/stage2/forecasts` | ✅ ONLINE | 2ms | {'forecasts': [], 'count': 0} |
| `/api/stage2/accuracy` | ✅ ONLINE | 2ms | {'error': 'No completed forecasts found', 'count': |
| `/api/stage3/regime/current` | ✅ ONLINE | 2ms | {'regime': 'SIDEWAYS', 'confidence': 0.6, 'strateg |
| `/api/snapshot` | ✅ ONLINE | 209ms | {'timestamp': 1760540774.8972857, 'portfolio': {}, |

### Data Feeds (Stocks + Crypto)**Status:**✅ HEALTHY**Health:**7/7 endpoints responding

| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|
| `/api/price/WOLF` | ✅ ONLINE | 6ms | {'symbol': 'WOLF', 'price': 32.58, 'prev_close': 3 |
| `/api/price/SPY` | ✅ ONLINE | 4ms | [{'error': 'Symbol SPY not supported', 'supported' |
| `/api/price/AAPL` | ✅ ONLINE | 5ms | [{'error': 'Symbol AAPL not supported', 'supported |
| `/api/price/BTC-USD` | ✅ ONLINE | 5ms | [{'error': 'Symbol BTC-USD not supported', 'suppor |
| `/api/crypto/price/bitcoin` | ⚠️ HTTP 503 | 3ms | {'detail': 'Crypto module not enabled. Set CRYPTO_ |
| `/api/crypto/price/ethereum` | ⚠️ HTTP 503 | 3ms | {'detail': 'Crypto module not enabled. Set CRYPTO_ |
| `/api/crypto/ohlcv/bitcoin?days=7` | ⚠️ HTTP 503 | 3ms | {'detail': 'Crypto module not enabled'} |

### News & Sentiment Analysis**Status:**⚠️ DEGRADED**Health:**2/3 endpoints online

| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|
| `/api/news` | ✅ ONLINE | 132ms | {'news': [{'title': 'These cooking oil stocks are  |
| `/api/news/recent` | ✅ ONLINE | 129ms | {'news': [], 'count': 0, 'timestamp': 1760540775.3 |
| `/api/watcher/ticker_news` | ⚠️ HTTP 422 | 5ms | {'detail': [{'loc': ['query', 'symbol'], 'msg': 'f |

### Cockpit UI & Frontend**Status:**✅ HEALTHY**Health:**6/6 endpoints accessible

| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|
| `/` | ✅ ONLINE | 11ms | OK |
| `/cockpit` | ✅ ONLINE | 4ms | OK |
| `/api/openapi.json` | ✅ ONLINE | 62ms | {'openapi': '3.1.0', 'info': {'title': 'Ghost — WO |
| `/api/docs` | ✅ ONLINE | 2ms | OK |
| `/health` | ✅ ONLINE | 2ms | {'ok': True, 'ts': 1760540775.4953885} |
| `/static/img/neo_glass_bg.webp` | ✅ ONLINE | 3ms | OK |

### Database & Backend Services**Status:**✅ HEALTHY**Health:**3/4 services responding

| Endpoint | Status | Latency | Notes |
|----------|--------|---------|-------|
| `/api/portfolio` | ✅ ONLINE | 3ms | {'positions': [{'symbol': 'WOLF', 'type': 'stock', |
| `/api/memory/stats` | ⚠️ HTTP 404 | 4ms | {'detail': 'Not Found'} |
| `/health` | ✅ ONLINE | 2ms | {'ok': True, 'ts': 1760540775.510488} |
| `/metrics` | ✅ ONLINE | 4ms | OK |

## 💡 Recommendations

1.**Verify environment variables**- Ensure OPENAI_API_KEY, CRYPTO_ENABLED, etc. are set
2.**Check SIM_MODE setting** - Currently set to 0 (live mode)
