# 🧠 OMNIBRAIN V10.3 - IMPLEMENTATION PLAN

**Mission**: Deliver fully working GHOST for STOCKS + CRYPTO with Telegram integration

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1: Crypto Module Foundation ✅ (Already Created)

- [x] `core/crypto/__init__.py` - Module exports
- [x] `core/crypto/crypto_providers.py` - CoinGecko, Binance, Coinbase providers
- [x] `core/crypto/crypto_predictor.py` - 24h prediction engine
- [x] Symbol map with 45+ cryptocurrencies
- [x] Quorum consensus (median of 2+ providers)
- [x] Database schema (crypto_predictions, crypto_forecast_points, crypto_actual_points)


### Phase 2: API Integration (NOW)

- [ ] Add crypto routes to wolf_app.py:
  - `GET /api/crypto/price/{symbol}`
  - `GET /api/crypto/predict/{symbol}`
  - `POST /api/crypto/predict/run`
  - `GET /api/crypto/watchlist`
- [ ] Update `/api/cockpit` to include crypto block when CRYPTO_ENABLED=1
- [ ] Add Prometheus metrics for crypto operations
- [ ] UTC timestamp handling throughout


### Phase 3: UI Updates (NOW)

- [ ] Rename "Ghost Predictions" panel title (already correct in current HTML)
- [ ] Add Stocks | Crypto toggle tabs
- [ ] Update prediction chart to support both asset classes
- [ ] Add prediction history scorecard with MAP
- [ ] Local time rendering via JavaScript
- [ ] Remove any 1970/1969 date artifacts


### Phase 4: Sentiment & Macro Brain

- [ ] Wire NEWS_SENTIMENT_ON=1 to fetch per-symbol sentiment
- [ ] Wire MACRO_BRAIN_ON=1 for QQQ/SOXX/SMH proxies
- [ ] FUSION_AI_ON=1 for weighted final score


### Phase 5: Telegram Integration

- [ ] Update `/alerts/test` with Railway URL
- [ ] Add `/status`, `/signal`, `/pnl` commands
- [ ] Free-form Q&A with current data
- [ ] Logging for webhook deliveries


### Phase 6: Time & Freshness Fixes

- [ ] Audit all timestamps → UTC timezone-aware
- [ ] Fix snapshot `as_of` generation
- [ ] Add `ghost_snapshot_asof` Prometheus gauge
- [ ] UI renders local time via JS


### Phase 7: Testing & Validation

- [ ] Unit tests for crypto providers
- [ ] Unit tests for crypto predictor
- [ ] Integration test for `/api/cockpit`
- [ ] Live test for crypto endpoints
- [ ] UI manual verification
- [ ] Metrics verification


### Phase 8: Documentation & Deployment

- [ ] Update README with OmniBrain section
- [ ] Update CRYPTO_MODULE_QUICKSTART.md
- [ ] Create CHANGELOG entry for v10.3
- [ ] Railway environment variables check
- [ ] Deployment verification


______________________________________________________________________

## 🔧 REQUIRED RAILWAY ENVIRONMENT VARIABLES

```bash

# Core Config

SIM_MODE=0
LOG_LEVEL=INFO
LOG_JSON=1
PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
ADMIN_IP_ALLOWLIST=127.0.0.1

# AI Config

AI_ON=1
AI_PROVIDER=openai
AI_MODEL=gpt-4o-mini
OPENAI_API_KEY=<your-key>

# Features

NEWS_SENTIMENT_ON=1
FUSION_AI_ON=1
MACRO_BRAIN_ON=1

# Crypto Config

CRYPTO_ENABLED=1
CRYPTO_SYMBOLS=BTC,ETH,SOL,BNB,DOGE,SHIB,PEPE,AVAX,DOT,MATIC,LINK,UNI,AAVE
CRYPTO_LOOKBACK_H=96
CRYPTO_FORECAST_H=48
CRYPTO_PRICE_SOURCE=coingecko
CRYPTO_QUORUM=coingecko,binance,coinbase

# Telegram (if using)

TELEGRAM_BOT_TOKEN=<your-bot-token>
TELEGRAM_CHAT_ID=<your-chat-id>

# Stock Price Providers

ALPHA_VANTAGE_API_KEY=<your-key>
POLYGON_API_KEY=<your-key>

# Security

GHOST_API_TOKEN=<your-secret-token>

```text

______________________________________________________________________

## 🎯 ACCEPTANCE TESTS

### Must Pass Before Complete

1. `GET /health` → `{"ok": true}`
2. `GET /api/cockpit` → Has `{as_of, portfolio, predictions: {stocks, crypto?}}`
3. `GET /api/crypto/price/BTC` → `price > 0, provider in [coingecko, binance, coinbase]`
4. `GET /api/crypto/predict/BTC` → `forecast_h=48, confidence ∈ [0,100], path non-empty`
5. UI shows "Ghost Prediction" panel with Stocks | Crypto tabs
6. `/metrics` includes `ghost_crypto_*` series
7. Telegram `/status`, `/signal`, `/pnl` work from Railway deployment
8. No 1970/1969 dates anywhere in UI or API responses


______________________________________________________________________

## 📊 PROGRESS TRACKING

**Current Status**: Starting Phase 2 (API Integration)

**Estimated Time**:

- Phase 2: 30 minutes
- Phase 3: 45 minutes
- Phase 4: 30 minutes
- Phase 5: 30 minutes
- Phase 6: 20 minutes
- Phase 7: 30 minutes
- Phase 8: 15 minutes **Total**: ~3.5 hours


______________________________________________________________________

## 🚀 NEXT ACTIONS

1. Complete crypto route integration in wolf_app.py
2. Update cockpit.html with Stocks | Crypto tabs
3. Add inline JavaScript for chart rendering
4. Test all endpoints locally
5. Deploy to Railway
6. Verify production


Let's build! 🔨
