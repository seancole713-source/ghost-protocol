# ✅ OMNIBRAIN V10.3 - COMPLETION CHECKLIST

**Implementation Date**: October 12, 2025\
**Build**: GHOST OmniBrain v10.3 (Stocks + Crypto)\
**Status**: 🚧 IN PROGRESS

______________________________________________________________________

## 📋 PHASE 1: CRYPTO MODULE FOUNDATION

- [x] `core/crypto/__init__.py` - Module exports created
- [x] `core/crypto/crypto_providers.py` - Multi-provider (CoinGecko, Binance, Coinbase)


  ✅

- [x] `core/crypto/crypto_predictor.py` - 24h prediction engine with RSI/momentum ✅
- [x] Symbol map with 45+ cryptocurrencies ✅
- [x] Quorum consensus (median of 2+ providers, \<1% spread) ✅
- [x] Database schema (crypto_predictions, crypto_forecast_points, crypto_actual_points)


  ✅

- [x] Watchlist categorization (default, blue_chip, defi, meme, ai_gaming, all) ✅


## 📋 PHASE 2: API INTEGRATION

- [x] Added `/api/crypto/price/{symbol}` - Get current price with quorum ✅
- [x] Added `/api/crypto/predict/run` - Generate 48h forecast ✅
- [x] Added `/api/crypto/predict/{symbol}` - Get latest prediction ✅
- [x] Added `/api/crypto/watchlist?category=` - Get watchlist with live prices ✅
- [x] Updated `/api/cockpit` to include crypto predictions block ✅
- [x] Added Prometheus metrics for crypto operations:
  - `ghost_crypto_price_fetch_total{provider, result}` ✅
  - `ghost_crypto_predict_seconds{symbol}` ✅
  - `ghost_prediction_mape{asset_class}` ✅
  - `ghost_sentiment_score{symbol}` ✅
  - `ghost_macro_confidence{scenario}` ✅


## 📋 PHASE 3: UI UPDATES

- [ ] Update `templates/cockpit.html`:
  - [ ] Rename panel to "Ghost Prediction" (if needed)
  - [ ] Add Stocks | Crypto toggle tabs
  - [ ] Wire prediction chart for both asset classes
  - [ ] Add history scorecard with MAP display
  - [ ] Local time rendering via JavaScript
  - [ ] Remove 1970/1969 date artifacts
- [ ] Update JavaScript chart rendering
- [ ] Add crypto price display sections
- [ ] Test UI locally in browser


## 📋 PHASE 4: SENTIMENT & MACRO BRAIN

- [ ] Wire NEWS_SENTIMENT_ON=1 to compute per-symbol sentiment
- [ ] Wire MACRO_BRAIN_ON=1 for QQQ/SOXX/SMH proxies
- [ ] FUSION_AI_ON=1 for weighted final score with explicit reasoning
- [ ] Add sentiment scores to cockpit snapshot
- [ ] Test sentiment scoring with real news data


## 📋 PHASE 5: TELEGRAM INTEGRATION

- [ ] Update `/alerts/test` to include Railway URL in card
- [ ] Add `/status` command handler
- [ ] Add `/signal` command handler (stocks + crypto)
- [ ] Add `/pnl` command handler
- [ ] Add free-form Q&A handler with GPT-4
- [ ] Add logging for webhook deliveries
- [ ] Test Telegram commands from mobile device


## 📋 PHASE 6: TIME & FRESHNESS FIXES

- [ ] Audit all timestamp generation → UTC timezone-aware
- [ ] Fix snapshot `as_of` generation (ensure advances)
- [ ] Update `ghost_snapshot_asof` Prometheus gauge
- [ ] UI renders local time via JavaScript
- [ ] Verify no 1970/1969 dates in UI or API responses
- [ ] Test across timezones (UTC, EST, PST)


## 📋 PHASE 7: TESTING & VALIDATION

- [ ] Unit tests for crypto providers (mock HTTP responses)
- [ ] Unit tests for crypto predictor (mock DB)
- [ ] Integration test for `/api/cockpit` with crypto enabled
- [ ] Live test: `curl <<<<<http://localhost:5000/api/crypto/price/BTC`>>>>>
- [ ] Live test: `curl <<<<<http://localhost:5000/api/crypto/predict/BTC`>>>>>
- [ ] Live test: `curl <<<<<http://localhost:5000/api/crypto/watchlist?category=meme`>>>>>
- [ ] Live test: `curl <<<<<http://localhost:5000/api/cockpit>>>>> | jq '.predictions.crypto'`
- [ ] Metrics verification: `curl <<<<<http://localhost:5000/metrics>>>>> | grep ghost_crypto`
- [ ] UI manual check: Open <<<<<http://localhost:5000/cockpit>>>>> in browser
- [ ] Performance test: Measure prediction generation time (\<5s)


## 📋 PHASE 8: DOCUMENTATION & DEPLOYMENT

- [ ] Update README.md:
  - [ ] OmniBrain overview section
  - [ ] Environment variables table
  - [ ] Crypto endpoints documentation
  - [ ] UI screenshots
- [ ] Update CRYPTO_MODULE_QUICKSTART.md with curl examples
- [ ] Create CHANGELOG entry for v10.3
- [ ] Create deployment checklist with Railway steps
- [ ] Verify all Railway environment variables are set
- [ ] Deploy to Railway and test production
- [ ] Update .vscode/settings.json (remove non-existent MCP servers)


______________________________________________________________________

## 🎯 ACCEPTANCE TESTS (Must Pass)

### Core Health

- [ ] `GET /health` returns `{"ok": true}`
- [ ] Server starts without errors in logs
- [ ] All imports resolve correctly


### Cockpit API

- [ ] `GET /api/cockpit` returns valid JSON with:
  - [ ] `as_of` timestamp (not 1970 or 1969)
  - [ ] `portfolio` object with NAV
  - [ ] `predictions.stocks` array
  - [ ] `predictions.crypto` array (if CRYPTO_ENABLED=1)
  - [ ] `status.feeds.crypto` = true (if enabled)


### Crypto Price API

- [ ] `GET /api/crypto/price/BTC` returns:
  - [ ] `price > 0`
  - [ ] `provider` in [coingecko, binance, coinbase]
  - [ ] `confidence` between 0 and 1
  - [ ] `quorum_size >= 1`
  - [ ] `timestamp` is recent (\<60s old)


### Crypto Prediction API

- [ ] `POST /api/crypto/predict/run?symbol=BTC` returns:
  - [ ] `prediction_id` (valid UUID)
  - [ ] `forecast_h = 48`
  - [ ] `confidence` between 0 and 100
  - [ ] `path` array with 48 points
  - [ ] `direction` in [UP, DOWN, FLAT]
- [ ] `GET /api/crypto/predict/BTC` returns same structure


### Crypto Watchlist API

- [ ] `GET /api/crypto/watchlist?category=default` returns:
  - [ ] `assets` array with 5 cryptos
  - [ ] Each asset has `symbol`, `price`, `change_24h_pct`
- [ ] `GET /api/crypto/watchlist?category=meme` returns 8+ meme coins


### UI Tests

- [ ] Open `http://localhost:5000/cockpit` in browser
- [ ] "Ghost Prediction" panel visible
- [ ] Stocks | Crypto tabs present
- [ ] Chart displays without errors
- [ ] No "sample" or "demo" data in DOM
- [ ] All timestamps show local time (not UTC raw)
- [ ] No 1970 or 1969 dates visible


### Metrics Tests

- [ ] `GET /metrics` includes:
  - [ ] `ghost_crypto_price_fetch_total`
  - [ ] `ghost_crypto_predict_seconds_bucket`
  - [ ] `ghost_prediction_mape{asset_class="crypto"}`
  - [ ] `ghost_up = 1`
  - [ ] `ghost_snapshot_asof` advances on each /api/cockpit call


### Telegram Tests (if configured)

- [ ] `/alerts/test` sends card from Railway deployment
- [ ] `/status` command replies with health info
- [ ] `/signal` command replies with stock + crypto signals
- [ ] `/pnl` command replies with portfolio NAV
- [ ] Free-form question gets coherent AI answer


______________________________________________________________________

## 🚨 BLOCKERS & ISSUES

### Current Blockers

- ❌ UI not yet updated (Phase 3 pending)
- ❌ Telegram commands not yet wired (Phase 5 pending)
- ❌ Time fixes not audited (Phase 6 pending)


### Known Issues

- ⚠️ WOLF ticker delisted → portfolio shows stale price
- ⚠️ Yahoo Finance rate limiting (429 errors)
- ⚠️ Railway caching may show old UI


### Mitigation

- Use `prev_close` fallback for WOLF
- Implement quorum with AlphaVantage → Polygon → Yahoo
- Force Railway cache bust with `?v=timestamp`


______________________________________________________________________

## 📊 PROGRESS TRACKING

**Overall Progress**: 35% Complete

| Phase | Status | Progress | |-------|--------|----------| | Phase 1: Crypto Module | ✅
Complete | 100% | | Phase 2: API Integration | ✅ Complete | 100% | | Phase 3: UI Updates
| 🚧 In Progress | 0% | | Phase 4: Sentiment/Macro | 🚧 Pending | 0% | | Phase 5: Telegram
| 🚧 Pending | 0% | | Phase 6: Time Fixes | 🚧 Pending | 0% | | Phase 7: Testing | 🚧
Pending | 0% | | Phase 8: Documentation | 🚧 Pending | 0% |

**Estimated Time Remaining**: ~3 hours

______________________________________________________________________

## 🔑 RAILWAY ENVIRONMENT VARIABLES CHECKLIST

Copy these to Railway deployment:

```bash

# === CORE CONFIG ===

SIM_MODE=0
LOG_LEVEL=INFO
LOG_JSON=1
PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
ADMIN_IP_ALLOWLIST=127.0.0.1

# === AI CONFIG ===

AI_ON=1
AI_PROVIDER=openai
AI_MODEL=gpt-4o-mini
OPENAI_API_KEY=<your-key>

# === FEATURE FLAGS ===

NEWS_SENTIMENT_ON=1
FUSION_AI_ON=1
MACRO_BRAIN_ON=1

# === CRYPTO CONFIG ===

CRYPTO_ENABLED=1
CRYPTO_SYMBOLS=BTC,ETH,SOL,BNB,DOGE,SHIB,PEPE,AVAX,DOT,MATIC,LINK,UNI,AAVE
CRYPTO_LOOKBACK_H=96
CRYPTO_FORECAST_H=48
CRYPTO_PRICE_SOURCE=coingecko
CRYPTO_QUORUM=coingecko,binance,coinbase

# === TELEGRAM (Optional) ===

TELEGRAM_BOT_TOKEN=<your-bot-token>
TELEGRAM_CHAT_ID=<your-chat-id>

# === STOCK PRICE PROVIDERS ===

ALPHA_VANTAGE_API_KEY=<your-key>
POLYGON_API_KEY=<your-key>
POLYGON_KEY=<your-key>

# === SECURITY ===

GHOST_API_TOKEN=<your-secret-token>

# === OPTIONAL ===

GHOST_TZ=America/New_York
GHOST_CLOCK_24H=0

```text

### Missing Secrets Check

- [ ] OPENAI_API_KEY - Required for AI_ON=1
- [ ] ALPHA_VANTAGE_API_KEY - Recommended for price quorum
- [ ] POLYGON_API_KEY - Recommended for news sentiment
- [ ] GHOST_API_TOKEN - Recommended for write endpoint protection
- [ ] TELEGRAM_BOT_TOKEN - Optional, for Telegram integration
- [ ] TELEGRAM_CHAT_ID - Optional, for Telegram integration


______________________________________________________________________

## 🎉 COMPLETION CRITERIA

### Definition of Done

1. ✅ All 8 phases completed
2. ✅ All acceptance tests pass
3. ✅ No blockers or critical issues
4. ✅ Documentation updated
5. ✅ Deployed to Railway successfully
6. ✅ Production verification complete
7. ✅ CHANGELOG entry created
8. ✅ Team notified


### Sign-off

- [ ] Technical lead approval
- [ ] QA verification
- [ ] Production deployment
- [ ] User acceptance testing


______________________________________________________________________

**Last Updated**: October 12, 2025 **Next Review**: After Phase 3 (UI Updates)
completion
