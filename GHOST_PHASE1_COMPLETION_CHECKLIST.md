# Ghost Protocol V3 - Phase 1 Completion Checklist

**Target**: Production-ready system with full data pipeline
**Status**: Cluster A Complete (Watchlist: 82 symbols in WatchlistManager, 25 in SmartWatcher)

## Backend Health & Endpoints

- [ ] `/health` returns 200 with correct JSON
- [ ] `/api/v3/cockpit/status` returns ghost_score and health flags
- [ ] `/api/v3/hunter/feed` returns crypto/stock movers with prices
- [ ] `/api/v3/goals/snapshot` returns ghost_score_v2 + goals
- [ ] `/api/v3/news/feed` returns news items (not empty)
- [x] `/api/v3/watchlist` returns 25+ symbols (PENDING RAILWAY VERIFICATION)
- [ ] `/api/v3/predictions/latest` returns predictions with confidence > 0
- [ ] `/api/v3/predictions/recent` returns recent predictions
- [ ] `/api/v3/predictions/history` returns prediction history
- [ ] `/api/v3/accuracy/summary` returns accuracy metrics
- [ ] `/api/v3/providers/health` returns provider status


## Cockpit V3 UI

- [ ] Top Movers panel loads crypto/stock prices
- [ ] Forecast panel shows direction + confidence > 0
- [ ] Prediction Accuracy panel shows real numbers
- [ ] News Feed shows articles (when available)
- [ ] Watchlist displays symbols
- [ ] Ghost Score matches backend (37-92 range)
- [ ] Goals progress shows percentages


## Prediction System

- [ ] Feature extraction gets 20+/25 features (not 3)
- [ ] Confidence values 40-85% (not 0%)
- [ ] Direction: UP/DOWN/FLAT based on real signals
- [ ] Predictions for multiple symbols (not just WOLF)
- [ ] Database has 30+ predictions across symbols


## Data Providers

- [ ] Crypto providers: BTC, ETH, SOL, BNB working
- [ ] Stock providers: AAPL, MSFT, NVDA working (if Polygon key exists)
- [ ] Provider health tracked (not all "unknown")
- [ ] Failures handled gracefully (no crashes)


## Railway + Docker

- [ ] Production build succeeds
- [ ] Railway healthcheck passes
- [ ] No crash loops in logs
- [ ] Volume persistence working


## Blockers (External Dependencies)

- ⚠️ News API key (Alpha Vantage/Finnhub) - if missing, news will be empty
- ⚠️ Polygon stock key - if missing, stock predictions limited
- ⚠️ ML model artifacts - if missing, use rule-based predictions


**Status**: Starting Phase 1 fixes...
