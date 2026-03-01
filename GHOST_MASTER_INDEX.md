# 🧬 GHOST PROTOCOL — MASTER SYSTEM INDEX

**Last Updated:** March 1, 2026  
**Purpose:** Single source of truth for ALL systems, their status, and where to find them.  
**Rule:** This file gets updated with every structural change.  
**Supersedes:** GHOST_INDEX.md, GHOST_CODEBASE_MAP.md, GHOST_BODY_MAP.md, all prior indexes.

---

## 📊 SYSTEM HEALTH AT A GLANCE

| System | Status | Location | Lines |
|--------|--------|----------|-------|
| 🐺 Wolf Core | ✅ LIVE | `wolf_app.py` | ~45,280 |
| 🧠 Intelligence Hub | ✅ LIVE | `core/intelligence_hub.py` | ~950 |
| 📰 News Brain | ✅ LIVE (internal only) | `core/intelligence/ghost_news_brain.py` | ~950 |
| 🔮 Ghost Scout | ✅ LIVE | `core/ghost_scout.py` | varies |
| 📈 Stock Engine | ✅ LIVE | `core/stock_engine.py` | varies |
| 🎯 Edge Whitelist | ✅ LIVE | `config/symbols.py` | ~50 |
| 🧪 Hub Tests | ✅ 38/38 | `tests/test_intelligence_hub.py` | ~400 |
| ⏰ Auto-Prediction | ✅ LIVE | `core/auto_prediction_loop.py` | varies |
| 📊 Beast Scheduler | ✅ LIVE | `core/beast_scheduler.py` | ~440 |
| 🛡️ Guardian Oracle | ✅ LIVE | `core/guardian_oracle.py` | varies |
| 🔔 Notification System | ✅ LIVE | `core/notifications.py` | varies |

---

## 🧠 INTELLIGENCE HUB — CENTRAL NERVOUS SYSTEM

**File:** `core/intelligence_hub.py`  
**Purpose:** Aggregates ALL 20 intelligence systems into one decision pipeline.

### What It Does
1. Runs 13 signal checkers (news, ML, patterns, ensemble, etc.)
2. Applies 3 post-processors (trust ladder, quality gate, market regime)
3. Runs 3 safety gates (killswitch, drawdown, exposure)
4. Produces: direction adjustment (CONFIRM/FLIP/WEAKEN/BLOCK), confidence delta, trust boost
5. Self-improvement engine runs every 6 hours

### 20 Intelligence Systems

| # | System | Weight | Status | Source |
|---|--------|--------|--------|--------|
| 1 | News Brain | 0.20 | ✅ WIRED | `ghost_news_brain.py` → hub cache |
| 2 | Ensemble Model | 0.20 | ✅ WIRED | `core/ensemble_model.py` |
| 3 | ML/XGBoost | 0.15 | ✅ WIRED | `models/` directory |
| 4 | Opus Brain | 0.12 | ✅ WIRED | `core/intelligence/opus_brain.py` |
| 5 | Pattern Intelligence | 0.10 | ✅ WIRED | `core/pattern_intelligence/` |
| 6 | Ensemble Forecaster | 0.10 | ✅ WIRED | `core/crypto/ensemble_forecaster.py` |
| 7 | Ghost Intel Sources | 0.05 | ✅ WIRED | `ghost_intel/sources.py` |
| 8 | Ghost Intel Integration | 0.04 | ✅ WIRED | `ghost_intel/integration.py` |
| 9 | Impact Model | 0.04 | ✅ WIRED | `ghost_intel/impact_model.py` |
| 10 | Trust Ladder | post | ✅ WIRED | Hub internal (multiplier→delta) |
| 11 | Quality Gate | post | ✅ WIRED | Hub internal (advisory only) |
| 12 | Market Regime | post | ✅ WIRED | Hub internal |
| 13 | Killswitch | gate | ✅ WIRED | Env `PREDICTIONS_ENABLED` |
| 14 | Drawdown Gate | gate | ✅ WIRED | Hub internal |
| 15 | Exposure Gate | gate | ✅ WIRED | Hub internal |
| 16 | Dynamic Exit System | output | ✅ WIRED | Returns SL/TP levels |
| 17 | Self-Improvement | loop | ✅ WIRED | Runs every 6 hours |
| 18 | Guardian Alerts | event | ✅ WIRED | News brain creates on CRITICAL |
| 19 | Auto-Pause | event | ✅ WIRED | 4-hour pause on CRITICAL news |
| 20 | Direction Normalizer | util | ✅ WIRED | UP/DOWN ↔ BUY/SELL conversion |

### Hub Entry Points (3 engines wired)

| Engine | Location | Hub Call |
|--------|----------|----------|
| Ghost Scout | `core/ghost_scout.py` → `_make_prediction()` | `hub.analyze(symbol, ...)` |
| Stock Engine | `wolf_app.py` ~line 8290 | `hub.analyze(symbol, ...)` |
| Turbo/Crypto Engine | `wolf_app.py` ~line 9667 | `hub.analyze(symbol, ...)` |

---

## 📰 NEWS BRAIN — INTERNAL ONLY (NO TELEGRAM)

**File:** `core/intelligence/ghost_news_brain.py`

### Data Flow
```
RSS Feeds + CryptoPanic → Claude Analysis → Intelligence Hub Cache → Prediction Adjustments
     (14+ feeds)            (every 30 min)      (update_news_brain_cache)    (direction/confidence)
```

### What Ghost Does With News
- **CRITICAL event** → Auto-pauses trading 4 hours + guardian alerts
- **HIGH event** → Guardian alerts for affected symbols
- **Predictions at risk** → Hub adjusts direction (FLIP/WEAKEN) and confidence
- **NO Telegram** → Ghost uses news internally, user gets nothing

### All News → Hub Feed Points
| Location | File |
|----------|------|
| 30-min news loop | `wolf_app.py` ~line 4996 |
| `/api/v3/news/analyze` | `wolf_app.py` ~line 43528 |
| `/api/v3/news/analyze-with-auto-pause` | `wolf_app.py` ~line 43640 |
| Beast scheduler | `core/beast_scheduler.py` ~line 268 |

---

## 🎯 EDGE WHITELIST — 13 PROVEN SYMBOLS

**File:** `config/symbols.py`

```
STOCKS (7):  PANW, NET, FTNT, DDOG, T, BMBL, XPO
CRYPTO (6):  ETH, XRP, LINK, CHZ, BTC, SOL
```

---

## ⏰ ALL BACKGROUND TASKS (34 total)

### Startup (one-shot)
| Task | What |
|------|------|
| Pre-populate `_LATEST_PREDICTIONS` | Load from DB |
| Startup predictions | Run edge symbols on boot |
| Intelligence Hub pre-init | Initialize singleton |

### Always Running
| Task | Interval | What |
|------|----------|------|
| Auto-Prediction Loop | 60min mkt / 120min off | Core predictions for all edge symbols |
| News Brain Analysis | 30 min | Claude analysis → Hub cache |
| Self-Improvement | 6 hours | Auto-tune thresholds |
| Notification Loop | 60s tick | TOP 10 at 8AM, watchdog 15min |
| Signal Dispatcher | 1 hour | Telegram dispatch check |
| Confidence Calibration | 6 hours | Rebuild calibration curves |
| Accuracy Evaluator | 1 hour | Evaluate outcomes → feedback |
| Paper Trade Reconciler | 15 min | Resolve due paper trades |
| Watchlist Scheduler | 1 min check | Predictions at open/close |
| Outcome Reconciler V2 | 1 hour | Check if predictions correct |
| Learning Cycle | 1 hour | Weight adjustments from outcomes |
| V2 Quality Updater | 24 hours | Symbol whitelist/blacklist |
| Auto-Calibrate | Sunday 5AM CT | Full strategy calibration |
| ML Retrain | 14 days | Retrain XGBoost |
| Online Calibrator | 6 hours | Horizon/strategy recalibration |

### Worker Mode Only
| Task | Interval | What |
|------|----------|------|
| Price Recorder | 60s | Record prices for touch-target |
| VIP Scanner | 60s | Scan microcap coins |
| Premarket Predictor | 7AM CT weekdays | Pre-market predictions |
| Full Market Scanner | 5AM CT + hourly | Full scan + movers |
| Auto-Execution | 5 min (if enabled) | Execute trades |
| Telegram Reports | 7AM + 8PM | Morning + evening reports |
| Money Game | 6AM / 6PM CT | Scout + resolve |
| Guardian Monitor | 5 min | 24/7 position monitoring |
| Movers Scanner | 30 min | Real-time biggest movers |
| Cascade Scheduler | 10 min | Pending cascade updates |
| Performance Dashboard | 1 hour | Monitoring + win rate alerts |
| Daily Briefing | 6AM CT | Top 5 picks |
| Morning Prophecy | 6AM CT | Guardian Oracle format |

---

## 📲 ALL TELEGRAM SEND PATHS

### ✅ Trade Signals (keep)
| Source | What | Trigger |
|--------|------|---------|
| Notification Loop | TOP 10 (5 stocks + 5 crypto) | 8AM CT daily |
| Watchdog | Target/stop-loss hit alerts | Every 15 min |
| `telegram_alerts.py` | Individual signals | On prediction (max 10/day) |

### ✅ System Alerts (keep)
| Source | What | Trigger |
|--------|------|---------|
| Guardian Oracle | Critical system alerts | Every 5 min |
| Risk Manager | Circuit breaker, kill switch | On event |
| Regime Filter | Regime change | On event (throttled 1hr) |

### ✅ Scheduled Reports (keep)
| Source | What | Schedule |
|--------|------|----------|
| Morning Prophecy | Guardian Oracle TOP 10 | 6AM CT |
| Daily Briefing | Top 5 picks | 6AM CT |
| Telegram Hunter | Morning + Evening | 7AM / 8PM CT |
| Auto-Calibrate | Calibration results | Sunday 5AM CT |

### ❌ News Dumps (ALL KILLED)
| Source | Status |
|--------|--------|
| `send_alert()` | ❌ Logs internally only |
| `handle_critical_event()` | ❌ Keeps guardian alerts, no Telegram |
| Beast scheduler `brain.send_alert()` | ❌ Removed, feeds Hub instead |
| Wolf news loop `brain.send_alert()` | ❌ Removed, logs risk internally |

---

## 📁 CORE MODULE INVENTORY (~160 files)

### Prediction / Forecasting
| File | Purpose |
|------|---------|
| `core/auto_prediction_loop.py` | Main 24/7 adaptive prediction engine |
| `core/cascade_predictor.py` | Multi-stage cascading predictions |
| `core/daily_predictions_engine.py` | 6AM daily top 5 picks |
| `core/data_pillar_predictor.py` | Data-pillar predictions |
| `core/ensemble_model.py` | Ensemble model predictor |
| `core/multi_forecast_engine.py` | Multi-model forecast |
| `core/multi_horizon.py` | Multi-timeframe horizons |
| `core/prediction_store.py` | Prediction storage |
| `core/prediction_tracker.py` | Accuracy tracking |
| `core/prediction_outcome_evaluator.py` | Outcome evaluation |
| `core/prediction_confidence_calibrator.py` | Confidence calibration |
| `core/predictions_killswitch.py` | Emergency kill switch |

### Intelligence / Brain
| File | Purpose |
|------|---------|
| `core/intelligence_hub.py` | **Central hub — all 20 systems** |
| `core/ghost_brain.py` | Core AI decision making |
| `core/ghost_scout.py` | Asset scouting |
| `core/world_context_engine.py` | World context awareness |
| `core/intelligence/ghost_news_brain.py` | News analysis via Claude |
| `core/intelligence/opus_brain.py` | Opus-level AI analysis |

### Market Analysis
| File | Purpose |
|------|---------|
| `core/market_mood.py` | Sentiment/mood detection |
| `core/market_regime.py` | Regime classification |
| `core/market_scanner.py` | Full market scanning |
| `core/market_gates.py` | Trading eligibility |
| `core/momentum_detector.py` | Momentum shifts |
| `core/spike_detector.py` | Price spikes |
| `core/technical_indicators.py` | Technical indicators |

### Scanning / Ranking
| File | Purpose |
|------|---------|
| `core/full_market_scanner.py` | 5AM full scan + hourly movers |
| `core/daily_top10_scanner.py` | Daily TOP 10 scanner |
| `core/mover_scanner.py` | Real-time biggest movers |
| `core/opportunity_scorer.py` | Opportunity scoring |

### Accuracy / Learning
| File | Purpose |
|------|---------|
| `core/accuracy_tracker.py` | Core accuracy |
| `core/learning_feedback_loop.py` | Weight adjustments |
| `core/autonomous_improvement.py` | 6h improvement cycles |
| `core/online_calibrator.py` | Calibration |
| `core/auto_calibrate.py` | Auto-calibration engine |

### ML / Models
| File | Purpose |
|------|---------|
| `core/xgboost_model.py` | XGBoost training + retrain |
| `core/model_store.py` | Model persistence |
| `models/xgboost_v2.joblib` | Production model |

### Trading / Execution
| File | Purpose |
|------|---------|
| `core/autonomous_trader.py` | Phase 5 auto-execution |
| `core/production_trading.py` | Production trading |
| `core/order_manager.py` | Order management |
| `core/position_manager.py` | Position management |
| `core/sl_tp_monitor.py` | SL/TP monitor |
| `core/alpaca_broker.py` | Alpaca integration |

### Risk
| File | Purpose |
|------|---------|
| `core/risk/risk_engine.py` | Core risk engine |
| `core/risk/risk_manager.py` | Risk with alerts |

### Price Providers
| File | Purpose |
|------|---------|
| `core/providers/price_quorum.py` | Multi-provider quorum |
| `core/providers/coinbase.py` | Coinbase |
| `core/providers/polygon.py` | Polygon.io |
| `core/providers/yahoo_finance.py` | Yahoo Finance |
| `core/providers/binance_ohlcv.py` | Binance OHLCV |

### Pattern Intelligence
| File | Purpose |
|------|---------|
| `core/pattern_intelligence/engine.py` | Pattern matching |
| `core/pattern_intelligence/btc_correlation.py` | BTC correlation |
| `core/pattern_intelligence/fear_greed.py` | Fear & Greed |
| `core/pattern_intelligence/social_sentiment.py` | Social sentiment |

### Data Pillars
| File | Purpose |
|------|---------|
| `core/data_pillars/price_engine.py` | Price data |
| `core/data_pillars/sentiment_engine.py` | Sentiment data |
| `core/data_pillars/technical_engine.py` | Technical analysis |
| `core/data_pillars/volume_engine.py` | Volume data |
| `core/data_pillars/world_engine.py` | World context |

### Ghost Intel
| File | Purpose |
|------|---------|
| `ghost_intel/sources.py` | External data APIs |
| `ghost_intel/integration.py` | Intel rules engine |
| `ghost_intel/taxonomy.py` | Asset classification |
| `ghost_intel/impact_model.py` | News impact scoring |

### Notifications
| File | Purpose |
|------|---------|
| `core/notifications.py` | TOP 10 consolidated |
| `core/telegram_alerts.py` | Formatted opportunities |
| `core/alert_pipeline.py` | Dedup + smart cap |
| `notifications/telegram.py` | Transport layer |

### Services
| File | Purpose |
|------|---------|
| `services/predictor.py` | Core prediction service |
| `services/outcome_reconciler_v2.py` | V2 outcome reconciler (active) |
| `services/actual_price_collector.py` | Price collection |

### Config
| File | Purpose |
|------|---------|
| `config/symbols.py` | Edge whitelist (13 symbols) |
| `config/settings.py` | App settings |

---

## 🗄️ DATABASE TABLES

### Core
| Table | Purpose |
|-------|---------|
| `predictions` | Core predictions |
| `prediction_outcomes` | Outcomes |
| `ghost_predictions` | Ghost predictions cache |
| `ghost_symbol_accuracy` | Per-symbol accuracy |
| `price_actuals` | Actual prices |
| `paper_trades` | Paper trades |
| `active_positions` | Active positions |

### Forecast
| Table | Purpose |
|-------|---------|
| `forecast_48h` | 48-hour forecasts |
| `forecasts` | Forecast records |
| `forecast_scores` | Accuracy scores |

### Learning
| Table | Purpose |
|-------|---------|
| `learned_weights` | Feature weights |
| `signal_performance` | Signal performance |
| `calibrator_*` (4 tables) | Calibration data |

### News / Guardian
| Table | Purpose |
|-------|---------|
| `news_analysis` | News analysis results |
| `guardian_alerts` | Guardian alerts |
| `system_state` | System state |

### System
| Table | Purpose |
|-------|---------|
| `state` | Key-value store |
| `api_keys` | API keys |
| `v2_quality_config` | Quality config |
| `ghost_symbol_trust` | Trust scores |

---

## 📡 KEY API ENDPOINTS

### Predictions
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v3/predictions/latest` | GET | All predictions (limit=25, edge-first) |
| `/api/predict/run?symbol=ETH` | GET | Trigger prediction |

### Intelligence
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v3/intelligence/status` | GET | Hub status + news cache age |
| `/api/v3/news/analyze` | GET | Trigger news → Hub |
| `/api/v3/news/analyze-with-auto-pause` | POST | Analysis + auto-pause |
| `/api/v3/trading/pause-status` | GET | Pause status |

### Health
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health (includes git_sha) |
| `/readiness` | GET | Readiness probe |

---

## 🏗️ DEPLOYMENT

| Setting | Value |
|---------|-------|
| Platform | Railway |
| URL | `https://ghost-protocol-production.up.railway.app` |
| Auto-deploy | ✅ On push to `main` |
| Repository | `seancole713-source/ghost-protocol` |

### Required Env Vars
| Var | Status | Purpose |
|-----|--------|---------|
| `ANTHROPIC_API_KEY` | ✅ Set | Claude for News Brain |
| `OPENAI_API_KEY` | ✅ Set | GPT fallback |
| `TELEGRAM_BOT_TOKEN` | ✅ Set | Trade alerts only |
| `TELEGRAM_CHAT_ID` | ✅ Set | Chat target |
| `EDGE_WHITELIST_ENABLED` | ✅ =1 | 13 proven symbols |

---

## 🔧 ALL FIXES THIS SESSION

| # | Commit | Fix |
|---|--------|-----|
| 1 | `d54e5de` | Intelligence Hub created — 20 systems → 1 pipeline |
| 2 | `ff5c883` | Wired into Stock + Turbo engines |
| 3 | `6ef668e` | Quality Gate advisory-only |
| 4 | `cf405b3` | Trust ladder multiplier→delta |
| 5 | `274313c` | Confidence cap 0.85 not 0.92 |
| 6 | `c022932` | Dedup cache fix |
| 7 | `610e291` | Double trust ladder removed |
| 8 | `058d21f` | API limit 10→25 + edge sort |
| 9 | `b133eb2` | Stop Telegram news spam |
| 10 | (pending) | Beast scheduler + full retroactive index |

---

## 📋 PREVIOUS INDEXES (all superseded)

| File | Date |
|------|------|
| `GHOST_INDEX.md` | Jan 27, 2026 |
| `GHOST_CODEBASE_MAP.md` | Jan 28, 2026 |
| `GHOST_BODY_MAP.md` | Jan 7, 2026 |
| `GHOST_BLUEPRINT.md` | Dec 11, 2025 |
| `GHOST_BASELINE_MANIFEST.md` | Dec 12, 2025 |
| `AUTOPSY_INDEX.md` | Dec 8, 2025 |
| `GHOST_PREDICTION_PIPELINE.md` | Dec 13, 2025 |

---

*This index is the single source of truth. Updated with every structural change.*
