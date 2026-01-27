# 🎯 GHOST PROTOCOL - MASTER NAVIGATION GUIDE

**Last Updated:** January 27, 2026  
**Purpose:** OCD-level organized reference for navigating the Ghost codebase

---

## 📊 CODEBASE STATISTICS

| Metric | Count |
|--------|-------|
| **wolf_app.py** | 40,269 lines |
| **core/ modules** | 169 files |
| **services/** | 5 files |
| **ghost_intel/** | 9 files |
| **Root .py files** | 154 files |
| **Documentation (.md)** | 555 files |

---

## 🏗️ ARCHITECTURE OVERVIEW

```
ghost-protocol/
├── wolf_app.py              # 🐺 MAIN APPLICATION (40K lines - FastAPI)
├── core/                    # 🧠 CORE BUSINESS LOGIC (169 modules)
│   ├── Prediction Engine
│   ├── Trading Systems
│   ├── Intel & Analysis
│   └── Infrastructure
├── services/                # ⚙️ BACKGROUND SERVICES (5 files)
├── ghost_intel/             # 🔍 INTELLIGENCE MODULE (9 files)
└── *.md                     # 📚 DOCUMENTATION (555 files)
```

---

## 🐺 WOLF_APP.PY - SECTION MAP

The main 40K line file is organized into these sections:

| Line Range | Section | Purpose |
|------------|---------|---------|
| 1-750 | **Imports & Config** | Environment, logging, globals |
| 750-2400 | **Authentication** | JWT, Bearer tokens, auth deps |
| 2400-4600 | **Startup & Background Tasks** | App lifecycle, schedulers |
| 4600-7000 | **Core Prediction Loop** | Auto-prediction, notification loop |
| 7000-9400 | **Metrics & Monitoring** | Prometheus, health checks |
| 9400-10100 | **Stock Engine Endpoints** | `/api/v3/stock/*` |
| 10100-12200 | **Prediction Endpoints** | `/api/v3/predictions/*` |
| 12200-14400 | **Accuracy & Evaluation** | Outcome tracking, verification |
| 14400-19000 | **Watchlist & Alerts** | Telegram, watchlist management |
| 19000-23500 | **Telegram Formatting** | Message builders, alert sending |
| 23500-26500 | **V2 Quality System** | Whitelist/blacklist management |
| 26500-33000 | **Trading & Execution** | Paper trades, live execution |
| 33000-40000 | **Utility & Debug** | Scanner, debug endpoints |

---

## 🧠 CORE MODULES - CATEGORIZED

### 📈 PREDICTION ENGINE (The Brain)
| File | Purpose | Key Function |
|------|---------|--------------|
| `stock_engine.py` | Stock-specific predictions | `predict(symbol, bypass_calendar)` |
| `ensemble_predictor.py` | ML ensemble (LSTM+XGBoost+Transformer) | `get_ensemble_predictor()` |
| `cascading_predictor.py` | Multi-stage prediction cascade | Stage 1-6 flow |
| `daily_predictions_engine.py` | Daily batch predictions | `run_daily_scan()` |
| `multi_horizon_forecaster.py` | 6h/24h/48h forecasting | Time-horizon logic |

### 🧪 QUALITY & FILTERING
| File | Purpose | Key Function |
|------|---------|--------------|
| `v2_quality.py` | **V2 Quality System** | `should_predict()`, whitelist/blacklist |
| `quality_gate.py` | Quality gates | Confidence thresholds |
| `v2_pick_filter.py` | Pick filtering | Symbol filtering |
| `v2_verification.py` | Outcome verification | Win rate tracking |

### 🔒 MARKET GATES (Safety Checks)
| File | Purpose | Key Function |
|------|---------|--------------|
| `market_gates.py` | VIX, SPY regime checks | `check_vix()`, `check_spy_regime()` |
| `economic_calendar.py` | FOMC/CPI/NFP blackouts | `is_fomc_blackout()` |
| `stock_gates.py` | Stock-specific gates | Market hours, earnings |
| `sector_momentum.py` | Sector rotation | Sector strength scoring |

### 🔍 INTEL MODULE (Ghost Intel)
| File | Purpose | Key Function |
|------|---------|--------------|
| `ghost_intel/integration.py` | Intel rules application | `apply_intel_to_prediction()` |
| `ghost_intel/taxonomy.py` | Asset themes (AI/GPU) | Theme classification |
| `ghost_intel/sources.py` | Data sources | Polygon, Yahoo, FRED |
| `ghost_intel/impact_model.py` | News impact scoring | Sentiment analysis |

### 💰 TRADING & EXECUTION
| File | Purpose | Key Function |
|------|---------|--------------|
| `alpaca_broker.py` | Alpaca API integration | Live trading |
| `paper_tracker.py` | Paper trading | Simulated trades |
| `position_manager.py` | Position tracking | Open/close positions |
| `risk_manager.py` | Risk limits | Stop loss, take profit |
| `order_manager.py` | Order lifecycle | Order routing |

### 📊 ACCURACY & TRACKING
| File | Purpose | Key Function |
|------|---------|--------------|
| `accuracy_tracker.py` | Win rate tracking | Per-symbol accuracy |
| `postgres_accuracy.py` | PostgreSQL persistence | DB storage |
| `prediction_evaluator.py` | Outcome evaluation | Did prediction hit? |
| `target_touch_evaluator.py` | Target touch tracking | Price target hits |

### 📱 ALERTS & NOTIFICATIONS
| File | Purpose | Key Function |
|------|---------|--------------|
| `telegram_alerts.py` | Telegram sending | `send_alert()` |
| `ghost_notifications.py` | Notification routing | Multi-channel |
| `alert_manager.py` | Alert queue | Rate limiting |
| `top10_aggregator.py` | Daily TOP 10 | Aggregated alerts |

### 💾 DATA & STORAGE
| File | Purpose | Key Function |
|------|---------|--------------|
| `prediction_store.py` | Prediction persistence | SQLite/PostgreSQL |
| `cache_manager.py` | In-memory caching | TTL cache |
| `db_engine.py` | Database connections | Connection pooling |
| `price_quorum.py` | Multi-source price | Price consensus |

---

## ⚙️ SERVICES MODULE

| File | Purpose |
|------|---------|
| `predictor.py` | Main prediction service |
| `outcome_reconciler.py` | V1 outcome checker |
| `outcome_reconciler_v2.py` | V2 outcome checker (yfinance fallback) |
| `actual_price_collector.py` | Historical price collection |

---

## 🔑 KEY CONFIGURATION FILES

| File | Purpose |
|------|---------|
| `ghost_v2_quality.json` | V2 whitelist/blacklist/trial_stocks |
| `.env` | Environment variables (secrets) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container build |
| `railway.json` | Railway deployment config |

---

## 🔌 API ENDPOINT MAP

### Health & Status
```
GET  /health                      → System health check
GET  /api/v2/quality/status       → V2 quality filter status
GET  /api/v3/predictions/latest   → Latest cached predictions
```

### Stock Predictions
```
GET  /api/v3/stock/predict/{symbol}?bypass_calendar=true  → Stock prediction
GET  /api/v3/stock/debug/{symbol}                         → Debug price data
GET  /api/v3/stock/batch?symbols=NVDA,AMD                 → Batch predictions
```

### Crypto Predictions
```
GET  /api/crypto/predict/{symbol}  → Crypto prediction
GET  /api/v3/opus/predict/{symbol} → Full prediction with Intel
```

### V2 Quality System
```
GET  /api/v2/quality/status                    → Current whitelist/blacklist
POST /api/v2/quality/reload                    → Reload from JSON
GET  /api/v2/quality/test-should-predict?symbol=NVDA&confidence=75
```

### Alerts
```
POST /alerts/test?send=true                   → Test Telegram
POST /alerts/predictions/send                 → Full prediction alert
POST /alerts/top10/force                      → Force TOP 10 send
```

### Paper Trading
```
GET  /api/v3/paper/trades?limit=10            → Recent paper trades
GET  /api/v3/paper/summary                    → Paper trading summary
```

---

## 🐛 KNOWN BUG PATTERNS

### Pattern 1: Missing Except Block
```python
# WRONG - causes syntax error
try:
    do_something()
# No except!

@APP.get("/next-endpoint")  # SyntaxError here
```

### Pattern 2: Price Field Naming
```python
# Different modules use different field names:
pred.get("entry_price")        # Some places
pred.get("price_at_prediction") # Other places
pred.get("price")              # Legacy
```

### Pattern 3: Symbol Format (Crypto)
```python
# Polygon: "X:BTCUSD"
# CoinGecko: "bitcoin"
# Internal: "BTC"
# Always normalize!
```

---

## 🔄 PREDICTION FLOW

```
1. Trigger
   └── /api/v3/watchlist/trigger-prediction OR 6AM cron

2. V2 Quality Gate
   └── core/v2_quality.py::should_predict()
       ├── Whitelist? → Proceed
       ├── Trial Stock? → Check 70% confidence
       ├── Blacklist? → BLOCK
       └── Unknown? → BLOCK (strict mode)

3. Market Gates
   └── core/economic_calendar.py::economic_calendar_gate()
       ├── FOMC blackout? → BLOCK
       ├── CPI blackout? → BLOCK
       └── NFP blackout? → BLOCK

4. Price Data
   └── core/stock_engine.py::_get_technical_indicators()
       ├── Polygon API (primary)
       └── yfinance (fallback)

5. Ensemble Prediction
   └── core/ensemble_predictor.py::predict()
       ├── LSTM model
       ├── XGBoost model
       └── Transformer model
       → Weighted average

6. Intel Boost
   └── ghost_intel/integration.py::apply_intel_to_prediction()
       ├── Theme match? +5-8% confidence
       ├── 2025 Winner? +5%
       └── Sector leader? +3%

7. Confirmation Count
   └── RSI, MACD, Bollinger, Volume
       → Need 3+ confirmations

8. Output
   └── Direction, Confidence, Entry, Target, Stop Loss
```

---

## 📋 QUICK COMMANDS

```bash
# Check deployment
curl -s https://ghost-protocol-production.up.railway.app/health | python3 -m json.tool

# Test stock prediction
curl -s "https://ghost-protocol-production.up.railway.app/api/v3/stock/predict/NVDA?bypass_calendar=true" | python3 -m json.tool

# Check V2 quality
curl -s "https://ghost-protocol-production.up.railway.app/api/v2/quality/status" | python3 -m json.tool

# Send test alert
curl -s -X POST "https://ghost-protocol-production.up.railway.app/alerts/test?send=true"

# Send prediction alert
curl -s -X POST "https://ghost-protocol-production.up.railway.app/alerts/predictions/send"

# Reload V2 quality from JSON
curl -s -X POST "https://ghost-protocol-production.up.railway.app/api/v2/quality/reload"

# Syntax check wolf_app.py
python3 -m py_compile wolf_app.py && echo "OK"

# Push to production
git add -A && git commit -m "message" && git push
```

---

## 🔧 COMMON FIXES

### Fix 1: Syntax Error in wolf_app.py
```bash
python3 -m py_compile wolf_app.py
# Look at line number, usually missing except/finally
```

### Fix 2: Price Data Unavailable
```python
# Add Polygon fallback in stock_engine.py
polygon_key = os.getenv("POLYGON_API_KEY")
# Make async HTTP call to Polygon API
```

### Fix 3: V2 Quality Blocking
```bash
# Add symbol to trial_stocks in ghost_v2_quality.json
# Then reload: POST /api/v2/quality/reload
```

### Fix 4: FOMC Blocking
```python
# Use bypass_calendar=True for testing
await engine.predict(symbol, bypass_calendar=True)
```

---

## 📁 FILE NAMING CONVENTIONS

| Pattern | Meaning |
|---------|---------|
| `*_v2.py` | Version 2 (current) |
| `*_OLD.py` | Deprecated, don't use |
| `*_tracker.py` | State tracking |
| `*_manager.py` | CRUD operations |
| `*_engine.py` | Core processing |
| `*_scheduler.py` | Cron/timing |
| `*_alerts.py` | Notification related |

---

## 🎯 PRIORITY FILES FOR DEBUGGING

1. **wolf_app.py** - Main app, all endpoints
2. **core/v2_quality.py** - Quality filtering
3. **core/stock_engine.py** - Stock predictions
4. **core/ensemble_predictor.py** - ML models
5. **ghost_intel/integration.py** - Intel rules
6. **core/economic_calendar.py** - Market blackouts
7. **services/predictor.py** - Prediction service
8. **ghost_v2_quality.json** - Whitelist/blacklist config

---

## ✅ TODAY'S FIXES (Jan 27, 2026)

| Bug | Root Cause | Fix |
|-----|------------|-----|
| Stocks not predicting | V2 Quality blocking (no trial tier) | Added `trial_stocks` to V2 |
| NVDA blocked | FOMC blackout | Added `bypass_calendar` param |
| Price $0 in alerts | yfinance failing | Added Polygon fallback |
| Crypto price $0 | Wrong field name | Added multiple field fallbacks |
| Deploy failing | Syntax error | Fixed missing except block |
| Trial stocks not syncing | Not in PostgreSQL | Fixed `_save_config()` |

---

**Remember:** When lost, start with `wolf_app.py` line numbers from grep, trace the function to `core/` module, check the data flow.
