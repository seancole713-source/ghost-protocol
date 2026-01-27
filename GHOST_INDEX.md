# 🗂️ GHOST PROTOCOL - COMPLETE INDEX

**THE DEFINITIVE OCD-LEVEL CODEBASE MAP**  
**Last Updated:** January 27, 2026

---

## 📊 INVENTORY SUMMARY

| Category | Count | Status |
|----------|-------|--------|
| Total Python Files | 476 | Mixed (active + legacy) |
| Root Scripts (*.py) | 154 | 🔴 NEEDS CLEANUP |
| Core Modules | 169 | 🟡 Some duplicates |
| Services | 5 | ✅ Active |
| Ghost Intel | 9 | ✅ Active |
| Documentation (*.md) | 556 | 🔴 SPRAWL - needs consolidation |
| Directories | 50+ | 🟡 Some legacy |

---

## 🏗️ DIRECTORY STRUCTURE - DEFINITIVE

```
ghost-protocol/
│
├── 🐺 wolf_app.py           # MAIN APPLICATION - 40,269 lines
│                             # ALL FastAPI endpoints live here
│
├── 📁 core/                  # BUSINESS LOGIC - 169 modules
│   ├── ai_advisor/          # AI advisory system
│   ├── crypto/              # Crypto-specific logic
│   ├── data_pillars/        # Data processing
│   ├── features/            # Feature engineering
│   ├── intelligence/        # Intel processing
│   ├── metrics/             # Metrics collection
│   ├── pattern_intelligence/ # Pattern detection
│   ├── providers/           # Data providers
│   ├── research/            # Research tools
│   ├── risk/                # Risk management
│   └── workers/             # Background workers
│
├── 📁 services/             # BACKGROUND SERVICES - 5 files
│   ├── predictor.py         # Core prediction service
│   ├── outcome_reconciler.py # V1 outcome checker
│   ├── outcome_reconciler_v2.py # V2 with yfinance fallback
│   └── actual_price_collector.py # Price collection
│
├── 📁 ghost_intel/          # INTELLIGENCE MODULE - 9 files
│   ├── sources.py           # External data APIs
│   ├── integration.py       # Intel rules engine
│   ├── taxonomy.py          # Asset classification
│   └── impact_model.py      # News impact scoring
│
├── 📁 routes/               # API ROUTE MODULES
│   └── news_routes.py       # News API routes
│
├── 📁 models/               # ML MODELS
│   ├── ensemble/            # Ensemble model weights
│   ├── production/          # Production models
│   └── trained/             # Training outputs
│
├── 📁 data/                 # DATA STORAGE
├── 📁 logs/                 # LOG FILES
├── 📁 migrations/           # DB MIGRATIONS
├── 📁 templates/            # HTML TEMPLATES
├── 📁 static/               # STATIC ASSETS
├── 📁 tests/                # TEST FILES
├── 📁 scripts/              # UTILITY SCRIPTS
├── 📁 tools/                # DEVELOPER TOOLS
├── 📁 utils/                # UTILITY MODULES
├── 📁 docs/                 # DOCUMENTATION
├── 📁 dashboard/            # NEXT.JS DASHBOARD
└── 📁 .archive/             # ARCHIVED/DEPRECATED
```

---

## 🎯 ACTIVE FILES - WHAT MATTERS

### 🐺 MAIN APPLICATION
| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `wolf_app.py` | 40,269 | Main FastAPI app | ✅ ACTIVE |

### 📦 SERVICES (5 files)
| File | Purpose | Status |
|------|---------|--------|
| `services/predictor.py` | Core prediction engine | ✅ ACTIVE |
| `services/outcome_reconciler.py` | V1 outcome checker | ✅ ACTIVE |
| `services/outcome_reconciler_v2.py` | V2 outcome checker | ✅ ACTIVE |
| `services/actual_price_collector.py` | Price collection | ✅ ACTIVE |

### 🔍 GHOST INTEL (9 files)
| File | Purpose | Status |
|------|---------|--------|
| `ghost_intel/__init__.py` | Package init | ✅ |
| `ghost_intel/sources.py` | Data source APIs | ✅ ACTIVE |
| `ghost_intel/integration.py` | Intel rules engine | ✅ ACTIVE |
| `ghost_intel/taxonomy.py` | Asset themes | ✅ ACTIVE |
| `ghost_intel/impact_model.py` | News impact | ✅ ACTIVE |
| `ghost_intel/market_analysis.py` | Market analysis | ✅ ACTIVE |
| `ghost_intel/news_collector.py` | News aggregation | ✅ ACTIVE |
| `ghost_intel/sentiment.py` | Sentiment analysis | ✅ ACTIVE |
| `ghost_intel/technical.py` | Technical analysis | ✅ ACTIVE |

---

## 🧠 CORE MODULES - CATEGORIZED

### ⭐ CRITICAL (Always Used)
| File | Purpose | Imported By |
|------|---------|-------------|
| `core/ai_memory.py` | AI conversation memory | wolf_app.py |
| `core/concurrency.py` | Rate limiting | wolf_app.py |
| `core/price_quorum.py` | Multi-source pricing | wolf_app.py |
| `core/providers/turbo_provider.py` | Fast price fetcher | wolf_app.py |

### 📈 PREDICTION ENGINE
| File | Purpose | Active? |
|------|---------|---------|
| `core/stock_engine.py` | Stock predictions | ✅ YES |
| `core/ensemble_predictor.py` | ML ensemble | ✅ YES |
| `core/cascading_predictor.py` | Cascade predictions | ✅ YES |
| `core/daily_predictions_engine.py` | Daily batch | ✅ YES |
| `core/multi_horizon_forecaster.py` | Multi-timeframe | ✅ YES |
| `core/prediction_store.py` | Prediction storage | ✅ YES |
| `core/prediction_engine.py` | Legacy engine | ⚠️ MAYBE |
| `core/prediction_evaluator.py` | Outcome eval | ✅ YES |
| `core/prediction_manager.py` | Prediction CRUD | ⚠️ CHECK |

### 🔒 MARKET GATES
| File | Purpose | Active? |
|------|---------|---------|
| `core/market_gates.py` | VIX/SPY checks | ✅ YES |
| `core/economic_calendar.py` | FOMC/CPI blackouts | ✅ YES |
| `core/stock_gates.py` | Stock-specific gates | ✅ YES |
| `core/sector_momentum.py` | Sector rotation | ✅ YES |

### 🧪 V2 QUALITY SYSTEM
| File | Purpose | Active? |
|------|---------|---------|
| `core/v2_quality.py` | Whitelist/blacklist | ✅ CRITICAL |
| `core/v2_pick_filter.py` | Pick filtering | ✅ YES |
| `core/v2_verification.py` | Outcome verification | ✅ YES |
| `core/quality_gate.py` | Quality thresholds | ✅ YES |

### 📊 ACCURACY TRACKING
| File | Purpose | Active? |
|------|---------|---------|
| `core/accuracy_tracker.py` | Win rate tracking | ✅ YES |
| `core/accuracy_tracking.py` | Duplicate? | ⚠️ CHECK |
| `core/accuracy_dashboard.py` | Dashboard data | ✅ YES |
| `core/accuracy_dashboard_v2.py` | V2 dashboard | ⚠️ MAYBE |
| `core/postgres_accuracy.py` | PostgreSQL storage | ✅ YES |
| `core/live_accuracy.py` | Real-time accuracy | ⚠️ CHECK |
| `core/touch_accuracy_metrics.py` | Target touch | ⚠️ CHECK |

### 💰 TRADING
| File | Purpose | Active? |
|------|---------|---------|
| `core/alpaca_broker.py` | Alpaca integration | ✅ YES |
| `core/paper_tracker.py` | Paper trading | ✅ YES |
| `core/position_manager.py` | Position tracking | ✅ YES |
| `core/order_manager.py` | Order lifecycle | ✅ YES |
| `core/trade_executor.py` | Trade execution | ✅ YES |
| `core/trade_manager.py` | Trade CRUD | ⚠️ CHECK |
| `core/trade_signal_manager.py` | Signal management | ⚠️ CHECK |
| `core/trading_orchestrator.py` | Orchestration | ⚠️ CHECK |

### 🔧 RISK MANAGEMENT
| File | Purpose | Active? |
|------|---------|---------|
| `core/risk_manager.py` | Risk limits | ✅ YES |
| `core/risk_engine.py` | Risk calculations | ⚠️ CHECK |
| `core/risk_dashboard.py` | Risk visualization | ⚠️ CHECK |
| `core/execution_risk.py` | Execution risk | ⚠️ CHECK |
| `core/enhanced_risk_shell.py` | Enhanced risk | ⚠️ CHECK |

### 📱 ALERTS
| File | Purpose | Active? |
|------|---------|---------|
| `core/telegram_alerts.py` | Telegram sending | ✅ YES |
| `core/telegram_formatting.py` | Message formatting | ✅ YES |
| `core/ghost_notifications.py` | Multi-channel | ⚠️ CHECK |
| `core/alert_manager.py` | Alert queue | ⚠️ CHECK |
| `core/top10_aggregator.py` | Daily TOP 10 | ✅ YES |

### 💾 DATA & STORAGE
| File | Purpose | Active? |
|------|---------|---------|
| `core/db_engine.py` | Database connections | ✅ YES |
| `core/cache_manager.py` | In-memory cache | ✅ YES |
| `core/portfolio.py` | Portfolio data | ✅ YES |
| `core/portfolio_manager.py` | Portfolio CRUD | ⚠️ CHECK |
| `core/portfolio_persistence.py` | Portfolio storage | ⚠️ CHECK |
| `core/portfolio_tracker.py` | Portfolio tracking | ⚠️ CHECK |
| `core/watchlist_manager.py` | Watchlist CRUD | ✅ YES |
| `core/watchlist_service.py` | Watchlist service | ⚠️ CHECK |
| `core/watchlist_signals.py` | Watchlist signals | ⚠️ CHECK |

---

## 🔴 ROOT SCRIPTS - CLEANUP NEEDED

### ✅ ACTIVE (Keep)
| File | Purpose |
|------|---------|
| `wolf_app.py` | Main application |
| `main.py` | Entry point |
| `db.py` | Database setup |
| `signals.py` | Signal definitions |
| `universe.py` | Asset universe |

### ⚠️ TEST FILES (Move to tests/)
```
test_*.py (50+ files)
quick_test.py
quick_validate.py
```

### ⚠️ MIGRATION SCRIPTS (One-time use)
```
migrate_*.py
apply_*_migration.py
backfill_*.py
trigger_migration.py
```

### ⚠️ DIAGNOSTIC SCRIPTS (Move to tools/)
```
diagnostic_scanner.py
diagnostics.py
railway_diagnostic.py
repair_diagnostic.py
repair_optimize.py
check_*.py
verify_*.py
```

### ⚠️ TRAINING SCRIPTS (Move to scripts/)
```
train_ml_models.py
train_ml_models_v2.py
train_ml_models_v3.py
train_ml_models_v3_hourly.py
train_models_now.py
retrain_model.py
retrain_model_old.py
retrain_monthly.py
```

### 🔴 LIKELY DEAD CODE (Archive)
```
activate_all_systems.py
activate_to_the_moon.py
add_missing_ui_endpoints.py
add_simulation_endpoint.py
add_stocks_to_watchlist.py
add_test_data.py
add_wolf_to_watchlist.py
apply_comprehensive_fixes.py
autofix_startup.py
chat_with_ghost.py
debug_uvicorn.py
deep_dive_audit.py
enhanced_price_fetcher.py
enhanced_rate_limiter.py
explain_issue.py
fix_news_order.py
fix_telegram.py
fix_timezone.py
force_price_refresh.py
generate_ops_report.py
generate_simulation_data.py
generate_status_report.py
ghost_agent_loop.py
ghost_bootstrap.py
ghost_brain_enhanced.py
ghost_diagnostic.py
ghost_init.py
ghost_rpc_guard.py
ghost_state.py
initialize_v2_whitelist.py
inspect_routes.py
intel_smoke_test.py
launch_ghost.py
monitor_prediction_accuracy.py
monitor_telegram_bot.py
ops_worker.py
query_production_winrates.py
reconcile_historical_predictions.py
reconcile_with_coingecko.py
regression_audit.py
reset_telegram_bot_name.py
send_audit_summary.py
send_completion_summary.py
send_issue_found.py
send_real_prediction.py
send_real_prediction_api.py
send_telegram_notification.py
show_v2_status.py
state_manager.py
system_verification.py
telegram_bot_security_integration.py
ui_verification_audit.py
update_whitelist_direct.py
validate_ghost_fixes.py
validate_ghost_predictions.py
verify_bug_fixes.py
verify_news_deployment.py
verify_postgres_migration.py
verify_production.py
verify_simulation.py
verify_stage1.py
```

---

## 📚 DOCUMENTATION - CONSOLIDATION NEEDED

**556 markdown files is TOO MANY.** 

### Proposed Consolidation:
```
docs/
├── README.md                    # Main readme
├── QUICKSTART.md               # Getting started
├── ARCHITECTURE.md             # System design
├── API_REFERENCE.md            # All endpoints
├── DEPLOYMENT.md               # Railway/Docker
├── TROUBLESHOOTING.md          # Common issues
├── CHANGELOG.md                # Version history
└── archive/                    # Old docs (don't delete)
```

---

## 🔌 API ENDPOINTS - COMPLETE MAP

### Health & Status
```
GET  /                           → Root redirect
GET  /health                     → Health check
GET  /api/status                 → System status
GET  /metrics                    → Prometheus metrics
```

### Stock Predictions
```
GET  /api/v3/stock/predict/{symbol}?bypass_calendar=true  → Single stock
GET  /api/v3/stock/batch?symbols=NVDA,AMD                 → Batch stocks
GET  /api/v3/stock/debug/{symbol}                         → Debug data
```

### Crypto Predictions
```
GET  /api/crypto/predict/{symbol}              → Single crypto
GET  /api/v3/opus/predict/{symbol}             → Full prediction
GET  /api/v3/predictions/latest                → Cached predictions
```

### V2 Quality System
```
GET  /api/v2/quality/status                    → Current config
POST /api/v2/quality/reload                    → Reload from JSON
GET  /api/v2/quality/test-should-predict?symbol=X&confidence=75
```

### Watchlist
```
GET  /api/v3/watchlist                         → Get watchlist
POST /api/v3/watchlist/add?symbol=X            → Add symbol
POST /api/v3/watchlist/remove?symbol=X         → Remove symbol
POST /api/v3/watchlist/trigger-prediction      → Trigger scan
```

### Alerts
```
POST /alerts/test?send=true                    → Test alert
POST /alerts/predictions/send                  → Full prediction alert
POST /alerts/top10/force                       → Force TOP 10
GET  /api/alerts/history                       → Alert history
```

### Paper Trading
```
GET  /api/v3/paper/trades?limit=10             → Recent trades
GET  /api/v3/paper/summary                     → Trading summary
POST /api/v3/paper/trade                       → Execute paper trade
```

### Portfolio
```
GET  /api/v3/portfolio                         → Current portfolio
GET  /api/v3/portfolio/performance             → Performance metrics
```

### Accuracy
```
GET  /api/accuracy/dashboard                   → Accuracy dashboard
GET  /api/accuracy/by-symbol/{symbol}          → Per-symbol accuracy
GET  /api/v2/accuracy/summary                  → V2 accuracy
```

---

## ⚙️ CONFIGURATION FILES

| File | Purpose | Edit Frequency |
|------|---------|----------------|
| `ghost_v2_quality.json` | V2 whitelist/blacklist | Often |
| `.env` | Environment secrets | Rarely |
| `requirements.txt` | Python dependencies | Sometimes |
| `Dockerfile` | Container build | Rarely |
| `railway.json` | Railway config | Rarely |
| `pyproject.toml` | Project config | Rarely |

---

## 🚀 QUICK REFERENCE COMMANDS

### Check System
```bash
# Syntax check
python3 -m py_compile wolf_app.py

# Health check
curl https://ghost-protocol-production.up.railway.app/health

# V2 status
curl https://ghost-protocol-production.up.railway.app/api/v2/quality/status
```

### Test Predictions
```bash
# Stock prediction
curl "https://ghost-protocol-production.up.railway.app/api/v3/stock/predict/NVDA?bypass_calendar=true"

# Send Telegram alert
curl -X POST "https://ghost-protocol-production.up.railway.app/alerts/predictions/send"
```

### Deploy
```bash
git add -A && git commit -m "message" && git push
```

---

## 🧹 RECOMMENDED CLEANUP ACTIONS

### Phase 1: Organize Root
1. Move 50+ `test_*.py` files → `tests/`
2. Move diagnostic scripts → `tools/`
3. Move training scripts → `scripts/`
4. Archive dead code → `.archive/`

### Phase 2: Consolidate Core
1. Merge duplicate accuracy modules
2. Merge duplicate portfolio modules
3. Merge duplicate risk modules
4. Document which modules are active

### Phase 3: Consolidate Docs
1. Create 6 canonical docs
2. Move 550+ files → `docs/archive/`
3. Update README with links

### Phase 4: Code Quality
1. Add type hints throughout
2. Add docstrings to all functions
3. Create API documentation
4. Add unit tests

---

## ✅ VERIFICATION CHECKLIST

Before deploying:
- [ ] `python3 -m py_compile wolf_app.py` passes
- [ ] `/health` returns 200
- [ ] Stock prediction works
- [ ] Telegram alert sends
- [ ] V2 quality status correct

---

**This is your single source of truth. When lost, start here.**
