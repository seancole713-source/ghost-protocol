# 🧬 GHOST PROTOCOL — MASTER SYSTEM INDEX

**Last Updated:** March 1, 2026  
**Purpose:** Single source of truth for ALL systems, their status, and where to find them.  
**Rule:** This file gets updated with every structural change.

---

## 📊 SYSTEM HEALTH AT A GLANCE

| System | Status | Location | Lines |
|--------|--------|----------|-------|
| 🐺 Wolf Core | ✅ LIVE | `wolf_app.py` | ~45,268 |
| 🧠 Intelligence Hub | ✅ LIVE | `core/intelligence_hub.py` | ~950 |
| 📰 News Brain | ✅ LIVE | `core/intelligence/ghost_news_brain.py` | ~950 |
| 🔮 Ghost Scout | ✅ LIVE | `core/ghost_scout.py` | varies |
| 📈 Stock Engine | ✅ LIVE | `core/stock_engine.py` | varies |
| 🎯 Edge Whitelist | ✅ LIVE | `config/symbols.py` | ~50 |
| 🧪 Tests | ✅ 38/38 | `tests/test_intelligence_hub.py` | ~400 |

---

## 🧠 INTELLIGENCE HUB — CENTRAL NERVOUS SYSTEM

**File:** `core/intelligence_hub.py`  
**Created:** This session (commits d54e5de → 058d21f)  
**Purpose:** Aggregates ALL 20 intelligence systems into one decision pipeline.

### What It Does
1. Runs 13 signal checkers (news, ML, patterns, ensemble, etc.)
2. Applies 3 post-processors (trust ladder, quality gate, market regime)
3. Runs 3 safety gates (killswitch, drawdown, exposure)
4. Produces: direction adjustment (CONFIRM/FLIP/WEAKEN/BLOCK), confidence delta, trust boost
5. Self-improvement engine runs every 6 hours

### 20 Intelligence Systems — Wiring Status

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

## 📰 NEWS BRAIN — INTERNAL ONLY

**File:** `core/intelligence/ghost_news_brain.py`  
**Status:** ✅ Feeds Intelligence Hub. NO Telegram sends.

### Data Flow
```
RSS Feeds + CryptoPanic → Claude Analysis → Intelligence Hub Cache → Prediction Adjustments
     (14+ feeds)            (every 30 min)      (update_news_brain_cache)    (direction/confidence)
```

### What Ghost Does With News
- **CRITICAL event** → Auto-pauses trading 4 hours + guardian alerts
- **HIGH event** → Guardian alerts for affected symbols
- **Predictions at risk** → Hub adjusts direction (FLIP/WEAKEN) and confidence
- **NO Telegram** → Ghost uses news internally, doesn't spam the user

### Key Methods
| Method | Purpose | Telegram? |
|--------|---------|-----------|
| `analyze_news()` | Fetch + Claude analysis | ❌ No |
| `send_alert()` | Log internally | ❌ No (disabled) |
| `handle_critical_event()` | Guardian alerts + auto-pause | ❌ No (disabled) |
| `analyze_news_with_auto_pause()` | Full analysis + auto-actions | ❌ No |

---

## 🎯 EDGE WHITELIST — 13 PROVEN SYMBOLS

**File:** `config/symbols.py`  
**Source of truth:** Code default (ignores env var overrides)

```
STOCKS (7):  PANW, NET, FTNT, DDOG, T, BMBL, XPO
CRYPTO (6):  ETH, XRP, LINK, CHZ, BTC, SOL
```

---

## 📡 KEY API ENDPOINTS

### Predictions
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v3/predictions/latest` | GET | All predictions (default limit=25, edge-first sort) |
| `/api/v3/predictions/latest?symbol=ETH` | GET | Single symbol prediction |
| `/api/predict/run?symbol=ETH` | GET | Trigger prediction for symbol |

### Intelligence
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v3/intelligence/status` | GET | Hub system status + news cache age |
| `/api/v3/news/analyze` | GET | Trigger news analysis → feeds Hub |
| `/api/v3/news/analyze-with-auto-pause` | POST | Analysis + auto-pause on CRITICAL |
| `/api/v3/news/history` | GET | Past news analyses |
| `/api/v3/trading/pause-status` | GET | Check if trading is paused |

---

## 🔧 BUGS FIXED THIS SESSION (8 commits)

| Commit | Fix | Impact |
|--------|-----|--------|
| `d54e5de` | Intelligence Hub created | 20 systems → 1 pipeline |
| `ff5c883` | Wired into Stock + Turbo engines | Main engines use hub (not just scout) |
| `6ef668e` | Quality Gate advisory-only | No longer blocks predictions <65% |
| `cf405b3` | Trust ladder multiplier→delta | `1.20` → `+0.20` not `+1.20` |
| `274313c` | Confidence cap 0.85 not 0.92 | Real variance restored (0.10–0.78) |
| `c022932` | Dedup cache fix | Crypto survives PredictionRejected |
| `610e291` | Double trust ladder removed | Turbo path no longer double-applies |
| `058d21f` | API limit 10→25 + edge sort | All 13 edge symbols visible by default |

---

## 📁 CRITICAL FILE LOCATIONS

### Core Brain
| File | Purpose | Lines |
|------|---------|-------|
| `wolf_app.py` | Main FastAPI app — ALL endpoints | ~45,268 |
| `core/intelligence_hub.py` | Central nervous system | ~950 |
| `core/ghost_scout.py` | Money game prediction engine | varies |
| `core/stock_engine.py` | Stock prediction engine | varies |
| `core/intelligence/ghost_news_brain.py` | News analysis via Claude | ~950 |
| `core/intelligence/opus_brain.py` | Opus-level AI analysis | varies |

### Configuration
| File | Purpose |
|------|---------|
| `config/symbols.py` | Edge whitelist (13 symbols) |
| `config/settings.py` | App settings |
| `Procfile` | Railway deployment entry point |
| `requirements.txt` | Python dependencies |

### Data
| File | Purpose |
|------|---------|
| `wolf.db` | SQLite — predictions, outcomes, trades |
| `ghost_predictions.db` | Prediction store |
| `models/` | ML model weights (XGBoost, ensemble) |

### Tests
| File | Purpose | Count |
|------|---------|-------|
| `tests/test_intelligence_hub.py` | Hub unit tests | 38 tests |

---

## 🏗️ DEPLOYMENT

| Setting | Value |
|---------|-------|
| Platform | Railway |
| URL | `https://ghost-protocol-production.up.railway.app` |
| Auto-deploy | ✅ On push to `main` |
| Repository | `seancole713-source/ghost-protocol` |
| Branch | `main` |

### Required Env Vars (Railway)
| Var | Status | Purpose |
|-----|--------|---------|
| `ANTHROPIC_API_KEY` | ✅ Set | Claude API for News Brain |
| `OPENAI_API_KEY` | ✅ Set | GPT fallback |
| `TELEGRAM_BOT_TOKEN` | ✅ Set | Trade alerts (not news) |
| `TELEGRAM_CHAT_ID` | ✅ Set | Chat target |
| `EDGE_WHITELIST_ENABLED` | ✅ =1 | Restrict to 13 proven symbols |
| `ALPHA_VANTAGE_API_KEY` | ❌ Not set | Would enable more data |

---

## 📋 PREVIOUS INDEX FILES (for reference)

| File | Date | Purpose |
|------|------|---------|
| `GHOST_INDEX.md` | Jan 27, 2026 | Original OCD codebase map |
| `GHOST_CODEBASE_MAP.md` | Jan 28, 2026 | Navigation guide |
| `GHOST_BODY_MAP.md` | Jan 7, 2026 | Body analogy system map |
| `GHOST_BLUEPRINT.md` | Dec 11, 2025 | Architecture reference |
| `GHOST_BASELINE_MANIFEST.md` | Dec 12, 2025 | Baseline snapshot |
| `AUTOPSY_INDEX.md` | Dec 8, 2025 | Autopsy documentation index |

---

*This index supersedes all previous indexes. Updated with every structural change.*
