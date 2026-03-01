# 🔍 GHOST PROTOCOL — PIPELINE WIRING AUDIT
## Date: March 1, 2026

### Pipeline Architecture (as-built)
```
GhostScout._make_prediction()
  → _get_prediction_from_engine()
    → MultiCryptoPredictor (MISSING FILE — ImportError caught, falls through)
    → _technical_prediction() (SMA + RSI — THIS IS THE ACTUAL ENGINE)
  → get_news_sentiment_for_symbol() (Alpha Vantage, optional)
  → _calculate_hold_period()

GhostScout.scout_all() / scout_all_fast()
  → Phase 1: _make_prediction() for all symbols
  → Phase 2: _apply_brain_analysis() → GhostBrain.analyze_batch()
  → Phase 3: Record trades via money_game_engine

GhostNotifications._maybe_send_top10()
  → V3Filter.filter_and_score()  (learning loop synced)
  → regime_filter.apply_regime_filter()  (crypto BUY gating)
  → format + send Telegram
```

---

## AUDIT RESULTS TABLE

| # | Module | File Path | EXISTS | IMPORTED by Pipeline | ACTIVELY CALLED during Prediction | What it SHOULD Do | STATUS |
|---|--------|-----------|--------|---------------------|----------------------------------|-------------------|--------|
| 1 | **News Brain** | `core/intelligence/ghost_news_brain.py` | ✅ YES | ❌ NO (only wolf_app) | ❌ NO — runs in separate background loop | Analyze breaking news with Claude, flag predictions_at_risk, auto-pause trading on critical events | **DISCONNECTED** — runs silently in background. `predictions_at_risk` data is logged but **never feeds back** to scout or brain. `_set_trading_paused()` writes to DB but **nothing reads** the pause flag before making predictions. |
| 2 | **News Sentiment** | `core/news_sentiment.py` | ✅ YES | ✅ YES (ghost_scout.py L112) | ✅ YES — called via `get_news_sentiment_for_symbol()` | Fetch Alpha Vantage news and compute sentiment score | **WIRED** — but returns empty articles[] and score=0 when `ALPHA_VANTAGE_API_KEY` is unset. Fallback is `articles = []` → sentiment=0.0 (neutral). Effectively a no-op without the API key. |
| 3a | **ML Model / XGBoost** | `core/multi_crypto_predictor.py` | ❌ **MISSING** | ✅ YES (ghost_scout.py L896) | ❌ NO — `ImportError` is caught, falls through to `_technical_prediction()` | Predict crypto direction using trained ML model | **DEAD** — File does not exist. The import fails silently. **100% of predictions come from the SMA+RSI fallback.** |
| 3b | **ML Trainer** | `core/ml_trainer.py` | ✅ YES | ❌ NO (only wolf_app L4527, L13102) | ❌ NO — only exposed via API endpoints | Train XGBoost model on historical data | **DISCONNECTED** — trains models, but the consumer (`multi_crypto_predictor.py`) doesn't exist. Training output goes nowhere usable by the prediction pipeline. |
| 3c | **Model Store** | `core/model_store.py` | ✅ YES | ❌ NO (only wolf_app startup) | ❌ NO — loads model at startup for filesystem caching | PostgreSQL model persistence for trained models | **DISCONNECTED** — stores/loads XGBoost pkl files, but no pipeline component reads them during prediction. |
| 3d | **Trained Models** | `models/trained/ghost_xgboost_v*.pkl` | ✅ YES (3 files) | N/A | ❌ NO | Serialized XGBoost models | **DEAD** — sitting on disk unused because `multi_crypto_predictor.py` doesn't exist. |
| 4a | **Opus Brain (Claude)** | `core/intelligence/opus_brain.py` | ✅ YES | ❌ NO (only wolf_app L10713+) | ❌ NO — only exposed via API endpoints (`/api/opus/*`) | Use Claude/Sonnet to analyze symbols with reasoning | **DISCONNECTED** — has `analyze()`, `research()`, `explain()`, `compare()` functions. Exposed via wolf_app API for manual use only. Never called during prediction cycle. |
| 4b | **Ghost Advisor** | `core/ghost_advisor.py` | ✅ YES | ❌ NO (only wolf_app L29001+) | ❌ NO — API endpoint only | Track positions, send SL/TP alerts | **DISCONNECTED** — fully built trade management system (entry/exit alerts, trailing stops). Only accessible via API, not wired into notification system or prediction pipeline. |
| 4c | **AI Advisor dir** | `core/ai_advisor/` | ✅ YES | ❌ NO | ❌ NO | Contains `accuracy_tracker.py` + `scanner.py` | **DEAD** — appears to be an older/alternate implementation. Not imported anywhere. |
| 4d | **Ghost Researcher** | `core/ghost_researcher.py` | ✅ YES | ❌ NO (not imported anywhere in pipeline) | ❌ NO | Deep AI research on symbols (company info, catalysts, sentiment) | **DISCONNECTED** — has `ResearchReport` dataclass + Claude integration. Not imported by scout, brain, or any pipeline file. |
| 5 | **Social Sentiment** | `core/social_sentiment.py` | ✅ YES | ❌ NO (only wolf_app L17441) | ❌ NO — API endpoint only | Twitter/Reddit sentiment monitoring | **DISCONNECTED** — requires `TWITTER_BEARER_TOKEN`, returns error when unset. Only accessible via `/api/social_sentiment/{symbol}` endpoint. Never influences predictions. |
| 6 | **Event Detector** | `core/event_detector.py` | ✅ YES | ❌ NO (not in any pipeline file) | ❌ NO | Detect market-moving events from multiple sources (earnings, news, on-chain) | **DEAD** — imports `event_memory` and `pattern_tracker`, but is never imported by scout, brain, notifications, or wolf_app. Completely orphaned. |
| 7a | **World Context** | `core/world_context.py` | ✅ YES | ❌ NO (only wolf_app L10209+) | ❌ NO — feeds into wolf_app's Stage 1 context, not into scout | Aggregate VIX, SPY, market mood | **DISCONNECTED** — used by wolf_app's Stage 1 + research snapshot for WOLF-specific predictions. Never imported by ghost_scout or ghost_brain. Brain gets regime/F&G data separately via `_gather_market_data()`. |
| 7b | **World Feed Fusion** | `core/world_feed_fusion.py` | ✅ YES | ❌ NO (only wolf_app L38234+) | ❌ NO — RSS NLP analysis, API endpoints only | Aggregate financial RSS feeds + NLP sentiment analysis | **DISCONNECTED** — exposed via wolf_app API endpoints. Not used in prediction pipeline. |
| 8 | **Learning Loop** | `core/learning_loop.py` | ✅ YES | ✅ YES (v3_filter.py L152) | ⚠️ PARTIAL — reads config for bias_correction + confidence_threshold | Auto-tune model parameters based on accuracy feedback | **PARTIALLY WIRED** — `v3_filter._sync_learning_loop()` reads `bias_correction` and `confidence_threshold` from learning_loop every 5 min. However, the learning_loop itself runs as a background hourly task. The actual parameter adjustments (from `adjust_parameters()`) write to memory.json/PostgreSQL and v3_filter reads them. **This is the only intelligence feedback loop that's actually wired.** |
| 9a | **Market Regime** | `core/market_regime.py` | ✅ YES | ❌ NO (not in pipeline) | ❌ NO | Background loop that periodically records market regime | **DISCONNECTED** — wraps `regime_detector` in a background loop. Not imported by any pipeline file. |
| 9b | **Regime Detector** | `core/regime_detector.py` | ✅ YES | ❌ NO (wolf_app L179 only) | ❌ NO — initialized at startup for Stage 3 API | HMM-based market regime classification (bull/bear/sideways/volatile) | **DISCONNECTED** — wolf_app imports and initializes it. Exposes via API endpoints. Ghost Brain has its own inline regime logic (from Fear & Greed via `_gather_market_data()`). Never called during scout's prediction cycle. |
| 9c | **Regime Filter** | `core/regime_filter.py` | ✅ YES | ✅ YES (ghost_notifications.py L2354) | ✅ YES — gates crypto BUYs when BTC is dumping | Gate crypto BUY signals during market dumps | **WIRED** — applied in `ghost_notifications._maybe_send_top10()` between V3 filter output and Telegram message formatting. Suppresses crypto BUYs when BTC down >2-5%. **This is working as designed.** |
| 10 | **Confidence Calibrator** | `core/confidence_calibrator.py` | ✅ YES | ❌ NO (only wolf_app L8827, L9172+) | ⚠️ PARTIAL — used in wolf_app's `/api/predict/run` stock engine, not in scout | Map predicted confidence → actual accuracy using historical data | **DISCONNECTED from main pipeline** — wired into wolf_app's stock engine (`calibrate_confidence_with_signals()` at L8827). Runs auto-build in background. **NOT used by ghost_scout's `_technical_prediction()` or brain.** |
| 11 | **Prediction Killswitch** | `core/prediction_killswitch.py` | ✅ YES | ❌ NO (wolf_app L4690 only) | ❌ NO — status logged at startup, `can_send_prediction()` never called before predictions | Emergency stop for all predictions | **BROKEN** — `get_killswitch()` is called at startup to log status. But `can_send_prediction()` is **never called** before scout runs predictions or before notifications are sent. Setting `PREDICTIONS_ENABLED=false` has **no effect** on the actual prediction pipeline. |
| 12a | **Dynamic Exits** | `core/dynamic_exits.py` | ✅ YES | ❌ NO (wolf_app L41243 only) | ❌ NO — API endpoint only | Calculate dynamic exit levels (trailing stops, SL/TP) | **DISCONNECTED** — exposed via API. Not integrated into position monitoring or paper trade tracking. |
| 12b | **SL/TP Monitor** | `core/sl_tp_monitor.py` | ✅ YES | ❌ NO — **COMMENTED OUT** in wolf_app (L5438-5442) | ❌ NO | Background loop checking SL/TP triggers on positions | **DEAD** — the startup code to launch this background task is **commented out** in wolf_app (lines 5438-5442). The system was built but explicitly disabled. |
| 13 | **Guardian Oracle** | `core/guardian_oracle.py` | ✅ YES | ❌ NO (wolf_app L6046+ only) | ❌ NO — personality/notification system, not prediction | Morning prophecies + 24/7 guardian alerts + reality checks | **DISCONNECTED** — a notification personality system. Initialized at startup but doesn't feed into or modify predictions. Runs independently. |
| 14 | **Self-Improvement Engine** | `core/self_improvement_engine.py` | ✅ YES | ❌ NO (wolf_app L5535+ only) | ❌ NO — runs hourly in background | Dynamic threshold tuning, universe expansion, confidence calibration | **DISCONNECTED** — runs as an hourly background task. Writes to `self_improvement_memory.json` but no pipeline component reads its output. Its threshold tuning and universe expansion don't flow back into scout or brain. |
| 15a | **Ensemble Predictor** | `core/ensemble_predictor.py` | ✅ YES | ❌ NO (wolf_app L8644+ only) | ❌ NO — used in wolf_app's stock engine, not scout | Multi-model ensemble (LSTM + XGBoost + Transformer) | **DISCONNECTED from main pipeline** — used in wolf_app's `/api/predict/run` stock prediction endpoint for directional prediction. **NOT used by ghost_scout.** Scout uses its own `_technical_prediction()`. Two completely separate prediction systems. |
| 15b | **Ensemble Forecaster** | `core/ensemble_forecaster.py` | ✅ YES | ❌ NO (wolf_app L178+ only) | ❌ NO — Stage 3 API only | Multi-model ensemble with dynamic weighting | **DISCONNECTED** — initialized at Stage 3 startup. Exposed via API endpoints (`/api/stage3/ensemble/*`). Not called during scout predictions. |
| 15c | **Multi-Horizon Forecaster** | `core/multi_horizon_forecaster.py` | ✅ YES | ❌ NO (wolf_app L37657+ only) | ❌ NO — API endpoint only | 3 concurrent forecasts (1h, 48h, 1wk) | **DISCONNECTED** — exposed via `/api/forecast/multi_horizon`. Not integrated into prediction pipeline. |
| 16 | **Pattern Intelligence** | `core/pattern_intelligence/` | ✅ YES (8 modules) | ⚠️ INDIRECT (wolf_app L8798 via `pattern_enhanced_predictor`) | ⚠️ PARTIAL — used in wolf_app stock engine only | Fear/Greed, funding rates, social sentiment, BTC correlation, pattern matching | **DISCONNECTED from main pipeline** — accessed via `pattern_enhanced_predictor.py` in wolf_app's stock engine. **NOT used by ghost_scout.** Scout doesn't use fear/greed, funding rates, or pattern matching for its SMA+RSI predictions. |
| 17 | **Quality Gate** | `core/quality_gate.py` | ✅ YES | ❌ NO (wolf_app L14439 only) | ❌ NO — API status endpoint only | 85% min accuracy, 10 max/day, dedup, min return | **DISCONNECTED** — exposes status via API. `can_send_prediction()` is **never called** before predictions are sent. The quality gate exists but has no enforcement point in the pipeline. |
| 18 | **Trust Ladder** | `core/trust_ladder.py` | ✅ YES | ❌ NO (wolf_app L9461+ only) | ❌ NO — API endpoint lookup only | Progressive accuracy: promote symbols through trust levels (48h→120h→168h holds) | **DISCONNECTED** — wolf_app reads trust data for API display. Not used by v3_filter, ghost_notifications, or scout to modify predictions or confidence. |
| 19 | **Santiment Signals** | `core/santiment_signals.py` | ✅ YES | ❌ NO (wolf_app L41162+ only) | ❌ NO — API endpoint only | On-chain data (social volume, whale activity) via Santiment API | **DISCONNECTED** — requires `SANTIMENT_API_KEY`. Exposed via API endpoints. Never called during prediction cycle. |
| 20 | **VWAP Signals** | `core/vwap_signals.py` | ✅ YES | ❌ NO (wolf_app L41201 only) | ❌ NO — API endpoint only | VWAP analysis (Volume Weighted Average Price) | **DISCONNECTED** — exposed via API. Not used in `_technical_prediction()`. Scout's technical analysis uses SMA + RSI only. |

---

## SUMMARY

### ✅ ACTUALLY WIRED INTO PREDICTION PIPELINE (3 of 20)
1. **News Sentiment** — called by scout, but returns 0 without Alpha Vantage key
2. **Learning Loop** → V3 Filter — bias_correction + confidence_threshold synced
3. **Regime Filter** — gates crypto BUYs in ghost_notifications

### ⚠️ PARTIALLY WIRED (2 of 20)
4. **Confidence Calibrator** — used in wolf_app stock engine, NOT in scout
5. **Pattern Intelligence** — used in wolf_app stock engine, NOT in scout

### ❌ COMPLETELY DISCONNECTED (15 of 20)
6. News Brain (runs silently, output unused)
7. ML Trainer (trains models that nothing loads)
8. Model Store (stores models nothing uses)
9. Opus Brain (API-only)
10. Ghost Advisor (API-only)
11. Ghost Researcher (never imported)
12. Social Sentiment (API-only)
13. Event Detector (orphaned — not imported anywhere)
14. World Context (wolf_app only, not scout)
15. World Feed Fusion (API-only)
16. Market Regime / Regime Detector (API-only)
17. Guardian Oracle (notification personality, not prediction)
18. Self-Improvement Engine (writes to JSON, nothing reads)
19. Ensemble Predictor / Forecaster (wolf_app stock engine only)
20. Multi-Horizon Forecaster (API-only)

### 💀 DEAD / BROKEN (4 of 20)
21. **multi_crypto_predictor.py** — FILE DOESN'T EXIST, ImportError caught silently
22. **SL/TP Monitor** — startup code COMMENTED OUT
23. **Prediction Killswitch** — exists but `can_send_prediction()` never called
24. **Quality Gate** — exists but enforcement never called

---

## CRITICAL FINDING: THE ACTUAL PREDICTION ENGINE

**Ghost's entire prediction pipeline runs on ONE function:** `GhostScout._technical_prediction()` (lines 941-1090 of ghost_scout.py).

This function uses:
- 5-day and 10-day SMA (Simple Moving Average)
- 10-day and 20-day SMA
- 14-period RSI
- 20-day price range position

**Everything else** — the ML models, ensemble predictors, pattern intelligence, VWAP, social sentiment, Santiment, event detection, Claude/Opus analysis, multi-horizon forecasts — **exists as code but is not connected to the prediction pipeline.**

The only post-processing that works:
1. **GhostBrain.analyze_batch()** — adjusts confidence/direction based on historical accuracy
2. **V3Filter** — filters by validated strategies + learning loop adjustments  
3. **Regime Filter** — gates crypto BUYs during BTC dumps

### TWO SEPARATE PREDICTION SYSTEMS
There are actually TWO prediction systems that don't talk to each other:
1. **GhostScout** (main pipeline) → SMA+RSI → Brain → V3 → Notifications → Telegram
2. **wolf_app stock engine** (`/api/predict/run`) → XGBoost ensemble + confidence calibrator + pattern intelligence → API response only

System #2 is more sophisticated but feeds NO predictions into the Telegram notification pipeline.
