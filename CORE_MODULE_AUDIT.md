# 🔍 CORE MODULE AUDIT — Comprehensive Classification
**Date:** February 24, 2026  
**Scope:** All ~175 files in `/core/` + subdirectories  
**Method:** Static import analysis (`wolf_app.py` + cross-core imports)

---

## A. PREDICTION PIPELINE (actively used in `run_single_prediction`)

These modules are **directly called** inside `run_single_prediction()` (wolf_app.py:8077). They execute on every single prediction call.

| # | Module | Pipeline Step | What It Does | Import Count |
|---|--------|---------------|--------------|:---:|
| 1 | **providers/turbo_provider.py** | PRICE FETCH | Fast-fail price fetcher (3s budget). Routes to Polygon/AlphaVantage/Yahoo. Top-level import. | 16 |
| 2 | **crypto/crypto_providers.py** | PRICE FETCH | Crypto price quorum (CoinGecko, Binance, Coinbase). Used when `is_crypto=True`. | 39 |
| 3 | **stock_engine.py** | STOCK ROUTING | Full stock prediction engine (24h horizon, 2% target). Stocks route here FIRST (line ~8143). Falls back to turbo if it fails. | 6 |
| 4 | **stock_gates.py** | STOCK ROUTING | Sub-module of stock_engine: economic calendar gate, sector momentum gate, multi-timeframe gate. | 2 |
| 5 | **data_pillars/feature_orchestrator.py** | FEATURE EXTRACTION | Orchestrates all 6 data pillars to extract ~50 features (RSI, MACD, volume, sentiment, etc). Line ~8363. | 6 |
| 6 | **data_pillars/technical_engine.py** | FEATURE EXTRACTION | Technical indicators sub-pillar (RSI, MACD, Bollinger, SMA). | 2 |
| 7 | **data_pillars/volume_engine.py** | FEATURE EXTRACTION | Volume analysis sub-pillar (volume spike, OBV, volume trend). | 3 |
| 8 | **data_pillars/sentiment_engine.py** | FEATURE EXTRACTION | News/social sentiment sub-pillar. | 2 |
| 9 | **data_pillars/price_engine.py** | FEATURE EXTRACTION | Price history sub-pillar. | 1 |
| 10 | **data_pillars/flow_engine.py** | FEATURE EXTRACTION | Order flow sub-pillar. | 1 |
| 11 | **data_pillars/world_context_engine.py** | FEATURE EXTRACTION | Macro/world context sub-pillar. | 1 |
| 12 | **data_pillars/base_pillar.py** | FEATURE EXTRACTION | Abstract base class for all pillars. | 7 |
| 13 | **feature_diagnostics.py** | FEATURE DIAGNOSIS | `diagnose_features()` — evaluates feature quality/staleness. Top-level import (line 108). | 2 |
| 14 | **stage1_integration.py** | CONTEXT INJECTION | Injects `world_context` + `market_mood` into features. Gated by `STAGE1_ENABLED`. Line ~8429. | 6 |
| 15 | **context_engine.py** | CONTEXT INJECTION | `WorldContextEngine` — VIX, SPY, macro data. Called via `stage1_integration`. | 8 |
| 16 | **market_mood.py** | CONTEXT INJECTION | Market sentiment aggregation. Called via `stage1_integration`. | 3 |
| 17 | **ensemble_predictor.py** | DIRECTION + CONFIDENCE | Combines LSTM + XGBoost + Transformer votes. Line ~8532. Critical for direction. | 13 |
| 18 | **directional_accuracy_tracker.py** | CONFIDENCE ADJUST | Adaptive UP/DOWN penalty based on live paper trade win rates. Line ~8561. | 2 |
| 19 | **pattern_enhanced_predictor.py** | CONFIDENCE BOOST | Fear/greed, funding rates, social sentiment, BTC correlation. Gated by `ENABLE_PATTERN_INTELLIGENCE`. Line ~8592. | 1 |
| 20 | **confidence_calibrator.py** | CONFIDENCE CALIBRATION | Signal-based confidence calibration system. Line ~8620 + ~8954. Core confidence pipeline. | 5 |
| 21 | **asset_performance_filter.py** | QUALITY GATE | Historical win rate adjustment. Can blacklist symbols. Line ~8756. | 1 |
| 22 | **market_gates.py** | QUALITY GATE | Regime filter + VIX gate + confirmation counter. Line ~8793. Async. | 9 |
| 23 | **asset_classification.py** | CLASSIFICATION | `is_crypto_symbol()`, `register_crypto_symbols()`. Used throughout. | 10 |
| 24 | **asset_classifier.py** | TARGET SIZING | `get_target_stop()`, `get_asset_type()`. Determines target/stop per asset type. Line ~9160. | 26 |
| 25 | **trust_ladder.py** | CONFIDENCE BOOST | Per-symbol trust levels (proven winners get 10-20% boost). Line ~9243. Gated by `TRUST_LADDER_ENABLED`. | 7 |
| 26 | **momentum_tracker.py** | MOMENTUM CALC | Calculates confidence trend (HOT/WARMING/STABLE/COOLING/COLD). Line ~9304. | 6 |
| 27 | **prediction_store.py** | STORAGE | `create_prediction()` + `PredictionRejected`. Writes to prediction store. Line ~9333. | 23 |
| 28 | **touch_calibration_sqlite.py** | STORAGE | Touch-target calibration + stage5/stage6 gating. Line ~9472. | 1 |
| 29 | **accuracy_tracker.py** | TRACKING | `record_forecast()` — registers prediction for 48h accuracy evaluation. Line ~9509. | 6 |
| 30 | **feedback_loop.py** | TRACKING | `get_adjusted_features()` — applies learned feature weights. Line ~9510. | 7 |
| 31 | **prediction_evaluator.py** | TRACKING | `_ensure_touch_columns()` — schema migration for touch columns. Line ~9553. | 3 |
| 32 | **paper_tracker.py** | PAPER TRADING | `log_signal()` — auto-logs paper trade for P&L tracking. Line ~9639. | 19 |
| 33 | **ghost_notifications.py** | PAPER TRADING | `V3_VALIDATED_STRATEGIES` dict — used for V3 strategy tagging. Line ~9640. | 24 |
| 34 | **position_sizer.py** | PAPER TRADING | Kelly Criterion + ATR position sizing. Line ~9714. Gated by `POSITION_SIZER_ENABLED`. | 4 |
| 35 | **trading_controls.py** | PAPER TRADING | Quality gate: checks symbol's recent win rate before logging trade. | 3 |
| 36 | **prediction_tracker.py** | STORAGE | Ghost predictions SQLite table schema. Imported as `_pt`. | 5 |
| 37 | **concurrency.py** | INFRASTRUCTURE | `AsyncRateLimiter`. Top-level import (line 73). Used for rate limiting. | 4 |
| 38 | **price_quorum.py** | INFRASTRUCTURE | `PriceDecision`, `PriceProvider`, `get_price_quorum`. Top-level import (line 74). | 6 |
| 39 | **models.py** | INFRASTRUCTURE | `Prediction`, `Direction`, `ScoredPrediction` dataclasses. Used by adapters/filters. | 3 |

### Pipeline Flow Summary
```
run_single_prediction(symbol)
├── 1. CLASSIFY: asset_classification → is_crypto?
├── 2. STOCK ROUTING: stock_engine (if stock + USE_STOCK_ENGINE=true)
│   ├── stock_gates (economic_calendar, sector_momentum, multi_timeframe)
│   ├── ensemble_predictor + feature_orchestrator
│   └── Returns early with stock result
├── 3. TURBO PRICE: turbo_provider (stock) or crypto_providers (crypto)
├── 4. FEATURES: feature_orchestrator.get_all_features()
│   ├── technical_engine (RSI, MACD, BB)
│   ├── volume_engine (volume spike, OBV)
│   ├── sentiment_engine (news sentiment)
│   ├── price_engine (price history)
│   ├── flow_engine (order flow)
│   └── world_context_engine (macro)
├── 5. DIAGNOSIS: feature_diagnostics.diagnose_features()
├── 6. CONTEXT: stage1_integration → context_engine + market_mood
├── 7. DIRECTION: RSI/MACD → ensemble_predictor.predict()
├── 8. CONFIDENCE ADJUST:
│   ├── directional_accuracy_tracker (adaptive penalty/bonus)
│   ├── pattern_enhanced_predictor (fear/greed alignment)
│   ├── confidence_calibrator (signal-based calibration)
│   ├── Multi-horizon consensus (RSI + MACD + momentum)
│   ├── asset_performance_filter (historical win rate)
│   ├── market_gates (regime + VIX + confirmation counter)
│   ├── ghost_intel (institutional intelligence) [external]
│   └── trust_ladder (proven symbol boost)
├── 9. STORE: prediction_store.create_prediction()
│   ├── touch_calibration_sqlite
│   ├── accuracy_tracker.record_forecast()
│   ├── feedback_loop.get_adjusted_features()
│   └── prediction_tracker (SQLite write)
└── 10. PAPER TRADE: paper_tracker.log_signal()
    ├── ghost_notifications (V3 strategies)
    ├── trading_controls (quality gate)
    └── position_sizer (Kelly sizing)
```

---

## B. BACKGROUND WORKERS (schedulers, evaluators, loops)

These modules run as `asyncio.create_task()` background loops started in `post_startup()`.

| # | Module | Trigger | What It Does | Env Gate |
|---|--------|---------|--------------|----------|
| 1 | **auto_prediction_loop.py** | 5-min interval | Main prediction scheduler — runs `run_single_prediction()` for all symbols every 5 min. | Always on |
| 2 | **beast_scheduler.py** | Symbol lists | Provides `HUNTER_STOCK_SYMBOLS` + `HUNTER_CRYPTO_SYMBOLS` lists. | Always on |
| 3 | **smart_scout.py** | 6AM/8AM/6PM CT | Money Game scout: full_scout(), get_elite_predictions(), run_daily_cycle(). | `MONEY_GAME_ENABLED=1` |
| 4 | **ghost_scout.py** | Via smart_scout | Symbol lists (`ALL_STOCKS`, `ALL_CRYPTO`) + resolve_trades(). | `MONEY_GAME_ENABLED=1` |
| 5 | **money_game_engine.py** | Via smart_scout | Ranking engine for Money Game: score symbols, track wins/losses. | `MONEY_GAME_ENABLED=1` |
| 6 | **vip_scanner.py** | 60s interval | Scans microcap VIP coins (WEPE, LILPEPE, etc). Cash-App alerts. | Always on |
| 7 | **premarket_predictor.py** | 7AM CT weekdays | Pre-market stock predictions. | Always on |
| 8 | **full_market_scanner.py** | 5AM CT + hourly | Full market scan + hourly mover detection. | Always on |
| 9 | **self_improvement_engine.py** | Hourly | Autonomous improvement cycles. | Always on |
| 10 | **autonomous_execution_engine.py** | 5-min interval | Phase 5: Executes trades via Alpaca broker. | `AUTO_EXECUTION_ENABLED=1` (default OFF) |
| 11 | **telegram_hunter.py** | Daily | Daily report loop + Telegram alerts. | Always on |
| 12 | **market_scanner.py** | Via telegram_hunter | `scan_all()` for top opportunities. | Always on |
| 13 | **prediction_price_recorder.py** | Continuous | Records live prices for pending predictions. | Always on |
| 14 | **ghost_notifications.py** | Continuous | Signal dispatch: formats + sends Telegram alerts for TOP 10. | `GHOST_SIGNAL_DISPATCH_ENABLED=1` |
| 15 | **daily_predictions_engine.py** | 6AM daily | Morning briefing task. Feeds `run_single_prediction_async`. | Always on |
| 16 | **performance_dashboard.py** | Continuous | Dashboard monitoring loop. | Always on |
| 17 | **cascade_scheduler.py** | Interval | Cascade prediction scheduling. | Always on |
| 18 | **cascading_predictor.py** | Via cascade_scheduler | Multi-timeframe cascading predictions. | Always on |
| 19 | **cron_scheduler.py** | Cron-based | Coordinates DailyTop10Scanner + GuardianOracle. | Always on |
| 20 | **guardian_oracle.py** | Via cron_scheduler | System health monitoring + alerts. | Always on |
| 21 | **daily_top_10_scanner.py** | Via cron_scheduler | Scans and ranks top 10 daily picks. | Always on |
| 22 | **realtime_market_movers.py** | Continuous | Detects real-time price movers. | Always on |
| 23 | **prediction_evaluator.py** | Startup eval | `evaluate_pending_predictions()` — checks outcomes of past predictions. | Always on |
| 24 | **feedback_loop.py** | Startup scheduler | `start_learning_scheduler()` — periodic learning cycles. | Always on |
| 25 | **v2_quality.py** | Startup init | Quality system initialization. | Always on |
| 26 | **auto_calibrate_scheduler.py** | Weekly | Weekly calibration scheduler. | Always on |
| 27 | **ml_trainer.py** | Periodic | ML model retrain scheduler. | Always on |
| 28 | **online_calibrator.py** | Startup init | Online confidence calibration. | Always on |
| 29 | **prediction_killswitch.py** | Startup check | Kill switch check before predictions. | Always on |
| 30 | **watchlist_prediction_scheduler.py** | Interval | Predictions for personal watchlist symbols. | `WATCHLIST_ENABLED` |
| 31 | **orchestrator.py** | Master control | `start_all_background_services()` — starts all workers when `ORCHESTRATOR_ENABLED=1`. | `ORCHESTRATOR_ENABLED=1` (default OFF) |
| 32 | **workers/** (5 files) | Via orchestrator | liquidity_monitor, macro_brain_worker, pattern_memory, reflex_trainer, workers_utils. | `ORCHESTRATOR_ENABLED=1` |
| 33 | **ai_advisor/scanner.py** | Startup | AI advisor scanner. | `AI_ADVISOR_ENABLED=1` (default OFF) |
| 34 | **intelligence/ghost_news_brain.py** | Startup | News analysis brain. | `NEWS_ANALYSIS_ENABLED=1` |

---

## C. API-ONLY (only accessible via HTTP endpoints, not in prediction loop)

These modules are only imported inside specific API route handlers.

| # | Module | API Endpoint(s) | What It Does |
|---|--------|-----------------|--------------|
| 1 | **accuracy_dashboard_v2.py** | `/api/accuracy/v2`, `/api/accuracy/dashboard/v2` | V2 accuracy dashboard data. |
| 2 | **accuracy_tracking.py** | `/api/accuracy/trending`, `/api/accuracy/confidence`, `/api/accuracy/alerts` | Accuracy trend + correlation analytics. |
| 3 | **live_accuracy.py** | `/api/accuracy/live` | Live accuracy dashboard by symbol. |
| 4 | **touch_accuracy_metrics.py** | `/api/accuracy/touch` | Touch-target accuracy summary. |
| 5 | **prediction_reconciliation.py** | `/api/accuracy/reconcile` | Manual reconciliation endpoint. |
| 6 | **prediction_calibration.py** | `/api/calibration/report` | Calibration diagnostics report. |
| 7 | **auto_calibrate.py** | `/api/calibration/run` | Manual calibration trigger. |
| 8 | **historical_simulator.py** | `/api/simulate` | Historical simulation engine. |
| 9 | **simulation_queue.py** | `/api/simulate/queue`, `/api/simulate/status` | Async simulation task queue. |
| 10 | **backtester.py** | `/api/backtest` | Backtesting engine. |
| 11 | **data_enhanced_predictor.py** | `/api/data-prediction` | Data-enhanced prediction API. |
| 12 | **data_collector.py** | `/api/data-prediction` | Data collection for enhanced predictor. |
| 13 | **telegram_alerts.py** | `/api/telegram/*` | Telegram alert management APIs. |
| 14 | **quality_gate.py** | `/api/quality-gate` | Quality gate status API. |
| 15 | **ab_testing.py** | `/api/ab-test` | A/B testing runner. |
| 16 | **feature_analyzer.py** | `/api/features/analyze` | Feature importance analysis. |
| 17 | **strategy_tester.py** | `/api/strategy-test` | Strategy backtesting. Stage 4. |
| 18 | **hedging_engine.py** | `/api/hedge` | Hedging suggestions. Stage 4. |
| 19 | **portfolio_manager.py** | `/api/portfolio/optimize` | Portfolio optimization. Stage 4. |
| 20 | **regime_detector.py** | `/api/regime` | Market regime detection. Stage 3. |
| 21 | **risk_engine.py** | `/api/risk` | Risk metrics. Stage 3. |
| 22 | **ensemble_forecaster.py** | `/api/forecast/ensemble` | Multi-model forecast API. Stage 3. |
| 23 | **learning_loop.py** | `/api/learning/*` | Learning stats + manual cycle trigger. |
| 24 | **execution_analytics.py** | Stage 5 APIs | Execution quality analytics. |
| 25 | **execution_risk.py** | Stage 5 APIs | Execution risk limits. |
| 26 | **order_manager.py** | Stage 5 APIs | Order placement/management. |
| 27 | **smart_router.py** | Stage 5 APIs | Smart order routing. |
| 28 | **watchlist_manager.py** | `/api/watchlist/*` | Watchlist CRUD operations. |
| 29 | **world_context.py** | `/api/world-context` | World context data API. |
| 30 | **economic_calendar.py** | `/api/economic-calendar` | Economic events API. |
| 31 | **edgar_integration.py** | `/api/edgar/*` | SEC EDGAR filings lookup. |
| 32 | **research/** (6 files) | `/api/research/*` | Deep research, earnings, news, seasonal, historical analysis. |
| 33 | **intelligence/ghost_brain.py** | `/api/intelligence/*` | Ghost Brain analysis API. |
| 34 | **intelligence/opus_brain.py** | `/api/opus/*` | Claude Opus analysis API. |
| 35 | **intelligence/micro_signals/micro_aggregator.py** | `/api/micro-signals` | Micro signal scanning API. |
| 36 | **intelligence/human_behavior/** (2 files) | `/api/narratives`, `/api/influencers` | Narrative detection + influencer tracking APIs. |
| 37 | **intelligence/historical/event_outcomes.py** | `/api/historical-events` | Historical event outcomes API. |
| 38 | **coinbase_provider.py** | `/api/predict/run` (inline) | Crypto price fallback inside prediction store. |
| 39 | **postgres_accuracy.py** | Via `get_accuracy_stats` | Postgres-backed accuracy stats for daily reports. |
| 40 | **context_engine.py** | `/api/context-engine` | Context engine status API (also used in pipeline via stage1). |
| 41 | **model_store.py** | Startup init | Model storage (weights, checkpoints). |
| 42 | **goals_tracker.py** | Startup init | Goal tracking initialization. |
| 43 | **migration_runner.py** | Startup | DB schema migrations. |
| 44 | **ai_memory.py** | `/api/memory/*` | AI memory store. Top-level import (line 2729). |
| 45 | **adapters.py** | Via ghost_notifications | Converts prediction formats (turbo → v3). |
| 46 | **v3_filter.py** | Via adapters | V3 quality filter for predictions. |
| 47 | **stock_engine.py** | `/api/opus/analyze` + pipeline | Also exposed via Opus analyze API. |

---

## D. DEAD CODE (never imported anywhere — 0 import references)

These 31 modules have **zero** imports from `wolf_app.py` OR any other `core/` module.

| # | Module | Description | Status |
|---|--------|-------------|--------|
| 1 | **accuracy_dashboard.py** | Original accuracy dashboard (replaced by v2) | SUPERSEDED |
| 2 | **agent_analytics.py** | Agent decision quality metrics | UNUSED |
| 3 | **agent_tools.py** | ChatGPT analyst tool adapters | UNUSED |
| 4 | **auto_execution.py** | Original broker execution (replaced by autonomous_execution_engine) | SUPERSEDED |
| 5 | **auto_reconciler.py** | Hourly prediction reconciler (replaced by outcome_reconciler_v2) | SUPERSEDED |
| 6 | **backtest_engine.py** | Walk-forward backtesting (replaced by backtester.py) | SUPERSEDED |
| 7 | **backtesting.py** | Another backtesting engine (replaced by backtester.py) | SUPERSEDED |
| 8 | **btc_correlation.py** | BTC correlation features (functionality in pattern_intelligence/) | SUPERSEDED |
| 9 | **cache_tools.py** | Cache namespace management | UNUSED |
| 10 | **config.py** | Centralized config (not actually imported — values duplicated in wolf_app.py) | UNUSED |
| 11 | **crypto_analyzer.py** | Advanced crypto market structure analysis | UNUSED |
| 12 | **daily_predictions_engine_OLD.py** | Old daily predictions engine (explicitly "_OLD") | SUPERSEDED |
| 13 | **event_detector.py** | Market event detection (imports event_memory but nothing imports event_detector) | ORPHANED |
| 14 | **ghost_pattern_trader.py** | Pattern-based trading (imports event_memory but never called) | ORPHANED |
| 15 | **ghost_researcher.py** | AI-powered deep research agent | UNUSED |
| 16 | **honest_telegram_formatter.py** | Honest accuracy formatting for Telegram | UNUSED |
| 17 | **portfolio.py** | Basic Position dataclass | SUPERSEDED (by portfolio_manager) |
| 18 | **prediction_explainer.py** | Prediction explanation generator | UNUSED |
| 19 | **price_validator.py** | Price sanity checks (logic inlined in wolf_app.py) | SUPERSEDED |
| 20 | **risk_dashboard.py** | Risk dashboard calculations | UNUSED |
| 21 | **runtime_config.py** | Dynamic runtime settings | UNUSED |
| 22 | **sentiment_analyzer.py** | Real-time sentiment aggregator (replaced by data_pillars/sentiment_engine) | SUPERSEDED |
| 23 | **service_auto_restart.py** | Background task restart logic | UNUSED |
| 24 | **smart_execution.py** | TWAP/VWAP execution engine | UNUSED |
| 25 | **top_10_scheduler.py** | Old Top 10 scheduler (replaced by daily_top_10_scanner + cron_scheduler) | SUPERSEDED |
| 26 | **trading_automation.py** | Prediction-to-order automation (replaced by autonomous_execution_engine) | SUPERSEDED |
| 27 | **v2_pick_filter.py** | V2 pick quality filter (replaced by v2_quality + market_gates) | SUPERSEDED |
| 28 | **var_calculator.py** | Value at Risk calculator | UNUSED |
| 29 | **volatility_engine.py** | Volatility-triggered prediction engine | UNUSED |
| 30 | **volume_analyzer.py** | Volume accumulation/distribution (replaced by data_pillars/volume_engine) | SUPERSEDED |
| 31 | **watchlist_telegram_alerts.py** | Watchlist-specific Telegram alerts | UNUSED |

### Dead Subdirectory Modules (0 imports):

| # | Module | Description |
|---|--------|-------------|
| 32 | **intelligence/micro_signals/insider_tracker.py** | Insider trading detection | 
| 33 | **intelligence/micro_signals/options_flow.py** | Options flow analysis |
| 34 | **intelligence/micro_signals/social_velocity.py** | Social media velocity |
| 35 | **intelligence/micro_signals/volume_analyzer.py** | Duplicate volume analyzer |
| 36 | **intelligence/micro_signals/whale_detector.py** | Whale transaction detection |
| 37 | **pattern_intelligence/btc_correlation.py** | BTC correlation (dup of btc_correlation.py) |
| 38 | **pattern_intelligence/funding_rates.py** | Funding rate analysis |
| 39 | **pattern_intelligence/gpt4_analyst.py** | GPT-4 analyst integration |
| 40 | **pattern_intelligence/pattern_fingerprint.py** | Pattern fingerprinting |
| 41 | **pattern_intelligence/pattern_matcher.py** | Pattern matching engine |
| 42 | **pattern_intelligence/signal_aggregator.py** | Signal aggregation |
| 43 | **pattern_intelligence/social_sentiment.py** | Social sentiment analysis |

**Total dead code: 43 modules**

---

## E. DISABLED (explicitly disabled via env var or commented out)

| # | Module | Disable Mechanism | Default State |
|---|--------|-------------------|---------------|
| 1 | **scheduled_predictions.py** | `if False:` hardcoded (line 5884) | **PERMANENTLY OFF** |
| 2 | **sl_tp_monitor.py** | Commented out (lines 5342-5347) | **PERMANENTLY OFF** |
| 3 | **order_sync.py** | Commented out (lines 5351-5356) | **PERMANENTLY OFF** |
| 4 | **autonomous_execution_engine.py** | `AUTO_EXECUTION_ENABLED=0` | OFF by default |
| 5 | **orchestrator.py** | `ORCHESTRATOR_ENABLED=0` | OFF by default |
| 6 | **ai_advisor/scanner.py** | `AI_ADVISOR_ENABLED=0` | OFF by default |
| 7 | **pattern_enhanced_predictor.py** | `ENABLE_PATTERN_INTELLIGENCE=1` | ON by default |
| 8 | **trust_ladder.py** | `TRUST_LADDER_ENABLED=1` | ON by default |
| 9 | **position_sizer.py** | `POSITION_SIZER_ENABLED=1` | ON by default |
| 10 | **stage1_integration.py** | `STAGE1_ENABLED` (try/except) | ON if imports work |
| 11 | **ghost_notifications.py** | `ACTIVE_TRACKING_ENABLED=1` | ON by default |
| 12 | **intelligence/ghost_news_brain.py** | `NEWS_ANALYSIS_ENABLED=1` | ON by default |
| 13 | **smart_scout.py** | `MONEY_GAME_ENABLED=1` | ON by default |

---

## F. SUPPORT MODULES (imported by other core modules, not directly by wolf_app.py)

| # | Module | Used By | Import Count |
|---|--------|---------|:---:|
| 1 | **money_game_engine.py** | ghost_scout, smart_scout, ghost_notifications, wolf_app APIs | 19 |
| 2 | **alpaca_broker.py** | autonomous_execution_engine, risk_manager, sl_tp_monitor, order_sync | 24 |
| 3 | **db_engine.py** | migration_runner | 6 |
| 4 | **event_memory.py** | ghost_notifications, pattern_tracker, event_detector | 6 |
| 5 | **pattern_tracker.py** | stock_engine, event_detector, event_memory | 7 |
| 6 | **v3_competition.py** | v3_shadow_predictor, v3_shadow_resolver | 7 |
| 7 | **v2_verification.py** | v2_quality | 3 |
| 8 | **opportunity_scorer.py** | telegram_hunter | 3 |
| 9 | **regime_filter.py** | ghost_notifications | 3 |
| 10 | **personal_watchlist.py** | watchlist_prediction_scheduler | 7 |
| 11 | **news_sentiment.py** | ghost_scout, market_gates | 3 |
| 12 | **trade_decision_engine.py** | autonomous_execution_engine | 1 |
| 13 | **active_tracking.py** | top10_aggregator | 3 |
| 14 | **top10_aggregator.py** | telegram_alerts | 4 |
| 15 | **polygon_integration.py** | Various providers | 7 |
| 16 | **providers/stock_providers.py** | pattern_tracker, world_context | 3 |
| 17 | **providers/binance_ohlcv.py** | market_gates, regime_filter | 3 |
| 18 | **providers/unified_provider.py** | daily_top_10_scanner | 3 |
| 19 | **providers/cache_utils.py** | Provider internals | 1 |
| 20 | **providers/yahoo_finance.py** | Provider internals | 1 |
| 21 | **crypto/vip_providers.py** | vip_scanner | 8 |
| 22 | **crypto/crypto_predictor.py** | smart_scout | 4 |
| 23 | **crypto/crypto_watchlist.py** | Various | 5 |
| 24 | **features/technical_indicators.py** | Feature engines | 2 |
| 25 | **metrics/ghost_score.py** | Various | 3 |
| 26 | **risk/risk_guard.py** | Various | 4 |
| 27 | **sector_momentum.py** | stock_gates, stock_engine | 3 |
| 28 | **economic_calendar.py** | stock_gates, stock_engine, wolf_app API | 5 |
| 29 | **multi_timeframe.py** | stock_gates | 3 |
| 30 | **online_calibrator.py** | paper_tracker, wolf_app startup | 8 |
| 31 | **world_feed_fusion.py** | feature_importance | 9 |

---

## Summary Statistics

| Category | Count | % of Total |
|----------|:-----:|:----------:|
| **A. Prediction Pipeline** | 39 modules | 22% |
| **B. Background Workers** | 34 modules | 19% |
| **C. API-Only** | 47 modules | 27% |
| **D. Dead Code** | 43 modules | 25% |
| **E. Disabled** | 3 permanently + 10 env-gated | — |
| **F. Support (cross-imported)** | 31 modules | 18% |

> Note: Some modules appear in multiple categories (e.g., `context_engine.py` is both pipeline and API).

### Key Findings

1. **43 dead modules (~25%)** can be safely deleted or archived. Most are superseded by newer implementations.
2. **The prediction pipeline has 39 active dependencies** — every prediction touches price fetching, 6 data pillars, ensemble voting, 5+ confidence adjusters, market gates, and paper trade logging.
3. **3 modules are permanently disabled** (scheduled_predictions, sl_tp_monitor, order_sync) — commented out or hardcoded `if False`.
4. **The entire `pattern_intelligence/` subdirectory** (7/8 files) is dead code — only `fear_greed.py` is imported (by `market_gates.py`).
5. **5 of 6 `intelligence/micro_signals/` files** are dead — only `micro_aggregator.py` is used via API.
6. **Broker execution path** (autonomous_execution_engine → alpaca_broker → risk_engine) is default OFF (`AUTO_EXECUTION_ENABLED=0`).
