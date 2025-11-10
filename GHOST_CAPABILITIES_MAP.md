# Ghost Capabilities Map — October 2025

## 1. Executive Snapshot

- **Mission**: Transform Ghost into a GPT-class, self-learning trading brain with
  live-first intelligence, deep memory, explainable decisions, and automation-ready
  hooks.
- **Core Strengths**: Mature FastAPI backend, rich cockpit UI, live multi-provider
  pricing (Alpha Vantage, Polygon, Yahoo/CoinGecko), SSE streaming, advisory workflows,
  alerting infrastructure, and comprehensive health/ops tooling.
- **Primary Gaps**: Limited AI memory (volatile deque), single-model forecasting, WOLF
  ticker delisted, underpowered decision engine (heuristics), no regime-aware risk,
  multi-asset intelligence not unified, UI overlay incomplete.
- **Immediate Priorities**: (1) Deploy new persistent AI memory, (2) migrate focus
  ticker, (3) modernize forecasting + decisioning, (4) finish prediction-vs-reality UI,
  (5) embed risk & regime intelligence.

## 2. Data & Provider Layer

| Capability | Current State | Data Sources | Gaps / Pain Points | Upgrade Opportunities
|
|------------|---------------|--------------|--------------------|-----------------------|
| **Equity Prices** | Live fetching via `enhanced_price_fetcher` with Alpha Vantage
primary, Polygon/Yahoo fallback. Prev-close cache, plausibility gates | Alpha Vantage,
Polygon, Yahoo HTTP | WOLF delisted → stale prices, rate limits cause WARNs, no quorum
consensus | Migrate to NVDA (or configurable basket), add quorum voting + confidence
weighting, stitch intraday history in `_collect_actual_prices` | | **Crypto Prices** |
CoinGecko live, stable | CoinGecko | Latency under load | Add Kraken/Coinbase secondary
for redundancy | | **News & Sentiment** | Polygon news feed optional, summarizer via
`llm/agent.py` | Polygon, internal summarizer | Sentiment scores not persisted, rate
limit fallback missing | Add sentiment scoring + storage in `ai_memory` | | **Market
Metadata** | Catalog (CoinGecko Top-500 + S&P 500) with search + paging | Static JSON +
CoinGecko | No regime data, no sector/factor context | Enrich with sector/industry,
volatility, correlations | | **State Persistence** | File/Redis options, Prometheus
metrics, SQLite for forecasts | Local FS / Redis / SQLite | Forecast grid persisted
transiently; memory limited | Expand SQLite schema for AI memory, regimes, RL metrics |

## 3. Intelligence & Learning Stack

| Layer | Current Implementation | Strengths | Limitations | Targeted Upgrades |
|-------|-----------------------|-----------|-------------|--------------------| |
**Memory** | In-memory deque (`AI_MEMORY`) capped at 100 entries; manual pruning |
Low-latency append/read | Volatile, no semantic search, no outcome linkage | ✅
`core/ai_memory.AIMemory` (SQLite + embeddings) to be integrated, add calibration,
summaries | | **Forecasting** | `_build_forecast_series` drift model w/ heuristics;
`_build_two_line_forecast` ready | Deterministic, stable grid generation, SSE streaming
| Single-model, no ML ensembles, limited accuracy metrics | Ensemble forecasters
(LSTM/XGBoost/Prophet) w/ dynamic weighting, residual analytics | | **Accuracy
Tracking** | `_compute_forecast_accuracy` with MAP/RMSE/Bias; SSE hook | Already wired
to UI & Prometheus | Historical backfill limited by actual data quality | Expand
realized price ingestion, add calibration curves & reliability plots | | **Decision
Engine** | Rules-based heuristics + Ghost Score; `/ai/decide` uses simple prompts |
Transparent base logic | Lacks RL/optimization, limited confidence scoring, no regime
awareness | PPO reinforcement learner, risk-adjusted scoring, explanation generation via
AI memory | | **Learning Signals** | K-NN prototypes in `llm/agent.py`; manual backtests
| Extensible base | No automated retraining, limited feature set | Add scheduled
retraining from AI memory, feature store | | **Automation Hooks** | Telegram alerts,
pseudo-orders, toggled manually | Infrastructure ready | No direct broker integration
(by design), limited triggered sequences | Expand to webhook automations, confirm risk
checks before signals |

## 4. User Experience & Visualization

- **Cockpit UI**: Vanilla JS + Canvas; sections for news, positions, diagnostics,
  forecast overlay.
- **SSE**: `/events` heartbeat + `/api/cockpit/stream` for forecast updates (backend
  complete).
- **Forecast Overlay**: Backend returns `two_line_overlay` with forecast, actual,
  accuracy metrics; frontend currently missing dual-line render + accuracy chips
  binding.
- **UI Baseline Contract**: `ui_dist/` provides frozen bundle; `templates/cockpit.html`
  is customizable (needs updates under `[ALLOW_UI_CHANGE]`).
- **Planned Enhancements**:
  - Render Ghost vs Live lines (solid vs dashed) with gap handling.
  - Surface confidence, regime badges, forecast source provenance.
  - Add memory + rationale panel, RL recommendation cards with explanations +
    confidence.
  - Introduce asset switcher for multi-asset intelligence.

## 5. Persistence, Ops & Observability

- **Datastores**: SQLite (`WOLF_SQLITE_PATH`), optional Redis; forecast tables exist
  (`realized_prices`, `forecast_scores`).
- **Logging**: Structured logging via `LOGGER.info`/`error`; event bus `_add_event` for
  UI diagnostics.
- **Metrics**: `/metrics` exposes Prometheus counters/histograms.
- **State Manager**: `state_manager.py` handles snapshot persistence, but
  memory/training artifacts not yet persisted.
- **Upgrades**: Extend SQLite schema for AI memory (`ai_memory` table), RL stats, regime
  history. Add migrations, background vacuum, health checks.

## 6. Gap & Impact Heatmap

| Area | Current Score (1-5) | Business Impact | Upgrade Priority | Notes |
|------|---------------------|-----------------|------------------|-------| | **Live
Data Integrity** | 3 | Critical | P0 | Ticker migration + quorum consensus | | **AI
Memory** | 2 | Critical | P0 | Replace deque with AIMemory, add tooling | | **Forecast
Accuracy** | 2 | High | P0 | Ensemble + calibration | | **Decision Intelligence** | 2 |
High | P0 | RL engine, explanation layer | | **Risk Management** | 2 | High | P0 |
Kelly, VaR/CVaR, regime gating | | **Prediction Overlay UI** | 3 | Medium | P0 |
Complete UI to reflect live accuracy | | **Automation Hooks** | 3 | Medium | P1 | Expand
triggers + webhooks | | **Multi-Asset Coverage** | 3 | Medium | P1 | Portfolio
optimization, correlations | | **Sentiment & News Learning** | 2 | Medium | P1 | Persist
sentiment, integrate into memory |

## 7. Upgrade Roadmap (Phase 0 → Phase 3)

- **Phase 0 (Now)**

  1. Integrate `core/ai_memory.AIMemory` into `wolf_app.py` (persisted recall, semantic
     search, calibration endpoints).
  2. Finish two-line overlay frontend (solid vs dashed, accuracy chips, SSE refresh).
  3. Migrate focus ticker from WOLF → NVDA (or configurable symbol). Update providers,
     tests, AI memory seeds.

- **Phase 1**

  1. Deploy ensemble forecasting stack with dynamic weighting + calibration metrics
     stored in memory.
  2. Launch reinforcement learning decision engine (PPO) with risk-aware scoring; log
     rationale to memory.
  3. Introduce market regime detector + risk engine (Kelly, VaR, drawdown guard) gating
     recommendations.

- **Phase 2**

  1. Expand multi-asset intelligence, correlation matrix, hedging suggestions.
  2. Persist sentiment & news embeddings, integrate into decision prompts.
  3. Add automation hooks (Telegram, webhooks) with safety rails + audit trail.

- **Phase 3**

  1. Autonomous retraining schedules, drift detection, model governance dashboard.
  2. Advanced visualization (probability cones, scenario analysis, explainability
     panels).
  3. Broker-neutral API abstraction for future order execution (optional toggle).

## 8. Key Files & Ownership

| Domain | Key Files | Owner Notes | |--------|-----------|-------------| | Backend Core
| `wolf_app.py`, `state_manager.py`, `enhanced_price_fetcher.py` | High-complexity;
coordinate changes with tests + health scripts | | AI & Learning | `core/ai_memory.py`,
`llm/agent.py`, `enhanced_price_fetcher.py`, planned `core/forecasters.py`,
`core/rl_engine.py` | New modules fall under AI evolution charter | | UI |
`templates/cockpit.html`, `static/ghost.js`, `ui_dist/index.html` | UI baseline contract
applies; use `[ALLOW_UI_CHANGE]` when modifying | | Infrastructure |
`utils/verify_live.py`, `ghost_diag.sh`, `docker-compose.yml` | Update verification
tooling as capabilities grow |

## 9. Next Steps Snapshot

1. **Integrate AIMemory** and expose new AI memory APIs for recall/calibration.
2. **Update cockpit overlay** to visualize prediction vs reality with live metrics.
3. **Switch ticker** to a liquid symbol (NVDA) and validate provider parity.
4. **Build advanced forecasting ensemble** + calibration logging.
5. **Implement RL + risk-aware decisioning** with explainable outputs.
6. **Extend to multi-asset intelligence** and automation hooks.

> This capabilities map will be versioned as we ship each phase. Update the "Current
> Score" and Roadmap after every major release to keep Ghost aligned with the GPT-class
> vision.
