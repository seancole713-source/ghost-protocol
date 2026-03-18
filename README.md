# Ghost Protocol

**Autonomous AI trading system** — directional predictions for crypto & stocks with ML ensemble, confidence calibration, and self-improvement loops.

## Architecture

```
wolf_app.py          ← FastAPI monolith (entrypoint: wolf_app:APP)
state.py             ← Shared global state (predictions, memory, heartbeat)
routes/              ← Extracted API route modules (APIRouter)
  picks.py             /api/v4/picks — today's trade picks
  history.py           /api/v4/history — resolved prediction history
  heartbeat.py         /api/v3/heartbeat/status — background task health
  subsystems.py        /api/v4/subsystems — full subsystem inventory
  news_routes.py       /api/news/* — news feed endpoints
  crypto_ohlcv_routes.py  /api/crypto/ohlcv/* — crypto OHLCV data
core/                ← Prediction engines, intelligence hub, brains
  ghost_brain.py       Core prediction engine
  ghost_learning_brain.py  Self-correction (symbol inversion)
  ensemble_predictor.py    XGBoost + market regime signals
  confidence_calibrator.py Confidence → accuracy mapping
  intelligence_hub.py  20-subsystem intelligence orchestrator
  quality_gate.py      Prediction quality filtering
  heartbeat.py         Background task monitoring
  db_pool.py           PostgreSQL connection pool
  intelligence/        Sub-modules (opus_brain, etc.)
  providers/           Price data providers (turbo, unified, coinbase)
config/              ← Symbol lists, settings
services/            ← Outcome reconciler v2, predictor, price collector
api/                 ← Cockpit UI endpoints (v2, v3), debug routes
notifications/       ← Telegram alerts, message formatters
llm/                 ← GPT-4 analyst, LLM agent integration
backtest/            ← Backtesting engine + strategies
models/              ← Trained ML models (XGBoost, ensemble)
static/              ← CSS, JS, images for cockpit UI
templates/           ← HTML templates for cockpit
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL="postgresql://..."   # PostgreSQL (Railway)
export TELEGRAM_BOT_TOKEN="..."          # Telegram alerts
export TELEGRAM_CHAT_ID="..."
export POLYGON_KEY="..."                 # Stock price data
export OPENAI_API_KEY="..."              # GPT-4 analysis (optional)

# Run
uvicorn wolf_app:APP --host 0.0.0.0 --port 8080
```

## Key Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Load balancer health check |
| `GET /cockpit` | Web dashboard |
| `GET /api/v4/picks` | Today's trade picks |
| `GET /api/v4/history` | Resolved prediction history |
| `GET /api/v4/subsystems` | Full subsystem inventory |
| `GET /api/v3/heartbeat/status` | Background task health |
| `POST /api/predict/run` | Trigger single prediction |

## How It Works

1. **Prediction Cycle** — Every 60 min, predicts direction for 600+ symbols (crypto + stocks)
2. **Ensemble Engine** — XGBoost + technical analysis + sentiment + momentum
3. **Intelligence Hub** — 20 subsystems: quality gate, trust ladder, confidence calibration, kill switch, VWAP, regime detection
4. **4 Brains** — Ghost Brain (core ML), Learning Brain (self-correction), News Brain (sentiment), Opus Brain (GPT-4)
5. **Outcome Tracking** — 48h reconciler checks predictions against actual prices
6. **Self-Improvement** — Learning Brain inverts persistently wrong symbols, calibrator adjusts confidence

## Deployment

Runs on **Railway** with PostgreSQL. Entry point: `wolf_app:APP` (Procfile: `web: uvicorn wolf_app:APP`).

## Commit History

| Step | Description |
|---|---|
| 1-4 | Foundation: cockpit, health checks, prediction gates |
| 5-7 | Bug fixes: evaluator, Opus Brain, Learning Brain |
| 8-9 | Direction mismatch fix, audit cleanup |
| 10 | **Structural cleanup** — deleted 700+ dead files, extracted routes, organized modules |
