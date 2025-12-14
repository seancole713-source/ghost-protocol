# Ghost Protocol — Profit-Ready Deliverables (AUTHORITATIVE)

## 1) Architecture map (Mermaid) + legend

```mermaid
flowchart TD
  %% Ingestion
  A[Symbol universe\n(STOCK_SYMBOLS/CRYPTO_SYMBOLS)] --> B[Price ingestion\n(core.providers + turbo_provider)]
  B --> C[Price quorum / consensus\n(core.price_quorum.get_price_quorum)]
  C -->|consensus OK| D[Feature extraction\n(run_single_prediction -> features dict)]
  C -->|quorum_failed / no_quotes| X[Fail-closed\nshould_predict=false, action=HOLD]

  %% Optional context/news
  N[News ingestion\n/api/news + context_engine (if enabled)] --> D

  %% Inference
  D --> E[Model inference\n(ensemble + regime + signals)]
  E --> F[Signal-based confidence calibration\n(core.confidence_calibrator)]
  F --> G[Outcome-driven calibration\n(expected accuracy)]

  %% Guardrails
  G --> H[Degraded-data guardrails\n(min feature availability)]
  H -->|degraded| X
  H -->|ok| I[Touch-target gating\n(core.touch_calibration_sqlite)]

  %% Telegram
  I -->|Stage5 >=70%| J[Message composition\n(core.telegram_alerts)]
  I -->|<70%| X
  J --> K[Send\nTelegram Bot API]

  %% Storage
  E --> S1[Prediction storage\n(core.prediction_store: SQLite/Postgres)]
  I --> S2[Touch target row\nwolf.db ghost_predictions]
  D --> S3[Feature/forecast tracking\ncore.accuracy_tracker forecasts]

  %% Scoring + feedback
  S1 --> R[Outcome reconciliation\n(outcomes table hit_direction, map, rmse)]
  S2 --> R
  R --> L[Learning loop\n(core.learning_loop / calibration curves)]
  L --> G

  %% Observability
  O1[/health + /api/v3/cockpit/status] --> O2[Regression auditor]
  O2 --> K
```

**Legend**
- **Consensus**: `core.price_quorum` collects multiple provider quotes; if quorum fails, the decision returns `reason=quorum_failed` and live predictions fail closed.
- **Degraded-data guardrails**: in `wolf_app.py` degraded/low-feature predictions are marked `should_predict=false` and never become Telegram signals.
- **Stage5/Stage6 gate**: `core.touch_calibration_sqlite` computes calibrated touch probabilities; Stage5 is the minimum 70% execution gate for sending.
- **Storage**
  - Predictions/outcomes: `core.prediction_store` (SQLite or Postgres)
  - Touch-target tracking: `wolf.db` / `ghost_predictions`
  - Forecast/feature tracking: `data/forecast_accuracy.db` via `core.accuracy_tracker`

---

## 2) Definition of accuracy + gating rules

**Primary metric (proof metric): Directional Hit Rate (DHR)**
- Source of truth: `prediction_store` `outcomes.hit_direction` (0/1)
- Definition: $\text{DHR} = \frac{\sum hit\_direction}{N}$ over a window (default last 30 days)
- This is what “≥70% accuracy” means for gating.

**Secondary metric: MAPE (Mean Absolute Percentage Error)**
- Source of truth: `prediction_store` `outcomes.map`
- Definition: average of `map` over the same window.

**Tolerance band (default)**
- Magnitude tolerance band: ±1.0% (default)
- Derived measure: `within_1pct_band_rate` = fraction of outcomes where `map <= 1.0`.

**Gating rules (must be true to alert)**
- **Live-only**: `SIM_MODE` must be `0` in production (enforced when `ENFORCE_LIVE=1`).
- **Fail-closed on live data**: if price quorum fails or features are degraded → `should_predict=false` and it never alerts.
- **Hard 70% gate**:
  - Preferred: `stage5_ok == true` (calibrated touch-probability ≥ 70% at ±1.0%).
  - Fallback: raw confidence ≥ `MIN_ALERT_CONFIDENCE` (defaults to 0.70).

---

## 3) Telegram format (exact template)

**Morning digest (Cash App style, plain text)**

```
GHOST — Daily Picks
As of: {ISO_TIMESTAMP}
Window: {WINDOW_LABEL}

{TICKER}  {PREDICTED_PCT:+.2f}%  ({CONFIDENCE_PCT}%)
- {WHY_SHORT}
{TICKER}  {PREDICTED_PCT:+.2f}%  ({CONFIDENCE_PCT}%)
- {WHY_SHORT}
...
```

**Per-signal (touch-target gated, plain text)**
- Uses `core.telegram_alerts.format_touch_target_signal()` which includes:
  - timestamp (implicit in send time), horizon/window, entry/target/stop, predicted move, confidence
  - gate tier + calibrated probabilities
  - short reasoning summary (signals fired + feature count)

---

## 4) Regression auditor checklist + single command

**Checklist (what the auditor verifies)**
- Git auth: can push from container (`git push --dry-run origin HEAD`).
- Deploy identity: `GET /health` contains `git_sha` and matches local `HEAD` prefix.
- Endpoints:
  - `GET /health`
  - `GET /api/v3/cockpit/status`
  - `GET /api/v3/predictions/latest`
  - `GET /cockpit` (HTML)
- Telegram send test (optional): controlled single test message via regression key.
- Accuracy tracker row after cycle:
  - reads `GET /api/v3/accuracy/tracker/status` (rows_total)
  - triggers one live cycle via `GET /api/predictions/run?symbol=SPY`
  - re-reads tracker status and confirms rows_total increments

**Single command**
```bash
python3 regression_audit.py --base-url https://ghost-protocol-production.up.railway.app
```

Optional Telegram check:
```bash
REGRESSION_KEY=*** python3 regression_audit.py --base-url https://ghost-protocol-production.up.railway.app --telegram-test
```

---

## 5) Verification results (facts-only)

Run `regression_audit.py` against production and paste the JSON output here.
