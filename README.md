# 👻 GHOST Protocol v10.2 (FastAPI, Live-First)

![Smoke Gate](<<<<<https://github.com/seancole713-source/GHOST/actions/workflows/ghost_smoke.yml/badge.sv>>>>>g)

Ghost is a FastAPI app with a clean cockpit UI for live quotes/signals, advisor
allocations, goals, and diagnostics. It runs live by default; a lightweight SIM mode is
available for demos and tests and can be toggled at runtime.

## 🚀 Features

- **Cockpit Dashboard**: Jinja UI with health banner; optional external React UI via

  `GHOST_UI_URL`

- **AI-Powered Analysis**: Ghost Score and Fusion AI for market insights
- **Portfolio Management**: Track positions, cash balance, and performance
- **Goal Setting**: Set and track financial goals with progress monitoring
- **Enhanced Advisor**: AI recommendations for portfolio optimization
- **Multi-Asset Support**: Cryptocurrencies and stocks
- **Ethereum Integration**: Web3 connectivity for DeFi operations
- **Real-time Data (Live)**: Live quotes for crypto via CoinGecko and stocks via Alpha

  Vantage (primary), Polygon (optional), yfinance (fallback). Optional SIM mode provides
  deterministic prices for demos/tests.

- **Advisory-Only**: Generate signals and queue pseudo-orders; no execution
- **Catalog**: Crypto Top-500 (CoinGecko) + S&P 500 constituents with search and paging

## 📋 Prerequisites

- Python 3.11+
- Optional: Alpha Vantage API key for equities, CoinGecko Pro key, Polygon key

## 🛠️ Local setup

1. **Clone the repository:**```bash

   git clone <<<<<https://github.com/seancole713-source/GHOST.git>>>>>
   cd GHOST

   ```text

1.**Install dependencies:**```bash

   pip install -r requirements.txt

   ```text

1.**Environment variables:**```bash

   cp .env.example .env

   ```text

   Key vars:

   - `SIM_MODE=0` to run live, `SIM_MODE=1` to start in SIM. You can toggle at runtime


     via `POST /api/mode`.

   - `ALPHAVANTAGE_API_KEY` or `ALPHA_VANTAGE_API_KEY` (either name works)
   - `COINGECKO_API_KEY` (optional Pro key)
   - `POLYGON_API_KEY` (optional)
   - `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` (alerts)
   - `ENABLE_REDIS=1` with `REDIS_URL` to persist state
   - `GHOST_UI_URL` to redirect to an external UI (or use built-in Jinja UI)
   - `ALLOWED_ORIGINS` for CORS (comma-separated)


   Minimal WOLF-only env (alternative):

   ```bash

   cp .env.wolf.example .env

   ```text

   Then set `ALPHAVANTAGE_API_KEY` (and `POLYGON_API_KEY` optionally) and Telegram vars
   if you want alerts. Optional persistence & heartbeat:

   - `WOLF_PERSIST_MODE=none|file|redis` (default: none)
   - For file: `WOLF_STATE_FILE=/data/wolf_state.json`
   - For redis: `REDIS_URL=redis://redis:6379/0`
   - `TELEGRAM_HEARTBEAT_ON_START=1` to send a status card on startup


## 🏃‍♂️ Running the Application

1.**Run the FastAPI server:**```bash

   uvicorn wolf_app:app --host 0.0.0.0 --port 5000

   ```text

1.**Open your browser and navigate to:**


   ```text

   <<<<<http://localhost:5000>>>>>

   ```text

The Ghost Trading Bot cockpit will be available with all features accessible through the
web interface.

If using GitHub Codespaces, forward port 5000 as public and use the provided URL, like:

```text

<<<<<https://<your-space>-5000.app.github.dev/>>>>>

```text

Tip: In VS Code, use Tasks → Run Task → "Run Ghost server (:5000)" or "Verify Live
(utils/verify_live.py)" for one-click workflows.

## 🔎 Live verification (one‑liner)

Run the end-to-end verifier against a live or local instance. It performs 10 checks
(health, provider parity, math audit, ETag stability, alerts dry-run, freshness, and ops
metrics) and prints PASS/WARN/FAIL with a final summary. The script exits non-zero only
if more than 2 checks fail.

Prereqs: install deps once

```bash

pip install -r requirements.txt

```text

Run against local

```bash

# Optional providers enable deeper parity checks; safe to omit

GHOST_URL=<<<<<http://127.0.0.1:5000>>>>> \
GHOST_API_TOKEN="${GHOST_API_TOKEN:-}" \
ALPHAVANTAGE_API_KEY="${ALPHAVANTAGE_API_KEY:-}" \
POLYGON_API_KEY="${POLYGON_API_KEY:-}" \
python utils/verify_live.py

```text

Run against a remote URL

```bash

GHOST_URL="<<<<<https://your-host.example.com">>>>> \
GHOST_API_TOKEN="<token-if-required>" \
python utils/verify_live.py

```text

Notes

- Price parity tries Alpha Vantage first, then Polygon prev-close; WARNs are expected if


  rate-limited or keys are unset.

- News parity is skipped unless `POLYGON_API_KEY` is provided (prints WARN but does not


  fail).

- Alerts test uses dry-run and respects Bearer auth when `GHOST_API_TOKEN` is set.
- Freshness threshold adapts to `TICK_INTERVAL_S` (defaults to 5s).


## ✅ Zero-placeholder + smoke gate

Ghost now enforces a single gate for placeholder scanning and cockpit health:

```text

scripts/check_no_placeholders.sh

```text

- Fails if any banned tokens (see `GHOST_NO_PLACEHOLDER_ENFORCEMENT.md` for patterns) appear in


   runtime files.

- Hits `/health`, `/cockpit`, and `/api/v3/cockpit/version` on the active target


   (local by default, `RAILWAY_URL` when set) to guarantee Cockpit V3 is the only UI.

- Runs automatically via `.githooks/pre-push` (install with `scripts/install_hooks.sh`)


   and in CI (`.github/workflows/ghost_smoke.yml`).

For production verification, run `scripts/check_railway_service.sh <railway-url>` right
after a deploy to confirm the live service matches the cockpit contract.

### UI Baseline (Frozen)

The cockpit UI in `ui_dist/`, `static/`, and `templates/` is locked as a baseline.
Backend/data updates are fine, but structural UI changes require an approved RFC. CI
will block PRs that modify frozen files unless the PR has the `ui-change-approved` label
or includes `[ALLOW_UI_CHANGE]` in the description (see `docs/UI_BASELINE_CONTRACT.md`).

### Environment setup (WOLF-only)

For a live, WOLF-only deployment with no placeholders, copy `.env.wolf.example` to
`.env` and adjust values as needed. That file is populated with concrete defaults
including an IP allowlist and build metadata. If you change networks or rotate secrets,
update `.env` accordingly. Never commit your real `.env` back to the repo.

Live price tuning (env, optional):

- `PRICE_TTL_OPEN_S` — Cache TTL for price ticks during market hours (default: 45).


  Helps reduce provider calls under rate limits.

- `PRICE_YAHOO_FIRST` — If set to 1/true, tries Yahoo HTTP provider first, then falls


  back to others.

- `PRICE_MAX_DEVIATION_OPEN` — Maximum allowed deviation ratio during market hours when


  plausibility gating prices (default inherits `PRICE_MAX_DEVIATION`, 0.5).

- `PRICE_PREV_ONLY_RESPECT_TTL` — If 1/true, when only prev_close is available and still


  fresh, return it without hammering providers.

These settings are reported via `GET /api/config` under `providers` and `ttl`, and also
shown in `/debug/price`.

## 📡 Key Endpoints

| Endpoint | Method | Description | |----------|--------|-------------| | `/` | GET |
Main cockpit dashboard UI | | `/ghostscore` | GET | AI performance scoring | | `/health`
| GET | JSON health; includes error_count | | `/api/secrets/health` | GET | Secrets
presence + provider probes | | `/events` | GET | SSE stream (heartbeat) | |
`/api/signals` | GET | Live trading signals (live-only) | | `/api/allocations/compute` |
POST | Compute position sizes from signals | | `/orders/place` | POST | Queue
pseudo-orders (advisory) | | `/orders/queue` | GET | Pending pseudo-orders | |
`/api/quotes?symbols=BTC,AAPL` | GET | Batch quotes with provider and ts | | `/api/mode`
| GET/POST | Get or set runtime mode (`{"enabled": true}` to enable SIM) | |
`/portfolio` | GET | Portfolio overview | | `/advisor/enhanced` | GET | Enhanced AI
advisor | | `/goals` | GET | Financial goals | | `/goals/lock` | POST | Lock/unlock
goals | | `/goals/progress` | GET | Goals progress tracking | | `/api/bank` | GET | Cash
and ledger | | `/api/set_cash` | POST | Set cash balance | | `/api/bank/reset` | POST |
Reset account | | `/api/positions/add` | POST | Add position + ledger | |
`/api/positions/close` | POST | Close position + ledger | | `/api/advisor_refresh` |
POST | Refresh AI analysis | | `/api/goal_plan` | POST | Create/update goals | |
`/source/status` | GET | System health check | | `/catalog/status` | GET | Catalog
status | | `/catalog/search?q=eth` | GET | Catalog search | |
`/watchlist?top=mixed&n=25&page=1` | GET | Watchlist with live prices | | `/logs/recent`
| GET | Recent logs |

Additional endpoints:

- Alert thresholds can be tuned via environment variables:
  - `ALERT_BUY_PCT` (default: 0.99) — BUY if price < avg_cost * ALERT_BUY_PCT
  - `ALERT_SELL_PCT` (default: 1.01) — SELL if price > avg_cost * ALERT_SELL_PCT
  - `ALERT_THROTTLE_S` (default: 60) — Minimum seconds between duplicate alerts
  - `ALERT_MODE` (fixed|band|trailing; default: fixed). For band/trailing:
    - `BAND_PCT` (default: 0.02)
    - `TRAIL_SELL_PCT` / `TRAIL_BUY_PCT` (default: 0.05 / 0.05)
  - Volatility gate (optional): `VOL_GATE=1` with `VOL_LOOKBACK_DAYS` (20), `VOL_K`


    (1.0), `VOL_TTL_S` (600)

## 🐺 WOLF Quickstart

Minimal flow to set your WOLF position, preview the signal, and dispatch an alert:

- Get current health
  - `GET /health`
- Check current position
  - `GET /api/position`
- Set/update position (qty and average cost)
  - `POST /api/position` with body `{ "qty": <float>, "avg_cost": <float> }`
  - If `GHOST_API_TOKEN` is set, include `Authorization: Bearer $GHOST_API_TOKEN`
- Preview signal (BUY/SELL/HOLD)
  - `GET /api/alerts`
- Dispatch alert (dedupe + throttle; Telegram if configured)
  - `POST /api/alerts/dispatch` (requires Bearer token if `GHOST_API_TOKEN` is set)
  - Dry-run (no Telegram; still records metrics with result="dry-run"):
    - `POST /api/alerts/dispatch?dry_run=1`
- Cockpit snapshot JSON
  - `GET /api/cockpit`
- Prometheus metrics
  - `GET /metrics`


Alert knobs (env): `ALERT_BUY_PCT`, `ALERT_SELL_PCT`, `ALERT_THROTTLE_S`, `ALERT_MODE`
(fixed|band|trailing), `BAND_PCT`, `TRAIL_SELL_PCT`, `TRAIL_BUY_PCT`, optional
volatility gate: `VOL_GATE`, `VOL_LOOKBACK_DAYS`, `VOL_K`.

Grafana dashboard: import `docs/grafana/ghost_wolf_dashboard.json` and point to your
Prometheus datasource.

Quickstart script:

```bash

bash utils/wolf_quickstart.sh                 # uses HOST env (default <<<<<http://127.0.0.1:500>>>>>0)
QTY=10 AVG=25.50 bash utils/wolf_quickstart.sh  # also updates position

```text

curl -X POST <<<<<http://localhost:5000/api/alerts/config>>>>> \
-H 'Content-Type: application/json' \
-H "Authorization: Bearer $GHOST_API_TOKEN" \
-d '{"mode":"band","band_pct":0.02,"vol_gate":1,"vol_k":1.0}' | jq

# Switch to trailing mode

curl -X POST <<<<<http://localhost:5000/api/alerts/config>>>>> \
-H 'Content-Type: application/json' \
-H "Authorization: Bearer $GHOST_API_TOKEN" \
-d '{"mode":"trailing","trail_sell_pct":0.06,"trail_buy_pct":0.04}' | jq

# Update position (protected when GHOST_API_TOKEN is set)

curl -X POST <<<<<http://localhost:5000/api/position>>>>> \
-H 'Content-Type: application/json' \
-H "Authorization: Bearer $GHOST_API_TOKEN" \
-d '{"qty": 10, "avg_cost": 25.5}' | jq

# Toggle HOLD (protected when GHOST_API_TOKEN is set)

curl -X POST <<<<<http://localhost:5000/api/alerts/hold>>>>> \
-H 'Content-Type: application/json' \
-H "Authorization: Bearer $GHOST_API_TOKEN" \
-d '{"hold": true}' | jq

# Dispatch alert (protected when GHOST_API_TOKEN is set)

curl -X POST "<<<<<http://localhost:5000/api/alerts/dispatch">>>>> \
-H "Authorization: Bearer $GHOST_API_TOKEN" | jq

# Dry-run dispatch (no Telegram; increments ghost_alerts_sent_total{result="dry-run"})

curl -X POST "<<<<<http://localhost:5000/api/alerts/dispatch?dry_run=1">>>>> \
-H "Authorization: Bearer $GHOST_API_TOKEN" | jq

````

Send alert (dedupe/throttle):

```bash

curl -X POST <<<<<http://localhost:5000/api/alerts/dispatch>>>>>

````

Send dry-run (no Telegram):

```bash

curl -X POST "<<<<<http://localhost:5000/api/alerts/dispatch?dry_run=1">>>>>

```text

Send STATUS card on demand (protected):

```bash

curl -X POST "<<<<<http://localhost:5000/alerts/status">>>>> -H "Authorization: Bearer $GHOST_API_TOKEN"

```text

## New configuration (persistence, metrics, security)

Environment variables:

- WOLF_PERSIST_MODE: none|file|redis|sqlite|auto (default: none). Use auto to try


  redis→sqlite→file.

- WOLF_STATE_FILE: file path for file persistence (default: /data/wolf_state.json).
- REDIS_URL: Redis connection URL for persistence.
- WOLF_SQLITE_PATH: SQLite DB path for persistence (default: /data/wolf.db).
- WOLF_AUTOSAVE_S: if > 0, periodically autosave the WOLF position.
- PROMETHEUS_MULTIPROC_DIR: enable Prometheus multiprocess /metrics when set.
- ADMIN_IP_ALLOWLIST: comma-separated IPs allowed to perform write operations.
- IDEMPOTENCY_TTL_S: seconds to cache Idempotency-Key results for /api/alerts/dispatch.


New endpoints:

- GET /api/version → { version, git_sha, build_time }
- GET /api/config → redacted configuration summary with ETag and caching.


Metrics:

- ghost_snapshot_asof: epoch of last /api/cockpit snapshot.


Grafana panel example (Value):

1. Query: `ghost_snapshot_asof`
2. Unit: Time → Date & time (from seconds)
3. Legend: Snapshot As Of


## 📊 Prometheus/Grafana: Example Panels

**Snapshot latency (p95, p50):**```promql

histogram_quantile(0.95, sum(rate(ghost_cockpit_snapshot_build_seconds_bucket[5m])) by (le))
histogram_quantile(0.5, sum(rate(ghost_cockpit_snapshot_build_seconds_bucket[5m])) by (le))

```text**Snapshot failures:**```promql

rate(ghost_cockpit_snapshot_failures_total[5m])

```text**App up:**```promql

ghost_up

```text

## 🧪 CI and Verification Gates

Workflows:

- `ci.yml` — Ruff + pytest + basic smoke; notifies Slack/Telegram on failure (if secrets


  set)

- `pr-verify.yml` — Runs the live verifier against a localhost server for PRs to main;


  notifies on failure

- `deploy.yml` — Deploys to production and, if `DEPLOY_BASE_URL` is set, runs


  post-deploy live verification; notifies on failure

- `staging.yml` — Deploys to staging and verifies against `STAGING_BASE_URL`; notifies


  on failure

- `verify-live-schedule.yml` — Nightly live verification against `VERIFY_BASE_URL`;


  notifies on failure

Secrets reference: see `docs/ci_secrets.md`.**Alert metrics:**

```promql

# Alerts sent by action/mode/result

sum by (action, mode, result) (rate(ghost_alerts_sent_total[5m]))

# Throttled (dedup) count

rate(ghost_alerts_throttled_total[5m])

# HOLD override state and mode one-hot gauges

ghost_alert_hold_override
ghost_alert_mode{mode="fixed"}
ghost_alert_mode{mode="band"}
ghost_alert_mode{mode="trailing"}

```text

- `/metrics` — Prometheus metrics (available when running `wolf_app:app` directly or via


  `main.py`).

- `/api/alerts` — Current WOLF BUY/SELL/HOLD signal preview.
- `/api/alerts/hold` — Toggle HOLD override: `{ "hold": true|false }`.
- `/api/alerts/dispatch` — Send alert card (dedupe/throttle + Telegram if configured).
- `/api/alerts/config` — GET current alert settings; POST to update at runtime.


Runtime alert config examples:

```bash

# View current config

curl <<<<<http://localhost:5000/api/alerts/config>>>>> | jq

# Switch to band mode with +/-2% and enable volatility gate

# Note: if GHOST_API_TOKEN is set, include Authorization header as shown

curl -X POST <<<<<http://localhost:5000/api/alerts/config>>>>> \
   -H 'Content-Type: application/json' \
   -H "Authorization: Bearer $GHOST_API_TOKEN" \
   -d '{"mode":"band","band_pct":0.02,"vol_gate":1,"vol_k":1.0}' | jq

# Switch to trailing mode

curl -X POST <<<<<http://localhost:5000/api/alerts/config>>>>> \
   -H 'Content-Type: application/json' \
   -H "Authorization: Bearer $GHOST_API_TOKEN" \
   -d '{"mode":"trailing","trail_sell_pct":0.06,"trail_buy_pct":0.04}' | jq

# Toggle market open/close scheduler at runtime

curl -X POST <<<<<http://localhost:5000/api/alerts/config>>>>> \
   -H 'Content-Type: application/json' \
   -H "Authorization: Bearer $GHOST_API_TOKEN" \
   -d '{"schedule_open_close": 1}' | jq
curl <<<<<http://localhost:5000/api/alerts/config>>>>> | jq   # shows schedule_open_close: true

```text

## 🎯 Examples

### Adding a Trading Position (simulated bank)

```bash

curl -X POST <<<<<http://localhost:5000/api/positions/add>>>>> \
   -H "Content-Type: application/json" \
   -d '{
      "symbol": "BTC",
      "qty": 0.1,
      "price_paid": 43250.0,
      "market": "crypto"
   }'

```text

### Setting a Financial Goal

```bash

curl -X POST <<<<<http://localhost:5000/api/goal_plan>>>>> \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Emergency Fund",
    "target": 10000.0,
    "deadline": "2024-12-31"
  }'

```text

### Checking Health

```bash

curl <<<<<http://localhost:5000/health>>>>>
curl <<<<<http://localhost:5000/api/secrets/health>>>>>

```text

## 🔧 Configuration

See `.env.example` for a complete reference. Notable:

- `SIM_MODE=0` (required)
- `ALPHAVANTAGE_API_KEY` or `ALPHA_VANTAGE_API_KEY`
- `COINGECKO_API_KEY`, `POLYGON_API_KEY`
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
- `ENABLE_REDIS`, `REDIS_URL`
- `GHOST_UI_URL`, `ALLOWED_ORIGINS`


### Fusion and AI flags (opt‑in)

These remain OFF by default to preserve existing behavior and tests. Enable any subset
as needed.

- News/Sentiment pipeline

  - `NEWS_SENTIMENT_ON=1` — Enable sentiment scoring and `news_signal` in the cockpit
  - `FINBERT_ON=1` — Use FinBERT for sentiment (if available); otherwise rule-based


    heuristics

  - `NEWS_LOOKBACK_MIN` (default: 60) — Lookback window in minutes for relevant news
  - `NEWS_DECAY_K` (default: 0.08) — Time-decay factor for aggregation

- Fusion of price + news into a final score and optional action override

  - `FUSE_DECISION_ON=1` — Allow the fused `final_score` to override BUY/SELL/HOLD


## 🧠 Ghost‑AI v1 (ChatGPT‑powered, WOLF‑only)

Ghost‑AI v1 adds a lightweight, safe advisory brain that reads the current WOLF snapshot
and returns a decision:

- Contract:


  `{ action: BUY|SELL|HOLD, confidence: 0-100, rationale, risks[], evidence[], checklist[], card }`

- Two ways to use it:
  - `POST /ai/decide` — Single step (LLM or deterministic fallback when AI is off)
  - `POST /ai/agent/run` — Tool‑using flow (the LLM can call `get_price`, `get_news`,


    `get_position`, and optionally `dispatch_alert`)

Setup (OpenAI‑compatible; no SDK required)

```bash

# Required for OpenAI usage

export OPENAI_API_KEY="<your-key>"

# Recommended defaults

export AI_PROVIDER=openai         # openai|ollama
export AGENT_MODEL=gpt-4o-mini    # preferred name; AI_MODEL remains an alias
export AGENTS_ENABLED=1           # preferred name; AI_ON remains an alias
export AI_TIMEOUT_S=10

# Optional: include the request context echo in responses

export AI_INCLUDE_CONTEXT=0

# Optional: automatically send the advisor card after /ai/agent/run

export AI_AGENT_AUTOSEND=0

```text

If you prefer running a local model via Ollama, set `AI_PROVIDER=ollama`,
`OLLAMA_BASE_URL=<<<<<http://127.0.0.1:11434`,>>>>> and an appropriate `AI_MODEL` (e.g.
`llama3.1:8b`).

Security

- If you set `GHOST_API_TOKEN`, writes to protected endpoints (including `/ai/decide`


  and `/ai/agent/run`) require the header `Authorization: Bearer $GHOST_API_TOKEN`.

- Never commit your real `OPENAI_API_KEY` or `.env` to the repo.


Try it

```bash

# Decision (requires Bearer only if GHOST_API_TOKEN is set)

curl -X POST <<<<<http://localhost:5000/ai/decide>>>>> \
   -H "Authorization: Bearer $GHOST_API_TOKEN"

# Agent (tool‑using)

curl -X POST <<<<<http://localhost:5000/ai/agent/run>>>>> \
   -H "Authorization: Bearer $GHOST_API_TOKEN"

```text

UI

- A compact panel is included in the cockpit (AI Decide button) that calls `/ai/decide`


  and shows `action / confidence / why`.

- If alerts are configured, `/ai/agent/run` can emit a Telegram‑ready card when


  `AI_AGENT_AUTOSEND=1`.

Implementation notes

- Calls use a simple `requests` HTTP client with small retry/backoff on 429/5xx.
- If `AI_ON=0` or the AI provider fails, `/ai/decide` returns a deterministic decision


  derived from the current rule‑based signal.

- The agent returns strict JSON and constructs a Telegram card if the model omits it.


### Macro Brain (opt‑in)

Advisory-only macro outlook that computes bull/base/bear scenarios using macro proxy
tickers (default: SMH, SOXX, QQQ), WOLF momentum vs previous close, and aggregated news
sentiment (if enabled).

Enable via env vars:

- `MACRO_BRAIN_ON=1` — Enable Macro Brain computation on each snapshot
- `MACRO_TICKERS="SMH,SOXX,QQQ"` — Comma-separated macro proxies to use
- `MACRO_LOOKBACK_DAYS=20` — Lookback window for proxy performance


When enabled, `/api/cockpit` includes an additional, non-breaking `macro` section:

```text

"macro": {
   "enabled": true,
   "confidence": 72,
   "scenarios": [
      {"name": "bull", "p": 0.51, "drivers": ["semis/tech momentum", "positive news"]},
      {"name": "base", "p": 0.28, "drivers": ["mean reversion", "range-bound"]},
      {"name": "bear", "p": 0.21, "drivers": ["risk-off", "mixed news"]}
   ],
   "summary": "Likely uptrend"
}

```text

Metrics:

- `ghost_macro_confidence{scenario}` — Confidence for the last advisory (0–100)
- `ghost_macro_refresh_total{result}` — Outcome counts for computations (ok,


  yfinance-missing, etc.)

Notes:

- If `yfinance` is unavailable, Macro Brain degrades gracefully and returns


  `{ "enabled": true, "error": "yfinance-missing" }`.

- With `MACRO_BRAIN_ON=0` (default), behavior is unchanged; no Telegram card formats are


  altered.

### HTTP, Security, and Tracing (new)

- `HTTP_POOL_ENABLED` (default: 1) — Enable HTTP connection pooling and retries for


  provider/news/webhook calls

- `HTTP_POOL_SIZE` (default: 10), `HTTP_POOL_RETRIES` (default: 2), `HTTP_TIMEOUT_S`


  (default: 8)

- `SECURE_HEADERS` (default: 1) — Add HSTS/CSP/Referrer-Policy/X-Content-Type-Options


  headers

- `HSTS_ON` (default: 1), `HSTS_MAX_AGE` (default: 15552000)
- `OTEL_ENABLED` (default: 0) — Enable OpenTelemetry tracing (Console exporter)
- `OTEL_SERVICE_NAME` (default: ghost-wolf)


### Backup and Restore

Use the utility to snapshot and restore state files:

- Backup: `./utils/backup_restore.sh backup`

- Restore: `./utils/backup_restore.sh restore backups/<STAMP>`

  - `FUSE_T_BUY` / `FUSE_T_SELL` — Thresholds for overriding to BUY/SELL (floats,


    symmetric by default)

  - `SENT_ALPHA` / `SENT_BETA` — Weights for price vs. news components in the fused


    score

- AI advisor and agent (advisory-only)

  - `AI_DECIDE_ON=1` — Enable `/ai/decide` endpoint (OpenAI/Ollama compatible)
  - `AI_AGENT_ON=1` — Enable `/ai/agent/run` tool-calling agent (advisory)
  - `AI_MIN_CONF` — Optional minimum confidence to consider AI advice actionable (no


    side effects unless you wire autosend)

  - Provider config: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OLLAMA_BASE_URL`, `AI_MODEL`


Metrics added when features are enabled

- `ghost_decision_final_score` (gauge) — Last fused decision score
- `ghost_why_now_count` (gauge) — Count of “Why now” reasons attached to the signal
- `ghost_llm_calls_total`, `ghost_llm_decisions_total` (counters),


  `ghost_llm_confidence` (gauge)

## � Security & Auth

Ghost uses a simple Bearer token for write operations. Set `GHOST_API_TOKEN` in the
environment to enable protection; clients must then include:

### Production examples: Docker Compose and Kubernetes

Docker Compose (using an `.env` file or environment variables):

```yaml

version: '3.8'
services:
   ghost:
      build: .
      image: ghost:latest
      ports:

         - "5000:5000"


      # Option A: load from .env

      env_file:

         - .env


      # Option B: reference host env vars

      environment:
         GHOST_API_TOKEN: ${GHOST_API_TOKEN}
         TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN:-}
         TELEGRAM_CHAT_ID: ${TELEGRAM_CHAT_ID:-}

```text

Example `.env` contents:

```env

GHOST_API_TOKEN=replace-with-strong-token
TELEGRAM_BOT_TOKEN=<optional>
TELEGRAM_CHAT_ID=<optional>

```text

Kubernetes Secret + Deployment:

```yaml

---
apiVersion: v1
kind: Secret
metadata:
   name: ghost-secrets
type: Opaque
stringData:
   GHOST_API_TOKEN: "replace-with-strong-token"
   TELEGRAM_BOT_TOKEN: "<optional>"
   TELEGRAM_CHAT_ID: "<optional>"
---
apiVersion: apps/v1
kind: Deployment
metadata:
   name: ghost
spec:
   replicas: 1
   selector:
      matchLabels: { app: ghost }
   template:
      metadata:
         labels: { app: ghost }
      spec:
         containers:

            - name: ghost


               image: ghost:latest
               ports:

                  - containerPort: 5000


               env:

                  - name: GHOST_API_TOKEN


                     valueFrom:
                        secretKeyRef:
                           name: ghost-secrets
                           key: GHOST_API_TOKEN

                  - name: TELEGRAM_BOT_TOKEN


                     valueFrom:
                        secretKeyRef:
                           name: ghost-secrets
                           key: TELEGRAM_BOT_TOKEN

                  - name: TELEGRAM_CHAT_ID


                     valueFrom:
                        secretKeyRef:
                           name: ghost-secrets
                           key: TELEGRAM_CHAT_ID

```text

````

     Kubernetes Service and Ingress examples:

     ```yaml

     ---
     apiVersion: v1
     kind: Service
     metadata:
        name: ghost
     spec:
        selector: { app: ghost }
        ports:

           - name: http


              port: 80
              targetPort: 5000
     ---
     apiVersion: networking.k8s.io/v1
     kind: Ingress
     metadata:
        name: ghost
        annotations:
           kubernetes.io/ingress.class: nginx
     spec:
        rules:

           - host: ghost.example.com


              http:
                 paths:

                    - path: /


                       pathType: Prefix
                       backend:
                          service:
                             name: ghost
                             port:
                                number: 80

     ```text

     Readiness/Liveness:

     - The app exposes `/ready` and `/live`. In K8s, you can configure probes in the Pod spec:


     ```yaml

     readinessProbe:
        httpGet: { path: /ready, port: 5000 }
        periodSeconds: 10
     livenessProbe:
        httpGet: { path: /live, port: 5000 }
        periodSeconds: 10

     ```text

````

```text

Authorization: Bearer <your-token>

```text

Protected endpoints (require Bearer when `GHOST_API_TOKEN` is set):

- `POST /api/position`
- `POST /api/alerts/hold`
- `POST /api/alerts/config`
- `POST /api/alerts/dispatch` (supports `?dry_run=1`)
- `POST /alerts/status`
- `POST /alerts/test`


Notes:

- OpenAPI docs (`/docs`) show a lock icon for protected routes. The scheme is optional;


  when `GHOST_API_TOKEN` is not set, routes are effectively open even if shown as
  secured.

- You can click “Authorize” in Swagger UI and paste your token to try protected calls


  from the browser.

- Correlation: the server accepts an optional `X-Request-ID` header and always returns


  one. The current request ID is also appended to Telegram cards as a final line, for
  example: `• Req: 6f1a4e1b...`.

Examples:

```bash

# Update position (protected)

curl -X POST <<<<<http://localhost:5000/api/position>>>>> \
   -H 'Content-Type: application/json' \
   -H "Authorization: Bearer $GHOST_API_TOKEN" \
   -d '{"qty": 10, "avg_cost": 25.5}' | jq

# Dispatch alert (protected)

curl -X POST "<<<<<http://localhost:5000/api/alerts/dispatch">>>>> \
   -H "Authorization: Bearer $GHOST_API_TOKEN" | jq

# Dry-run dispatch (no Telegram send; records metrics with result="dry-run")

curl -X POST "<<<<<http://localhost:5000/api/alerts/dispatch?dry_run=1">>>>> \
   -H "Authorization: Bearer $GHOST_API_TOKEN" | jq

# On-demand STATUS card (protected)

curl -X POST "<<<<<http://localhost:5000/alerts/status">>>>> \
   -H "Authorization: Bearer $GHOST_API_TOKEN"

```text

## �🐳 Docker

Build and run a slim multi-stage image:

```bash

docker build -t ghost:latest .
docker run --rm -p 5000:5000 --env-file .env ghost:latest

```text

The container respects `PORT` (default 5000) and exposes a healthcheck on `/health`.

### 24/7 with Docker Compose (app + Redis)

Quickly deploy Ghost with Redis-backed state and auto-restart:

1. Create an `.env` from the example:


   ```bash

   cp .env.example .env

   # fill in TELEGRAM_* and API keys as available

   ```text

1. Start the stack:


   ```bash

   docker compose up -d

   ```text

1. Verify:
   - Open `http://<server-ip>/health` → `{ "ok": true, ... }`
   - Visit `/` for the cockpit UI
   - Test Telegram: `curl -X POST <<<<<http://<server-ip>/alerts/test`>>>>>


The compose file maps `80:5000`, restarts the services unless stopped, and persists
Redis data in the `redis-data` volume. Set a domain and run a TLS proxy (Caddy/Traefik
or a Cloudflare Tunnel) for HTTPS.

Environment keys recommended for live mode reliability:

- `COINGECKO_API_KEY`
- `ALPHAVANTAGE_API_KEY` or `ALPHA_VANTAGE_API_KEY`
- `POLYGON_API_KEY` (optional)


Runtime SIM toggle (useful for demos):

```bash

curl <<<<<http://<server>/api/mode>>>>>              # {"mode":"live","sim_seed":42}
curl -X POST <<<<<http://<server>/api/mode>>>>> \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true}'                 # -> sim mode

```text

## ℹ️ Note on compatibility shim

The production runtime should target the WOLF-only core app `wolf_app:app` (as in the
Dockerfile CMD). For legacy tests and tooling, `main.py` re-exports the same ASGI app
and adds a minimal compatibility shim with SIM-only, in-memory endpoints to satisfy
older routes and contracts. The shim does not affect the WOLF logic.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading cryptocurrencies
and stocks involves substantial risk and may not be suitable for all investors. Always
do your own research and consider your financial situation before making investment
decisions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for
details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](<<<<<https://github.com/seancole713-source/GHOST/issue>>>>>s) page
2. Create a new issue with detailed information
3. Contact the development team


______________________________________________________________________

**Happy Trading! 👻**

______________________________________________________________________

## 📦 Release v0.1.0 (Initial Cockpit Snapshot)

Key changes:

- Unified `/api/cockpit` snapshot + SSE stream `/api/cockpit/stream`.
- Deterministic SIM price generator (toggle via `POST /api/mode`).
- Weighted average position merge fixed; stable entry pricing.
- Per-row PnL % normalized to 6 decimals (tests enforce tolerance ≤ 1e-4).
- Redis persistence optional (`ENABLE_REDIS=1`).
- Added `/api/providers/health` endpoint and lightweight in-process price cache.
- Test suite at cut: 14 passed, 2 skipped, 0 failed.


Tagging & push:

```bash

git pull --rebase
git tag v0.1.0
git push origin v0.1.0

```text

## 📈 Metrics & Monitoring

Ghost exposes Prometheus metrics (if `prometheus-client` is installed) at:

```text

/metrics

```text

Included metrics:

- `ghost_price_cache_hits{kind}` / `ghost_price_cache_misses{kind}` — local price cache


  efficiency.

- `ghost_price_provider_fetch_total{kind,provider,result}` — upstream provider fetch


  attempts (success/error).

- `ghost_cockpit_snapshot_build_seconds` (histogram) — latency distribution for building


  the cockpit snapshot (`/api/cockpit`). Use `histogram_quantile` to derive p90/p95.

- `ghost_cockpit_snapshot_failures_total` — count of snapshot build failures (e.g.,


  forced via `SNAP_FORCE_FAIL=1` in tests).

- `ghost_provider_fetch_seconds{provider}` (histogram) — latency for upstream provider


  price fetches (alpha, polygon, yf, sim, etc.).

Example scrape test:

```bash

curl -s <<<<<http://localhost:5000/metrics>>>>> | grep ghost_price_provider_fetch_total | head

```text

To deploy with Prometheus:

1. Add a job to your Prometheus config:


   ```yaml

   - job_name: ghost


      scrape_interval: 15s
      static_configs:

         - targets: ['ghost:5000']


   ```text

1. (Optional) If you run multiple Gunicorn/Uvicorn workers, enable Prometheus


   multiprocess mode:

   ```bash

   export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
   mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

   # then start your multi-worker server (ensure a fresh dir each restart)

   ```text

   Ghost does not yet aggregate multiprocess metrics automatically; this variable simply
   lets the Prometheus client switch collectors. Prefer a single process unless you need
   \>1 worker.

UI latency badge:

- The cockpit template now shows a small badge `tick: p50 / p95 ms` derived from recent


  client-side parse times of `/api/cockpit/stream` events (max 30 samples, rolling).

- Disable via `UI_LATENCY_BADGE=0` (environment variable) to hide it.


docker-compose Prometheus scrape (example to add to your Prometheus server, not inside
compose file itself):

```yaml

scrape_configs:

   - job_name: ghost


      static_configs:

         - targets: ['ghost:5000']


```text

1. (Optional) Add an alert for persistent provider errors (Polygon / Alpha Vantage rate


   limits):

   ```yaml

   - alert: GhostProviderErrorSpike


      expr: rate(ghost_price_provider_fetch_total{result="error"}[5m]) > 5
      for: 5m
      labels:
         severity: warning
      annotations:
         summary: "Ghost provider errors elevated"
         description: "Error fetch rate > 5 per minute for 5m"

   ```text

Validate after deploy:

```bash

curl -sS $PUBLIC_BASE_URL/health | jq
curl -sS $PUBLIC_BASE_URL/api/providers/health | jq
curl -sS $PUBLIC_BASE_URL/api/cockpit | jq '.snapshot_id, .portfolio.nav_total'

```text

## 🧭 Runbook (Operations Quick Links)

Endpoints:

- Health: `/health`, `/api/status`
- Metrics: `/metrics`
- Snapshot JSON: `/api/cockpit`
- Snapshot SSE: `/api/cockpit/stream`
- Alerts test: `POST /alerts/test` (verifies Telegram path)


Environment knobs (production):

```text

UI_LATENCY_BADGE=1
SNAP_FORCE_FAIL=0
SIM_MODE=0
ALPHAVANTAGE_API_KEY=xxxx
POLYGON_API_KEY=xxxx
COINGECKO_API_KEY=xxxx
TELEGRAM_BOT_TOKEN=xxxx
TELEGRAM_CHAT_ID=12345678

```text

Grafana / PromQL starters:

```text

rate(ghost_cockpit_snapshot_failures_total[5m])
histogram_quantile(0.5, sum(rate(ghost_cockpit_snapshot_build_seconds_bucket[5m])) by (le))
histogram_quantile(0.95, sum(rate(ghost_cockpit_snapshot_build_seconds_bucket[5m])) by (le))
sum by (provider)(rate(ghost_provider_fetch_seconds_sum[5m])) / sum
by(provider)(rate(ghost_provider_fetch_seconds_count[5m]))

```text

Alert suggestions:

```text

# P95 snapshot > 2s for 10m

histogram_quantile(0.95, sum(rate(ghost_cockpit_snapshot_build_seconds_bucket[5m])) by (le)) > 2

# Any snapshot failures sustained 5m

rate(ghost_cockpit_snapshot_failures_total[5m]) > 0

# Instance down (exporter / app outage)

ghost_up == 0

```text

Compose Prometheus scrape (external prometheus.yml):

```yaml

scrape_configs:

   - job_name: ghost


      scrape_interval: 15s
      static_configs:

         - targets: ['ghost:5000']


```text

______________________________________________________________________
