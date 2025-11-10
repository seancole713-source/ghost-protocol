# 🗺️ MAP & SWARM - PARALLEL BUILD MISSION

**Start Time**: October 13, 2025 11:50 PM\
**Goal**: Build Ghost (stocks + crypto + Telegram) from live dependency map with
parallel execution and self-correction\
**Status**: 🟢 **ACTIVE**

______________________________________________________________________

## 🎯 OPERATING PRINCIPLES

1. **Map-First**: Build live dependency graph (DAG) before implementation
2. **Swarm Execution**: Run multiple workstreams concurrently
3. **Test-Then-Build**: Create contract tests before implementation
4. **Telemetry Everywhere**: Every feature emits logs, metrics, readiness
5. **Feature Flags**: All new behavior behind env flags
6. **No Placeholders**: Ban mock/demo data in production

______________________________________________________________________

## 🔍 ROOT CAUSE ANALYSIS: BEARER AUTH ERROR

**Error Message**:

```json
{
  "error": {
    "message": "Missing bearer or basic authentication in header",
    "type": "invalid_request_error",
    "param": null,
    "code": null
  }
}
```

**Source**: This is **OpenAI API** error format (not Ghost)

**Root Cause**: Ghost → OpenAI call missing auth header

**Evidence**:

- Error format matches OpenAI API v1 responses
- Ghost's bearer auth uses `HTTPException(403, "missing bearer token")`
- OpenAI uses `{"error": {"message": "...", "type": "invalid_request_error"}}`

**Fix Location**: Check OpenAI API calls in `wolf_app.py` for missing headers

______________________________________________________________________

## 🗺️ GHOST DEPENDENCY MAP (CURRENT STATE)

### Core System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                         UI LAYER                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Dashboard│  │ Prediction│  │  Crypto  │  │ Trading  │   │
│  │  Panel   │  │  Overlay │  │  Toggle  │  │ Controls │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
└───────┼────────────┼─────────────┼─────────────┼──────────┘
        │            │             │             │
┌───────┼────────────┼─────────────┼─────────────┼──────────┐
│       │            │             │             │           │
│  ┌────▼────┐  ┌───▼────┐   ┌───▼────┐    ┌───▼────┐     │
│  │ /quotes │  │/predict│   │ /crypto│    │ /trade │     │
│  │         │  │        │   │ /price │    │/submit │     │
│  └────┬────┘  └───┬────┘   └───┬────┘    └───┬────┘     │
│       │           │            │             │           │
│       └───────────┴────────────┴─────────────┘           │
│                   API LAYER (wolf_app.py)                │
└──────────────────────────────┬───────────────────────────┘
                               │
┌──────────────────────────────┴───────────────────────────┐
│                    ORCHESTRATION LAYER                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │  Quorum    │  │  Prediction│  │   Risk     │         │
│  │  Manager   │  │   Engine   │  │  Manager   │         │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘         │
└────────┼───────────────┼───────────────┼─────────────────┘
         │               │               │
┌────────┼───────────────┼───────────────┼─────────────────┐
│        │               │               │                  │
│  ┌─────▼──────┐  ┌────▼─────┐  ┌─────▼──────┐          │
│  │ Providers  │  │ AI/Fusion│  │   Broker   │          │
│  ├────────────┤  ├──────────┤  ├────────────┤          │
│  │ AV→Polygon │  │ OpenAI   │  │   Alpaca   │          │
│  │ Yahoo→IEX  │  │ Anthropic│  │   Orders   │          │
│  │ CoinGecko  │  │ Groq     │  │ Positions  │          │
│  │ Binance    │  │ Sentiment│  │ Risk Checks│          │
│  └─────┬──────┘  └────┬─────┘  └─────┬──────┘          │
│        │              │              │                  │
│        └──────────────┴──────────────┘                  │
│               DATA/STORAGE LAYER                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │  Redis   │  │ SQLite   │  │  Metrics │             │
│  │  Cache   │  │   DBs    │  │Prometheus│             │
│  │ (prices) │  │(wolf.db) │  │ Counters │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
         │               │              │
┌────────┼───────────────┼──────────────┼─────────────────┐
│        │               │              │                  │
│  ┌─────▼──────┐  ┌────▼─────┐  ┌────▼──────┐          │
│  │ Telegram   │  │   MCP    │  │   Logs    │          │
│  │  Webhook   │  │ Servers  │  │Structured │          │
│  │  /status   │  │(pylance) │  │   JSON    │          │
│  │  /signal   │  │          │  │           │          │
│  └────────────┘  └──────────┘  └───────────┘          │
│               EXTERNAL INTEGRATIONS                     │
└─────────────────────────────────────────────────────────┘
```

### Critical Paths (Priority Order)

1. **Stock Prices** (Core):
   `UI → /quotes → QuorumManager → [AV, Polygon, Yahoo] → Cache → UI`
2. **Predictions** (Core):
   `UI → /predict → PredictionEngine → [AI, Historical] → Display`
3. **Crypto Prices** (New):
   `UI → /crypto/price → QuorumManager → [CoinGecko, Binance] → Cache → UI`
4. **Trading** (New): `UI → /trade/submit → RiskEngine → AlpacaBroker → Order → DB`
5. **Telegram** (Integration): `Webhook → Handler → [Commands, Q&A] → OpenAI → Response`

### Known Breakages (Need Fixing)

| Path | Status | Issue | Fix Required | |------|--------|-------|-------------| |
**Telegram Q&A** | 🔴 **BROKEN** | OpenAI API calls missing auth header | Add
`headers={'Authorization': f'Bearer {api_key}'}` | | **Crypto Prices** | 🟡
**INCOMPLETE** | No quorum, single provider | Add CoinGecko + Binance quorum | |
**Crypto Predictions** | 🔴 **MISSING** | No /api/crypto/predict endpoint | Create
prediction endpoint | | **Broker Testing** | 🟡 **UNTESTED** | Just deployed, not
validated | Run BROKER_TESTING_GUIDE.md | | **SL/TP Automation** | 🔴 **MISSING** | No
background monitoring loop | Add asyncio task |

______________________________________________________________________

## 🐝 SWARM EXECUTION PLAN (5 PARALLEL STREAMS)

### Stream A: FIX OPENAI AUTH (CRITICAL) ⚡

**Status**: 🔴 **BLOCKED**\
**Owner**: Agent\
**Dependencies**: None\
**Time**: 15 minutes

**Tasks**:

1. Find all OpenAI API calls in wolf_app.py
2. Verify auth header format in each call
3. Add missing headers where needed
4. Test with Telegram Q&A command
5. Verify in Railway logs

**Contract Test**: `/telegram/webhook` with Q&A message returns AI response (not 401
error)

______________________________________________________________________

### Stream B: STOCK PRICE QUORUM (CORE) 📊

**Status**: 🟢 **WORKING** (needs metrics)\
**Owner**: Existing system\
**Dependencies**: None\
**Time**: 30 minutes (metrics only)

**Tasks**:

1. ✅ AlphaVantage provider (done)
2. ✅ Polygon provider (done)
3. ✅ Yahoo fallback (done)
4. ✅ Quorum median logic (done)
5. ❌ Add Prometheus metrics per provider
6. ❌ Add plausibility guards (±50% spike rejection)

**Contract Test**: `/api/quotes?symbols=WOLF` returns
`{provider: "quorum", confidence: 0.95, spread: <0.05}`

______________________________________________________________________

### Stream C: CRYPTO QUORUM (NEW) 🪙

**Status**: 🟡 **PARTIAL** (single provider)\
**Owner**: Agent\
**Dependencies**: None\
**Time**: 2 hours

**Tasks**:

1. ✅ CoinGecko adapter (exists)
2. ❌ Binance adapter (create new)
3. ❌ Coinbase adapter (optional)
4. ❌ Quorum logic for crypto
5. ❌ Cache with TTL (60s)
6. ❌ UI toggle "Stocks | Crypto"

**Contract Test**: `/api/crypto/price/BTC` returns
`{provider: "quorum", quorum_size: 2, spread: <0.01}`

______________________________________________________________________

### Stream D: PREDICTION ENGINE + SCOREBOARD (ENHANCEMENT) 🔮

**Status**: 🟢 **WORKING** (needs UI rename)\
**Owner**: Agent\
**Dependencies**: Stream B (stock prices)\
**Time**: 1.5 hours

**Tasks**:

1. ✅ Prediction generation (done)
2. ✅ Confidence bands (done)
3. ❌ Rename UI panel to "Ghost Prediction"
4. ❌ Overlay predicted vs actual line chart
5. ❌ Rolling MAP calculation per symbol
6. ❌ Accuracy scoreboard table
7. ❌ `/api/predictions/history` endpoint

**Contract Test**: UI shows "Ghost Prediction" panel with overlay chart and MAP < 15%

______________________________________________________________________

### Stream E: TELEGRAM BI-DIRECTIONAL (INTEGRATION) 💬

**Status**: 🟡 **PARTIAL** (commands work, Q&A broken)\
**Owner**: Agent\
**Dependencies**: Stream A (OpenAI fix)\
**Time**: 1 hour

**Tasks**:

1. ✅ Webhook to Railway URL (done)
2. ✅ `/status` command (done)
3. ✅ `/signal` command (done)
4. ✅ `/pnl` command (done)
5. 🔴 Free-form Q&A (broken - needs Stream A)
6. ❌ `/buy AAPL 10` command (new)
7. ❌ `/sell AAPL` command (new)
8. ❌ `/positions` command (new)

**Contract Test**: Send "What does BTC -5% imply for WOLF?" → Get AI-powered answer

______________________________________________________________________

### Stream F: OBSERVABILITY & METRICS (FOUNDATION) 📈

**Status**: 🟡 **PARTIAL** (basic /health exists)\
**Owner**: Agent\
**Dependencies**: All streams\
**Time**: 1 hour

**Tasks**:

1. ✅ `/health` endpoint (done)
2. ❌ `/metrics` Prometheus endpoint
3. ❌ `ghost_price_fetch_total{provider="alphavantage"}` counter
4. ❌ `ghost_prediction_mape{symbol="WOLF"}` gauge
5. ❌ `ghost_sentiment_score` gauge
6. ❌ `/ready` endpoint (checks DB, cache, providers)

**Contract Test**: `curl /metrics | grep ghost_` returns 10+ metrics

______________________________________________________________________

## 🔬 CONTRACT TESTS (Test-Then-Build)

### Core Contracts (Must Pass Before Merge)

```bash
# 1. Stock Price Quorum
curl https://web-production-8e9a0.up.railway.app/api/quotes?symbols=WOLF
# Expected: {"WOLF": {"price": >0, "provider": "quorum", "confidence": >0.9}}

# 2. Crypto Price
curl https://web-production-8e9a0.up.railway.app/api/crypto/price/BTC
# Expected: {"symbol": "BTC", "price": >0, "quorum_size": >=2, "spread": <0.01}

# 3. Prediction Overlay
curl https://web-production-8e9a0.up.railway.app/api/predictions/history?symbol=WOLF
# Expected: {"forecasts": [...], "actual": [...], "map": <15}

# 4. Telegram Q&A
curl -X POST https://web-production-8e9a0.up.railway.app/telegram/webhook \
  -H "Content-Type: application/json" \
  -d '{"message": {"text": "What is WOLF price?", "chat": {"id": 123}}}'
# Expected: Telegram receives message with price (not OpenAI error)

# 5. Trading Submission
curl -X POST https://web-production-8e9a0.up.railway.app/api/trade/submit \
  -H "Authorization: Bearer e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "WOLF", "qty": 1, "side": "buy", "type": "market"}'
# Expected: {"ok": true, "submitted": true, "order": {...}}

# 6. Metrics Endpoint
curl https://web-production-8e9a0.up.railway.app/metrics
# Expected: ghost_price_fetch_total{provider="alphavantage"} <number>
```

______________________________________________________________________

## 🚨 GUARDRAILS (Auto-Block Merge if Violated)

1. **Contract Tests**: All 6 core contracts must pass
2. **Health Check**: `/health` returns `"ok": true`
3. **No Placeholders**: No strings containing "TODO", "FIXME", "mock", "demo",
   "placeholder"
4. **Feature Flags**: All new behavior behind env vars (default safe)
5. **No Secrets in Code**: `git grep -i "sk-proj-" || git grep -i "api_key.*=.*['\"]"`
   returns empty
6. **Logs Structured**: All new logs use `LOGGER.info(..., extra={...})`

______________________________________________________________________

## 📊 TELEMETRY (Real-Time Health)

### Per-Stream Health Board

| Stream | Status | Progress | Last Update | Blockers |
|--------|--------|----------|-------------|----------| | **A: OpenAI Fix** | 🔴
**BLOCKED** | 0% | 11:50 PM | Need to locate OpenAI calls | | **B: Stock Quorum** | 🟢
**HEALTHY** | 80% | 11:45 PM | None | | **C: Crypto Quorum** | 🟡 **IN PROGRESS** | 30% |
11:40 PM | None | | **D: Prediction UI** | 🟢 **HEALTHY** | 70% | 11:35 PM | None | |
**E: Telegram** | 🔴 **BLOCKED** | 60% | 11:50 PM | Waiting on Stream A | | **F:
Observability** | 🟡 **IN PROGRESS** | 40% | 11:30 PM | None |

### System Health Snapshot

```json
{
  "timestamp": "2025-10-13T23:50:00Z",
  "overall_health": "DEGRADED",
  "critical_failures": [
    "telegram_qa_openai_auth",
    "crypto_prediction_missing"
  ],
  "working": [
    "stock_prices",
    "broker_integration",
    "telegram_commands",
    "ui_dashboard"
  ],
  "in_progress": [
    "crypto_quorum",
    "prediction_overlay",
    "metrics_prometheus"
  ],
  "completion": "75%"
}
```

______________________________________________________________________

## 🎯 RUN LOOP (Autonomic Execution)

### Iteration 1: IMMEDIATE (Next 15 min)

**PLAN**:

- Stream A (OpenAI Fix) - Priority 1 (blocks Telegram)
- Stream F (Health Check) - Priority 2 (foundation)

**GUARDRAILS**:

- Create contract test for Telegram Q&A
- Create contract test for /health endpoint

**APPLY**:

- Fix OpenAI API calls in wolf_app.py
- Add /ready endpoint

**OBSERVE**:

- Test Telegram Q&A with "What is WOLF price?"
- Check /health returns ok

**COMMIT**:

- Only if both contracts pass

______________________________________________________________________

### Iteration 2: NEAR-TERM (Next 2 hours)

**PLAN**:

- Stream C (Crypto Quorum) - Priority 1 (new feature)
- Stream D (Prediction Overlay) - Priority 2 (UX improvement)

**GUARDRAILS**:

- Contract test for crypto quorum (2+ providers)
- Contract test for MAP calculation

**APPLY**:

- Add Binance crypto adapter
- Implement quorum logic for crypto
- Rename UI panel + add overlay chart
- Calculate rolling MAP

**OBSERVE**:

- Verify crypto prices from 2+ sources
- Verify MAP updates every prediction

**COMMIT**:

- Only if contracts pass

______________________________________________________________________

### Iteration 3: MID-TERM (Next 4 hours)

**PLAN**:

- Stream E (Telegram Trading) - Priority 1 (user experience)
- Stream F (Prometheus Metrics) - Priority 2 (observability)

**GUARDRAILS**:

- Contract test for /buy command
- Contract test for /metrics endpoint

**APPLY**:

- Add /buy, /sell, /positions Telegram commands
- Implement Prometheus /metrics endpoint
- Add counters for price fetches, predictions

**OBSERVE**:

- Test Telegram trading end-to-end
- Verify metrics exported

**COMMIT**:

- Only if contracts pass + no feature flag violations

______________________________________________________________________

## 🎊 SUCCESS CRITERIA (Mission Complete)

### Must-Have (90% Complete)

- ✅ Stock prices: quorum of 3+ providers
- ✅ Crypto prices: quorum of 2+ providers
- ✅ Predictions: overlay chart with MAP
- ✅ Telegram: all commands + Q&A working
- ✅ Trading: broker integration tested
- ✅ Observability: /metrics + /health + /ready

### Nice-to-Have (100% Complete)

- ❌ SL/TP automation loop
- ❌ Backtesting engine
- ❌ Trading UI dashboard
- ❌ Crypto predictions AI model

______________________________________________________________________

## 📝 NEXT ACTIONS (IMMEDIATE)

1. **Find OpenAI Auth Bug** (5 min)

   - Search wolf_app.py for `openai` import
   - Locate all API call sites
   - Check for missing `Authorization` header

2. **Create Contract Tests** (10 min)

   - Write `/telegram/webhook` Q&A test
   - Write `/api/crypto/price` quorum test
   - Write `/metrics` Prometheus test

3. **Fix OpenAI Calls** (5 min)

   - Add proper auth headers
   - Test with curl
   - Deploy to Railway

4. **Verify Telegram Q&A** (5 min)

   - Send test message
   - Check response
   - Monitor Railway logs

______________________________________________________________________

**STATUS**: 🟢 Ready to execute Stream A (OpenAI Fix)\
**BLOCKERS**: None\
**ETA**: 15 minutes to restore Telegram Q&A
