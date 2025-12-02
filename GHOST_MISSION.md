# GHOST PROTOCOL - CORE MISSION & SPECIFICATION
**Version:** 2.0  
**Date:** December 2, 2025  
**Status:** Production Spec

---

## 🎯 MISSION STATEMENT

Ghost Protocol is a **live signal generator + performance tracker + Telegram alert system** designed for **private use by one human owner**. Ghost's purpose is to identify high-probability trading opportunities and alert the owner when to act—**NEVER to execute trades automatically**.

---

## 🚫 PERMANENT CONSTRAINTS (NEVER VIOLATE)

### 1. No Execution Authority
- **NO BROKER INTEGRATION** for order placement
- **NO AUTO-BUY** or **AUTO-SELL** functionality
- **NO ORDER ROUTING** to any trading platform
- **NO P&L TRACKING FROM BROKER** accounts
- Ghost observes, predicts, and alerts—the human decides and executes

### 2. Single Owner Model
- Designed for **one human owner only** (not multi-tenant)
- Private deployment (Railway, self-hosted, etc.)
- No user authentication system needed
- Configuration via environment variables

### 3. Live Data Only
- **NO SIMULATION MODE** for production predictions
- Real market data required (Polygon, CoinGecko, Binance, etc.)
- Cache allowed (5-min TTL) but no stale fallbacks >1 hour
- Health checks must validate data freshness

---

## 📋 CORE RESPONSIBILITIES

### 1. **Signal Generation**
Ghost generates predictions for crypto and stocks with:
- **Direction:** UP / DOWN / FLAT
- **Confidence:** 0-100% (model certainty)
- **Expected Move:** % price change estimate
- **Timeframe:** 24h, 48h, 7d horizons
- **Reasoning:** Multi-signal rationale (momentum, volume, sentiment, etc.)

**Quality Standard:**
- Target: 85%+ accuracy on high-confidence signals (>55%)
- Speed: Crypto <500ms, Stocks <2000ms
- Coverage: 50+ symbols (crypto + stocks)

### 2. **Performance Tracking**
Ghost tracks its own accuracy and theoretical performance:
- **Accuracy Stats:** Win rate, avg error, best/worst symbols
- **Model-Implied Performance:** Theoretical % returns based on predictions
- **Goal Progress:** % return vs. dollar targets (dual tracking)
- **Historical Analysis:** 24h, 7d, 30d, all-time periods

**Metrics:**
- `model_edge_pct = win_rate * avg_winner_move`
- Track per period: daily, weekly, monthly, yearly
- Compare realized vs. target % returns

### 3. **Telegram Alerts**
Ghost sends actionable alerts via Telegram:

**Alert Types:**
1. **BUY CANDIDATE:** High upside opportunity detected
   - Confidence >55%, Score >80
   - Clear entry rationale
   - Expected move % and timeframe

2. **EXIT / DANGER:** Downside risk if owner holds position
   - Prediction shows DOWN movement
   - Position at risk based on recent prediction
   - Suggest EXIT or REDUCE exposure

3. **DAILY REPORTS:** Morning (7am) + Evening (8pm)
   - Top movers summary
   - Ghost Score health
   - Goal progress (% and $)

4. **WEEKLY ACCURACY:** Performance review
   - Win rate
   - Top performers
   - Model edge %

**Alert Rules:**
- Cooldown: 4 hours per symbol (no spam)
- Max rate: 5 instant alerts per hour
- Markdown formatting (mobile-friendly)
- Emoji indicators (🔥⭐✨ for urgency)

---

## 🔧 TECHNICAL ARCHITECTURE

### Inputs
1. **Market Data:**
   - Crypto: CoinGecko, Binance, Coinbase (quorum voting)
   - Stocks: Polygon (paid), yfinance, Yahoo HTTP, AlphaVantage
   - Forex/Indices: Optional (DXY, VIX, etc.)

2. **Configuration:**
   - VIP coins list (high-priority tracking)
   - Watchlist (custom symbols)
   - Confidence threshold (MIN_ALERT_CONFIDENCE=0.55)
   - Instant alert threshold (INSTANT_ALERT_THRESHOLD=80)
   - Goals: daily/weekly/monthly/yearly ($ and % targets)

3. **Historical Data:**
   - Predictions database (SQLite + Postgres)
   - Outcomes tracking (48h reconciliation)
   - Accuracy statistics

### Processing
1. **Feature Extraction:**
   - Technical: RSI, MACD, Bollinger Bands, momentum
   - Volume: Relative volume, volume divergence
   - Sentiment: News, social signals (Reddit, Twitter)
   - World Context: Macro indicators (DXY, VIX, SPY)

2. **Ensemble Prediction:**
   - Multiple signal sources (5-pillar system)
   - Confidence weighting
   - Direction voting
   - Expected move aggregation

3. **Opportunity Scoring:**
   - Multi-factor score (0-100)
   - Confidence × Expected Move × Signal Count
   - Risk-adjusted (volatility penalty)

4. **Accuracy Reconciliation:**
   - 48-hour outcome check
   - Direction correctness
   - Price error calculation
   - Statistics update

### Outputs
1. **API Responses:**
   - `/api/predict/run` - Generate prediction
   - `/api/v3/predictions/latest` - Cached predictions
   - `/api/v3/vip/snapshot` - VIP coins status
   - `/api/v3/cockpit/overview` - Dashboard data
   - `/api/v3/alerts/status` - Alert system health
   - `/api/v3/goals/snapshot` - Goal progress ($ + %)

2. **UI Dashboard (`/cockpit`):**
   - Top Movers (Stocks/Crypto)
   - VIP Coins Tracker
   - Ghost Forecast (24h/7d/14d)
   - News Feed
   - Prediction Accuracy Chart
   - Watchlist Grid
   - Ghost Health Score

3. **Telegram Messages:**
   - Formatted alerts (Markdown)
   - Visual indicators (emoji)
   - Actionable recommendations
   - Performance summaries

---

## 📊 GOALS SYSTEM V2 (% + DOLLARS)

### Data Model
```sql
CREATE TABLE goals (
    id INTEGER PRIMARY KEY,
    period TEXT,                -- 'daily', 'weekly', 'monthly', 'yearly'
    target_amount REAL,         -- Dollar goal (legacy)
    target_pct REAL,            -- % return goal (NEW)
    current_amount REAL,        -- Realized dollars
    realized_pct REAL,          -- Realized % (from trades)
    model_edge_pct REAL,        -- Ghost's theoretical edge
    start_date TEXT,
    end_date TEXT,
    status TEXT
);
```

### Metrics
1. **Target % Goal:** User-defined (e.g., daily=0.5%, weekly=2%, monthly=8%)
2. **Realized %:** Actual returns from owner's trades (manual input)
3. **Model Edge %:** Ghost's theoretical performance
   - Formula: `win_rate * avg_winner_move`
   - Example: 75% win rate × 3% avg move = 2.25% edge

### API Extensions
**POST `/api/v3/goals/set`**
```json
{
  "period": "daily",
  "target_amount": 500,        // Optional
  "target_pct": 0.5            // Optional
}
```

**GET `/api/v3/goals/snapshot`**
```json
{
  "goals_v2": {
    "daily": {
      "target": 500,
      "target_pct": 0.5,
      "current": 0,
      "realized_pct": 0.0,
      "model_edge_pct": 2.25,    // Ghost's theoretical edge
      "progress_pct": 0,          // $ progress
      "progress_vs_pct_goal": 0   // % progress
    },
    ...
  }
}
```

---

## 🎯 TELEGRAM ALERT SPEC

### BUY CANDIDATE Alert
```markdown
🔥 **BUY CANDIDATE: BTC**

**Direction:** ⬆️ UP
**Confidence:** 78%
**Expected Move:** +4.2%
**Timeframe:** 48h
**Current Price:** $86,487

**Rationale:**
✅ Strong momentum (RSI: 62)
✅ Volume surge (+45%)
✅ Positive sentiment (Reddit: 82%)

**Ghost Score:** 85/100
🎯 **ACTION:** Consider entry for 2-day upside
```

### EXIT / DANGER Alert
```markdown
⚠️ **DANGER: ETH Position Risk**

**Direction:** ⬇️ DOWN
**Confidence:** 72%
**Expected Move:** -3.8%
**Timeframe:** 24h
**Current Price:** $2,797

**Rationale:**
❌ Momentum weakening (RSI: 38)
❌ Volume drop (-32%)
❌ Negative divergence

**Ghost Score:** 68/100
🚨 **ACTION:** Consider EXIT if you hold ETH
```

### Daily Report (Evening)
```markdown
🌙 **Ghost Evening Report**
📅 December 2, 2025 - 8:00pm

**Ghost Health:** 92/100 (A)

**Top Movers Today:**
1. 🔥 XRP: +8.5% (Score: 88)
2. ⭐ SOL: +6.2% (Score: 82)
3. ✨ AAPL: +3.1% (Score: 75)

**Goal Progress:**
Daily: 42% ($210/$500)
Model Edge: +2.8%

**Predictions:** 12 signals sent
**Accuracy (7d):** 76.5%

Sleep well, tomorrow we hunt! 🎯
```

---

## 🔐 SECURITY & CONFIGURATION

### Required Environment Variables
```bash
# API Keys
POLYGON_API_KEY=<paid_tier_key>
OPENAI_API_KEY=<for_sentiment>
TELEGRAM_BOT_TOKEN=<bot_token>
TELEGRAM_CHAT_ID=<owner_chat_id>

# Configuration
MIN_ALERT_CONFIDENCE=0.55
INSTANT_ALERT_THRESHOLD=80
VIP_COINS=BTC,ETH,SOL,XRP,BNB

# Database
DATABASE_URL=postgresql://...
REDIS_URL=redis://...
```

### Deployment
- **Platform:** Railway, AWS, self-hosted
- **Health Check:** `/health` endpoint
- **Logs:** Structured JSON logs (trace_id, component)
- **Monitoring:** Ghost Score, provider health, alert delivery

---

## 📈 SUCCESS METRICS

### Performance Targets
- **Accuracy:** >85% on high-confidence signals
- **Speed:** Crypto <500ms, Stocks <2s
- **Uptime:** >99.5%
- **Model Edge:** >1% daily theoretical edge

### User Experience
- **Alert Latency:** <30s from signal detection
- **UI Responsiveness:** <200ms API responses
- **Zero Trade Execution:** NEVER place orders

### Data Quality
- **Provider Redundancy:** 3+ sources per asset
- **Cache Hit Rate:** >80%
- **Stale Data:** <1% of requests

---

## 🚀 ROADMAP (FUTURE ENHANCEMENTS)

### Phase 1: Core Stability (COMPLETE ✅)
- ✅ Multi-provider data quorum
- ✅ Prediction engine (48h horizon)
- ✅ Accuracy tracking (48h reconciliation)
- ✅ UI Dashboard (Cockpit V3)
- ✅ Alert system infrastructure
- ✅ Goals tracking ($ + %)

### Phase 2: Intelligence Upgrade (IN PROGRESS)
- ⚠️ LSTM model training (22 TODOs)
- ⚠️ Ensemble predictions (multi-model)
- ⚠️ Reddit sentiment integration
- ⚠️ Twitter sentiment API
- ⚠️ Earnings calendar integration

### Phase 3: Advanced Analytics (PLANNED)
- ❌ Backtesting engine (historical validation)
- ❌ Walk-forward analysis
- ❌ Monte Carlo simulation
- ❌ Strategy optimization
- ❌ Risk metrics (Sharpe, Sortino, max drawdown)

### Phase 4: User Experience (PLANNED)
- ❌ Mobile app (React Native)
- ❌ Custom alert templates
- ❌ Voice notifications (Telegram voice)
- ❌ Chart overlays (TradingView integration)

---

## 🔬 TESTING & VALIDATION

### Unit Tests
- Prediction accuracy calculation
- Goal progress computation
- Model edge formula
- Alert trigger logic

### Integration Tests
- Provider fallback chain
- Database dual-write (Postgres + SQLite)
- Telegram delivery
- API endpoint responses

### Performance Tests
- Load: 100 predictions/minute
- Latency: p95 <500ms (crypto), <2s (stocks)
- Memory: <2GB RSS
- Database: <100ms query time

---

## 📚 REFERENCE DOCUMENTS

- `FIX_VALIDATION_REPORT.md` - Recent fixes (VIP, Cockpit, Alerts)
- `ALPACA_INTEGRATION_COMPLETE.md` - Broker integration (DISABLED)
- `COCKPIT_V3_QUICK_START.md` - UI dashboard guide
- `TELEGRAM_HUNTER.md` - Alert system docs
- `core/goals_tracker.py` - Goals implementation
- `core/prediction_tracker.py` - Accuracy tracking

---

## ⚖️ LEGAL & ETHICS

### Disclaimer
Ghost Protocol is for **informational purposes only**. It provides predictions and alerts, NOT financial advice. The owner is solely responsible for all trading decisions and outcomes.

### Data Usage
- Market data: Licensed from providers (Polygon paid tier)
- User data: Single owner, no multi-tenant storage
- Privacy: No data sharing, local deployment

### Risk Warning
Trading involves substantial risk. Ghost's predictions are probabilistic, not guaranteed. Past performance (model edge) does not predict future results.

---

**END OF SPECIFICATION**

*This document defines Ghost Protocol's mission, constraints, and technical architecture. Any feature additions must align with these principles. Ghost observes, predicts, and alerts—humans execute.*
