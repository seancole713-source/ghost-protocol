# 🧬 GHOST PROTOCOL - ANATOMICAL BODY MAP

> *"If this trading system were a human body, where would each organ be?"*

---

## 🧠 THE BRAIN — Prediction Intelligence

**Function:** Makes trading predictions using machine learning (decides UP/DOWN/FLAT)

### Primary Files:
| File | Lines | Purpose |
|------|-------|---------|
| `core/ensemble_predictor.py` | 966 | **The Cerebral Cortex** - XGBoost ML model + Fear&Greed integration |
| `core/pattern_enhanced_predictor.py` | ~300 | Pattern recognition overlay |
| `core/cascading_predictor.py` | 742 | Multi-stage prediction lifecycle (48h→24h→6h) |
| `core/multi_horizon_forecaster.py` | ~400 | Different timeframe predictions |

### Key Classes & Functions:
```python
class XGBoostModel:           # The actual ML model (trained on historical data)
class EnsemblePredictor:      # Orchestrates XGBoost + market regime signals
def get_ensemble_predictor()  # Singleton accessor
def predict()                 # Main prediction method - returns direction, confidence
def get_fear_greed_signal()   # Contrarian market sentiment integration
def get_btc_trend()           # BTC correlation for altcoins
```

### Data Flow:
```
Technical Features → XGBoost Model → Raw Prediction
                          ↓
              Fear&Greed Index → Market Regime Adjustment
                          ↓
              Calibrated Prediction (UP/DOWN/FLAT + confidence%)
```

---

## ❤️ THE HEART — Main Orchestrator/Pump

**Function:** Central application that keeps everything alive and pumping

### Primary Files:
| File | Lines | Purpose |
|------|-------|---------|
| `wolf_app.py` | **37,612** | **The Heart Itself** - FastAPI main app, ALL endpoints, startup/shutdown |
| `core/watchlist_prediction_scheduler.py` | 387 | Market open/close scheduling |
| `core/cascade_scheduler.py` | 244 | Background cascade monitoring |
| `core/beast_scheduler.py` | ~300 | Daily TOP 10 scheduling |
| `core/auto_prediction_loop.py` | ~200 | Continuous prediction cycles |

### Key Functions:
```python
APP = FastAPI()               # The heart chamber itself
async def _on_startup()       # Heart's first beat (initialization)
def run_prediction()          # Main prediction trigger (heartbeat)
def run_single_prediction()   # Single symbol prediction pump
start_cascade_scheduler()     # Schedule background updates
start_auto_prediction_loop()  # Continuous monitoring
```

### Heartbeat Cycle:
```
Startup → Initialize All Systems → Start Schedulers
                    ↓
    Market Open → Generate Predictions → Send Alerts
                    ↓
    Continuous Loop → Monitor → Update → Repeat
```

---

## 👀 THE EYES — Price Data Inputs

**Function:** Sees current market prices from multiple sources

### Primary Files:
| File | Lines | Purpose |
|------|-------|---------|
| `core/providers/turbo_provider.py` | 767 | **Primary Vision** - Fast-fail price fetching with timeouts |
| `core/crypto/crypto_providers.py` | 1,100 | Crypto price sources (Binance, CoinGecko, Coinbase) |
| `core/providers/stock_providers.py` | ~500 | Stock price sources (yfinance, Yahoo HTTP, AlphaVantage) |
| `core/polygon_integration.py` | ~400 | Polygon.io stock data |
| `core/coinbase_provider.py` | ~200 | Coinbase crypto prices |

### Key Classes & Functions:
```python
class TurboProvider:          # Fast-fail wrapper with health monitoring
def turbo_stock_price()       # Get stock price (2s timeout)
def turbo_crypto_price()      # Get crypto price (2s timeout)
def get_crypto_price_quorum() # Multi-source consensus
def validate_crypto_price()   # Sanity check (BTC > $10k, etc.)
```

### Vision Provider Chain:
```
STOCKS:  yfinance → Yahoo HTTP → AlphaVantage → Polygon → Cache
CRYPTO:  Binance → CoinGecko → Coinbase → Cache
                    ↓
          Fastest valid response wins (2s timeout each)
```

---

## 👂 THE EARS — External Signals

**Function:** Listens to market sentiment, news, and external data

### Primary Files:
| File | Lines | Purpose |
|------|-------|---------|
| `core/ensemble_predictor.py:33-106` | 73 | **Fear & Greed Index** - Market sentiment (alternative.me API) |
| `core/news_sentiment.py` | 224 | News article sentiment scoring |
| `core/santiment_signals.py` | 265 | Santiment social data integration |
| `core/data_collector.py` | 608 | Multi-source data aggregation |
| `core/social_sentiment.py` | ~200 | Twitter/Reddit sentiment |

### Key Classes & Functions:
```python
def get_fear_greed_index()    # 0-100 market fear/greed
def get_fear_greed_signal()   # Contrarian trading signal
class SantimentProvider:      # Social media whale tracking
def fetch_news_sentiment()    # News article scoring
class DataCollector:          # Unified data hub
```

### External Signal Sources:
```
Fear & Greed API (alternative.me) → Market Regime Context
Alpha Vantage NEWS_SENTIMENT    → News Impact Score
Santiment GraphQL               → Social Whale Activity
Reddit/Twitter APIs             → Social Buzz Score
                    ↓
        Combined Sentiment Signal (-1.0 to +1.0)
```

---

## 🧠💾 THE MEMORY — Data Storage

**Function:** Stores predictions, outcomes, and learning data

### Primary Files:
| File | Lines | Purpose |
|------|-------|---------|
| `core/prediction_store.py` | 1,676 | **Long-term Memory** - PostgreSQL/SQLite abstraction |
| `core/paper_tracker.py` | 656 | Paper trade tracking (what would have happened) |
| `core/accuracy_tracker.py` | ~400 | Performance memory (learning from past) |
| `core/trade_journal.py` | ~300 | Trade history storage |

### Key Classes & Functions:
```python
class PredictionStore:        # Unified storage abstraction
def save_prediction()         # Store new prediction
def get_prediction_history()  # Recall past predictions
def get_prediction_points()   # Get data for charts
class PaperTracker:           # Track hypothetical P&L
def validate_price()          # Memory guard (reject corrupt data)
```

### Storage Architecture:
```
PostgreSQL (Production) ←→ prediction_store.py ←→ SQLite (Local Dev)
         ↓
   Tables: predictions, prediction_cascades, paper_trades,
           prediction_outcomes, accuracy_metrics
```

---

## 🔔 THE NERVOUS SYSTEM — Alerts & Notifications

**Function:** Sends alerts when important events happen

### Primary Files:
| File | Lines | Purpose |
|------|-------|---------|
| `core/telegram_hunter.py` | 711 | **Primary Nerve** - Telegram instant alerts |
| `core/ghost_notifications.py` | 1,588 | Scheduled TOP 10 reports (8AM daily) |
| `core/alert_manager.py` | 120 | Circuit breaker & risk alerts |
| `core/telegram_alerts.py` | ~300 | General alert queue |
| `core/watchlist_telegram_alerts.py` | ~200 | Watchlist-specific alerts |

### Key Classes & Functions:
```python
def send_telegram_message()   # Core alert sender
def send_hunter_alert()       # High-score opportunity alert
def send_daily_report()       # Morning/evening summaries
async def alert_processor_loop() # Background alert monitoring
class WatchlistPredictionScheduler: # Alert scheduling
```

### Alert Types:
```
Instant Alerts (score ≥ 80):
  🔥 High-confidence opportunity detected

Scheduled Reports:
  📊 8 AM: Daily TOP 10 (5 stocks + 5 crypto)
  📈 12 PM, 4 PM, 8 PM: Updates if >3% moves

System Alerts:
  🚨 Circuit breaker active
  ⚠️  Risk kill switch triggered
```

---

## 🖐️ THE HANDS — Trading Actions

**Function:** Executes trades (paper or live)

### Primary Files:
| File | Lines | Purpose |
|------|-------|---------|
| `core/alpaca_broker.py` | 461 | **Right Hand** - Alpaca broker integration |
| `core/paper_tracker.py` | 656 | **Left Hand** - Paper trading simulation |
| `core/order_manager.py` | ~400 | Order lifecycle management |
| `core/execution_risk.py` | ~450 | Pre-trade risk checks |
| `core/smart_router.py` | ~300 | Order routing optimization |

### Key Classes & Functions:
```python
class AlpacaBroker:           # Live/paper trading via Alpaca API
def place_order()             # Submit buy/sell order
class OrderManager:           # Track order states
class PaperTracker:           # Simulate trades without risk
def check_execution_risk()    # Pre-trade validation
```

### Trading Flow:
```
Prediction Generated → Risk Check → Order Created
                           ↓
        Paper Mode:  PaperTracker.log_trade()
        Live Mode:   AlpacaBroker.place_order()
                           ↓
            Order Tracking → Fill/Cancel → Journal Entry
```

---

## 👄 THE MOUTH — API Output

**Function:** Speaks to the outside world (UI, webhooks, integrations)

### Primary Files:
| File | Lines | Purpose |
|------|-------|---------|
| `wolf_app.py` | 37,612 | **Primary Vocal Cords** - All FastAPI endpoints |
| `routes/news_routes.py` | ~200 | News feed endpoints |
| `api/cockpit_v3_live_endpoints.py` | ~800 | Cockpit UI endpoints |

### Key Endpoints:
```python
@APP.get("/health")           # Am I alive?
@APP.get("/api/predict/run")  # Generate prediction
@APP.get("/api/v3/predictions/latest") # Get recent predictions
@APP.get("/api/v3/watchlist") # Get watchlist
@APP.post("/telegram/webhook") # Receive Telegram commands
@APP.get("/api/cockpit/stream") # SSE real-time stream
```

### Endpoint Categories:
```
Health & Status:  /health, /api/status, /debug/routes
Predictions:      /api/predict/run, /api/v3/predictions/*
Watchlist:        /api/v3/watchlist, /api/watchlist/*
Market Data:      /api/news/*, /api/movers/*
Trading:          /api/trade/*, /api/portfolio/*
Cockpit UI:       /api/cockpit/*, /cockpit (HTML)
Webhooks:         /telegram/webhook, /api/webhook/*
```

---

## 🩸 THE BLOOD — Data Flow Between Components

**Function:** Carries information between all organs

### Data Flow Diagram:
```
                    ┌─────────────────────────────────────────────────────┐
                    │                    THE HEART                        │
                    │                   (wolf_app.py)                      │
                    │   Startup → Schedulers → Request Handling → Output  │
                    └──────────────────────┬──────────────────────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────────┐
        │                                  │                              │
        ▼                                  ▼                              ▼
┌───────────────┐                 ┌───────────────┐               ┌───────────────┐
│   THE EYES    │                 │   THE EARS    │               │   THE MOUTH   │
│ (Price Data)  │                 │  (Sentiment)  │               │    (API)      │
│               │                 │               │               │               │
│ turbo_provider│                 │ fear_greed    │               │ FastAPI       │
│ crypto_provs  │                 │ news_sentiment│               │ endpoints     │
│ stock_provs   │                 │ santiment     │               │ webhooks      │
└───────┬───────┘                 └───────┬───────┘               └───────┬───────┘
        │                                  │                              │
        └──────────────────────────────────┼──────────────────────────────┘
                                           │
                                           ▼
                              ┌─────────────────────────┐
                              │        THE BRAIN        │
                              │  (ensemble_predictor)   │
                              │                         │
                              │   Features + Sentiment  │
                              │          ↓              │
                              │   XGBoost Prediction    │
                              │          ↓              │
                              │   Direction + Confidence│
                              └───────────┬─────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
              ▼                           ▼                           ▼
    ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
    │    THE MEMORY   │         │ THE NERVOUS SYS │         │    THE HANDS    │
    │ (prediction_    │         │ (telegram_      │         │ (alpaca_broker) │
    │     store)      │         │     hunter)     │         │ (paper_tracker) │
    │                 │         │                 │         │                 │
    │ PostgreSQL/     │         │ Telegram alerts │         │ Paper trades    │
    │ SQLite storage  │         │ Daily reports   │         │ Live orders     │
    └─────────────────┘         └─────────────────┘         └─────────────────┘
```

### Blood Flow Sequence:
```
1. EYES see price: turbo_provider → {"BTC": $100,000, "WOLF": $3.50}

2. EARS hear sentiment: fear_greed → 25 (Extreme Fear = BUY signal)

3. BRAIN processes:
   - Technical features (RSI, MACD, volume)
   - Sentiment features (fear/greed, news)
   - XGBoost model prediction
   - Market regime adjustment
   → {"direction": "UP", "confidence": 0.72}

4. MEMORY stores:
   prediction_store.save_prediction() → PostgreSQL

5. NERVOUS SYSTEM alerts:
   telegram_hunter.send_alert() → "🔥 BTC UP 72% confidence"

6. HANDS act:
   paper_tracker.log_trade() → Simulated $1000 BTC buy

7. MOUTH reports:
   /api/v3/predictions/latest → JSON to UI
```

---

## 📊 ORGAN HEALTH STATUS

| Organ | Primary File | Status | Lines of Code |
|-------|-------------|--------|---------------|
| 🧠 Brain | `ensemble_predictor.py` | ✅ Healthy | 966 |
| ❤️ Heart | `wolf_app.py` | ✅ Healthy | 37,612 |
| 👀 Eyes | `turbo_provider.py` | ✅ Healthy | 767 |
| 👂 Ears | `fear_greed + news_sentiment` | ✅ Healthy | ~500 |
| 💾 Memory | `prediction_store.py` | ✅ Healthy | 1,676 |
| 🔔 Nervous System | `telegram_hunter.py` | ✅ Healthy | 711 |
| 🖐️ Hands | `alpaca_broker.py` | ⚠️ Paper Only | 461 |
| 👄 Mouth | `wolf_app.py` (endpoints) | ✅ Healthy | (included in Heart) |

---

## 🔬 DETAILED FILE INVENTORY

### Brain Files (Prediction Logic)
```
core/ensemble_predictor.py         966 lines  - XGBoost + Fear&Greed
core/pattern_enhanced_predictor.py ~300 lines - Pattern overlay
core/cascading_predictor.py        742 lines  - 48h→24h→6h lifecycle
core/multi_horizon_forecaster.py   ~400 lines - Multi-timeframe
core/data_enhanced_predictor.py    ~350 lines - Data-driven signals
models/ensemble/                              - Trained model files (.pkl)
```

### Heart Files (Orchestration)
```
wolf_app.py                        37,612 lines - THE MAIN APP
core/watchlist_prediction_scheduler.py 387 lines - Market scheduling
core/cascade_scheduler.py              244 lines - Background monitoring
core/beast_scheduler.py                ~300 lines - Daily TOP 10
core/auto_prediction_loop.py           ~200 lines - Continuous loop
core/cron_scheduler.py                 ~150 lines - Cron-style tasks
```

### Eye Files (Price Data)
```
core/providers/turbo_provider.py   767 lines  - Fast-fail wrapper
core/crypto/crypto_providers.py    1,100 lines - Crypto sources
core/providers/stock_providers.py  ~500 lines - Stock sources
core/polygon_integration.py        ~400 lines - Polygon.io
core/coinbase_provider.py          ~200 lines - Coinbase
core/providers/yahoo_finance.py    ~300 lines - Yahoo backup
```

### Ear Files (External Signals)
```
core/ensemble_predictor.py (lines 33-106) - Fear & Greed
core/news_sentiment.py            224 lines - News scoring
core/santiment_signals.py         265 lines - Social data
core/data_collector.py            608 lines - Multi-source hub
core/social_sentiment.py          ~200 lines - Twitter/Reddit
core/market_mood.py               ~150 lines - Market mood indicator
```

### Memory Files (Storage)
```
core/prediction_store.py          1,676 lines - Main storage abstraction
core/paper_tracker.py             656 lines   - Paper trade memory
core/accuracy_tracker.py          ~400 lines  - Performance tracking
core/trade_journal.py             ~300 lines  - Trade history
core/portfolio_persistence.py     ~250 lines  - Portfolio state
```

### Nervous System Files (Alerts)
```
core/telegram_hunter.py           711 lines   - Instant alerts
core/ghost_notifications.py       1,588 lines - Scheduled reports
core/alert_manager.py             120 lines   - System alerts
core/telegram_alerts.py           ~300 lines  - Alert queue
core/watchlist_telegram_alerts.py ~200 lines  - Watchlist alerts
```

### Hand Files (Trading)
```
core/alpaca_broker.py             461 lines - Broker integration
core/paper_tracker.py             656 lines - Paper trading
core/order_manager.py             ~400 lines - Order management
core/execution_risk.py            ~450 lines - Risk checks
core/smart_router.py              ~300 lines - Order routing
core/position_sizer.py            ~200 lines - Position sizing
```

---

## 🎯 QUICK REFERENCE

**Where does prediction happen?**
→ `core/ensemble_predictor.py` → `EnsemblePredictor.predict()`

**Where is the main API?**
→ `wolf_app.py` → 300+ endpoints defined

**Where do prices come from?**
→ `core/providers/turbo_provider.py` → Multi-source with 2s timeout

**Where is Fear & Greed?**
→ `core/ensemble_predictor.py` lines 33-106

**Where are predictions stored?**
→ `core/prediction_store.py` → PostgreSQL/SQLite

**Where do Telegram alerts come from?**
→ `core/telegram_hunter.py` → `send_telegram_message()`

**Where is the broker?**
→ `core/alpaca_broker.py` → `AlpacaBroker` class

---

*Last Updated: January 7, 2026*
*Total Lines of Code Mapped: ~50,000+*
