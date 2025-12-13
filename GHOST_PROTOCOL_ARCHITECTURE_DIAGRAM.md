# Ghost Protocol — Full System Architecture & Decision Pipeline

**Date**: December 13, 2025  
**Status**: Production Architecture (Web + Worker Split)  
**Purpose**: Executive-level visual representation of the complete prediction pipeline

---

## 🎯 System Overview

Ghost Protocol is a multi-stage AI-powered trading prediction system that processes market data through 9 sequential stages, applying progressive confidence filtering, ensemble modeling, and continuous learning loops to generate high-confidence (≥70%) directional forecasts.

---

## 📊 Complete System Architecture Diagram

```mermaid
flowchart TB
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 0: SYSTEM WAKE & ORCHESTRATION
    %% ═══════════════════════════════════════════════════════════════
    START([🚀 SYSTEM START]) --> SCHEDULER{Scheduler Trigger}
    SCHEDULER -->|Market Hours| MKT_CHECK{Market Status Check}
    SCHEDULER -->|24/7 Crypto| CRYPTO_CHECK{Crypto Mode}
    
    MKT_CHECK -->|Market OPEN| STOCK_UNIVERSE[📋 Stock Universe<br/>135 symbols]
    MKT_CHECK -->|Market CLOSED| SKIP_STOCKS[⏸️ Skip Stocks<br/>Wait Next Open]
    CRYPTO_CHECK --> CRYPTO_UNIVERSE[📋 Crypto Universe<br/>10 top coins]
    
    STOCK_UNIVERSE --> STAGE1
    CRYPTO_UNIVERSE --> STAGE1
    SKIP_STOCKS --> CRYPTO_UNIVERSE
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 1: DATA INGESTION (Parallel Feeds)
    %% ═══════════════════════════════════════════════════════════════
    STAGE1[🌐 STAGE 1: DATA INGESTION<br/>━━━━━━━━━━━━━━━━━━━━━━]
    STAGE1 --> PRICE_FEED[💰 Live Price Feeds<br/>• Polygon.io<br/>• Yahoo Finance<br/>• CoinGecko<br/>• Binance]
    STAGE1 --> OHLCV[📈 Historical OHLCV<br/>• 90-day lookback<br/>• 1h/4h/1d bars]
    STAGE1 --> ORDERBOOK[📊 Market Microstructure<br/>• Order book depth<br/>• Volume profiles<br/>• Bid/Ask spreads]
    STAGE1 --> SENTIMENT[🗞️ Sentiment Feeds<br/>• News APIs<br/>• Reddit PRAW<br/>• Twitter trends<br/>• Fear/Greed Index]
    STAGE1 --> WORLD_CTX[🌍 World Context<br/>• Economic calendar<br/>• Fed announcements<br/>• Market regime]
    
    PRICE_FEED --> RAW_DATA[(🗄️ Raw Market Dataset)]
    OHLCV --> RAW_DATA
    ORDERBOOK --> RAW_DATA
    SENTIMENT --> RAW_DATA
    WORLD_CTX --> RAW_DATA
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 2: DATA VALIDATION & QUALITY GATES
    %% ═══════════════════════════════════════════════════════════════
    RAW_DATA --> STAGE2[🔍 STAGE 2: DATA VALIDATION<br/>━━━━━━━━━━━━━━━━━━━━━━]
    
    STAGE2 --> VAL_COMPLETE{Data Completeness<br/>≥ 80% fields?}
    VAL_COMPLETE -->|FAIL| REFETCH[🔄 Re-fetch<br/>Max 3 retries]
    VAL_COMPLETE -->|PASS| VAL_LATENCY{Latency Check<br/>< 5s old?}
    
    VAL_LATENCY -->|FAIL| STALE[⚠️ Mark Stale<br/>Use cached price]
    VAL_LATENCY -->|PASS| VAL_QUORUM{Provider Quorum<br/>≥ 2 sources agree?}
    
    VAL_QUORUM -->|FAIL| FALLBACK[⚠️ Fallback Logic<br/>• Use prev close<br/>• Degrade confidence]
    VAL_QUORUM -->|PASS| NORM_DATA[✅ Normalized Dataset]
    
    REFETCH -->|Success| VAL_COMPLETE
    REFETCH -->|Max Retries| SKIP_SYM[❌ Skip Symbol<br/>Log failure]
    STALE --> FALLBACK
    FALLBACK --> NORM_DATA
    SKIP_SYM --> END_SKIP([⏭️ Next Symbol])
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 3: FEATURE ENGINEERING (Parallel Engines)
    %% ═══════════════════════════════════════════════════════════════
    NORM_DATA --> STAGE3[⚙️ STAGE 3: FEATURE ENGINEERING<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━]
    
    STAGE3 --> TECH_IND[📊 Technical Indicators<br/>• RSI, MACD, Bollinger<br/>• Moving averages<br/>• Support/Resistance]
    STAGE3 --> VOL_METRICS[📉 Volume Metrics<br/>• Volume spike detection<br/>• OBV, A/D Line<br/>• Smart money flow]
    STAGE3 --> MOMENTUM[🚀 Momentum Engines<br/>• Price momentum<br/>• Trend strength<br/>• Breakout detection]
    STAGE3 --> SENT_SCORE[💬 Sentiment Scoring<br/>• News polarity<br/>• Social mentions<br/>• Macro sentiment]
    STAGE3 --> REGIME_CLASS[🌡️ Regime Classification<br/>• Bull/Bear/Sideways<br/>• Volatility regime<br/>• Risk-on/off]
    
    TECH_IND --> FEATURE_VEC[(🎯 Feature Vector<br/>92% availability<br/>150+ features)]
    VOL_METRICS --> FEATURE_VEC
    MOMENTUM --> FEATURE_VEC
    SENT_SCORE --> FEATURE_VEC
    REGIME_CLASS --> FEATURE_VEC
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 4: MODEL EXECUTION (Parallel Models)
    %% ═══════════════════════════════════════════════════════════════
    FEATURE_VEC --> STAGE4[🤖 STAGE 4: MODEL EXECUTION<br/>━━━━━━━━━━━━━━━━━━━━━━━]
    
    STAGE4 --> STAT_MODEL[📐 Statistical Models<br/>• ARIMA<br/>• GARCH volatility<br/>• Kalman filters]
    STAGE4 --> ML_MODEL[🧠 ML Models<br/>• LightGBM<br/>• Neural networks<br/>• Time series models]
    STAGE4 --> ENSEMBLE[🎭 Ensemble Predictors<br/>• Multi-horizon<br/>• Strategy voting<br/>• Meta-learner]
    
    STAT_MODEL --> RAW_PRED[(📊 Raw Predictions<br/>• Direction probs<br/>• Forecast curves<br/>• Confidence scores)]
    ML_MODEL --> RAW_PRED
    ENSEMBLE --> RAW_PRED
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 5: ENSEMBLE & CONFIDENCE RESOLUTION ⭐ CRITICAL GATE
    %% ═══════════════════════════════════════════════════════════════
    RAW_PRED --> STAGE5[⭐ STAGE 5: CONFIDENCE RESOLUTION<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
    
    STAGE5 --> MODEL_AGREE{Model Agreement<br/>≥ 60% consensus?}
    MODEL_AGREE -->|NO| LOW_CONF[⚠️ Low Confidence<br/>Conflicting signals]
    MODEL_AGREE -->|YES| CONF_WEIGHT[⚖️ Confidence Weighting<br/>• Historical accuracy<br/>• Regime alignment<br/>• Signal strength]
    
    CONF_WEIGHT --> CONF_CALC[📊 Calculate Final Confidence<br/>Weighted average of model scores]
    
    CONF_CALC --> CONF_GATE{⭐ CONFIDENCE GATE<br/>Confidence ≥ 70%?}
    CONF_GATE -->|FAIL < 70%| MONITOR[📋 Monitor Only<br/>No trade signal<br/>Log for learning]
    CONF_GATE -->|PASS ≥ 70%| HIGH_CONF[✅ High Confidence<br/>Continue to risk filter]
    
    LOW_CONF --> MONITOR
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 6: ACCURACY & RISK FILTER ⭐ QUALITY GATE
    %% ═══════════════════════════════════════════════════════════════
    HIGH_CONF --> STAGE6[🎯 STAGE 6: ACCURACY & RISK FILTER<br/>━━━━━━━━━━━━━━━━━━━━━━━━━━━━━]
    
    STAGE6 --> ACC_CHECK{Historical Accuracy<br/>Recent MAP < 15%?}
    ACC_CHECK -->|FAIL| ACCURACY_FAIL[⚠️ Poor Track Record<br/>Downgrade confidence]
    ACC_CHECK -->|PASS| VOL_CHECK{Volatility Risk<br/>ATR within limits?}
    
    VOL_CHECK -->|HIGH RISK| VOL_FAIL[⚠️ Extreme Volatility<br/>Reduce position size]
    VOL_CHECK -->|PASS| REGIME_CHECK{Regime Alignment<br/>Model fits regime?}
    
    REGIME_CHECK -->|MISMATCH| REGIME_FAIL[⚠️ Regime Conflict<br/>Skip or hedge]
    REGIME_CHECK -->|PASS| FALSE_SIG{False Signal Filter<br/>Noise suppression}
    
    FALSE_SIG -->|LIKELY NOISE| SUPPRESS[⛔ Suppress Signal<br/>Below quality threshold]
    FALSE_SIG -->|CLEAN SIGNAL| RISK_PASS[✅ Risk Filter PASSED<br/>Ready for output]
    
    ACCURACY_FAIL --> FEEDBACK_LOOP
    VOL_FAIL --> RISK_PASS
    REGIME_FAIL --> SUPPRESS
    SUPPRESS --> MONITOR
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 7: FINAL PREDICTION OUTPUT
    %% ═══════════════════════════════════════════════════════════════
    RISK_PASS --> STAGE7[🎯 STAGE 7: FINAL PREDICTION<br/>━━━━━━━━━━━━━━━━━━━━━━━━]
    
    STAGE7 --> PRED_ARTIFACT[📦 Prediction Artifact<br/>━━━━━━━━━━━━━━━━━━<br/>• Symbol: BTC<br/>• Direction: UP ⬆️<br/>• Confidence: 75.2%<br/>• Horizon: 48h<br/>• Target: $45,200<br/>• Stop: $42,800<br/>• Signals: 12/15<br/>━━━━━━━━━━━━━━━━━━]
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 8: DISTRIBUTION & ALERTS
    %% ═══════════════════════════════════════════════════════════════
    PRED_ARTIFACT --> STAGE8[📡 STAGE 8: DISTRIBUTION<br/>━━━━━━━━━━━━━━━━━━━━━]
    
    STAGE8 --> COCKPIT_UI[🖥️ Cockpit UI<br/>Real-time dashboard]
    STAGE8 --> ALERTS[📢 Alert System<br/>• Telegram bot<br/>• Push notifications<br/>• Email summaries]
    STAGE8 --> LOGGING[📝 Logging & Storage<br/>• SQLite DB<br/>• Prediction history<br/>• Performance logs]
    STAGE8 --> TRACKING[📊 Performance Tracking<br/>• Live accuracy<br/>• P&L tracking<br/>• Risk metrics]
    
    COCKPIT_UI --> USER_VIEW[👁️ User View]
    ALERTS --> USER_VIEW
    LOGGING --> DB[(💾 Ghost DB)]
    TRACKING --> DB
    
    %% ═══════════════════════════════════════════════════════════════
    %% STAGE 9: FEEDBACK & LEARNING LOOP 🔄
    %% ═══════════════════════════════════════════════════════════════
    DB --> STAGE9[🔄 STAGE 9: LEARNING LOOP<br/>━━━━━━━━━━━━━━━━━━━━━━]
    
    STAGE9 --> WAIT_OUTCOME[⏳ Wait for Outcome<br/>24h / 48h / 7d]
    WAIT_OUTCOME --> ACTUAL_PRICE[💰 Capture Actual Price<br/>Market close / horizon end]
    
    ACTUAL_PRICE --> COMPARE[⚖️ Compare Prediction vs Reality<br/>• Direction correct?<br/>• Price target hit?<br/>• Within confidence bands?]
    
    COMPARE --> CALC_ACCURACY[📊 Calculate Accuracy Metrics<br/>• MAP (Mean Absolute % Error)<br/>• Hit rate<br/>• Confidence calibration]
    
    CALC_ACCURACY --> UPDATE_WEIGHTS[⚙️ Update Model Weights<br/>• Boost accurate models<br/>• Penalize poor performers<br/>• Adjust thresholds]
    
    UPDATE_WEIGHTS --> STORE_LEARNING[💾 Store Learning Artifacts<br/>• Feature importance<br/>• Error patterns<br/>• Regime performance]
    
    STORE_LEARNING --> FEEDBACK_LOOP[🔄 FEEDBACK TO STAGES 4-6]
    FEEDBACK_LOOP -.->|Update model weights| STAGE4
    FEEDBACK_LOOP -.->|Adjust confidence thresholds| STAGE5
    FEEDBACK_LOOP -.->|Refine accuracy filters| STAGE6
    
    %% ═══════════════════════════════════════════════════════════════
    %% MONITORING & CONTROL
    %% ═══════════════════════════════════════════════════════════════
    MONITOR -.->|Low confidence predictions| STAGE9
    USER_VIEW --> MANUAL_OVERRIDE{Manual Override?}
    MANUAL_OVERRIDE -->|Yes| KILL_SWITCH[🛑 Kill Switch<br/>Stop all trading]
    MANUAL_OVERRIDE -->|No| CONTINUE[▶️ Continue Pipeline]
    CONTINUE --> END([✅ PIPELINE COMPLETE])
    KILL_SWITCH --> END
    
    %% ═══════════════════════════════════════════════════════════════
    %% STYLING
    %% ═══════════════════════════════════════════════════════════════
    classDef startEnd fill:#00ff00,stroke:#006600,stroke-width:3px,color:#000
    classDef stage fill:#1e90ff,stroke:#0047ab,stroke-width:2px,color:#fff
    classDef critical fill:#ff6b6b,stroke:#c92a2a,stroke-width:3px,color:#fff
    classDef pass fill:#51cf66,stroke:#2f9e44,stroke-width:2px,color:#000
    classDef fail fill:#ffa94d,stroke:#e67700,stroke-width:2px,color:#000
    classDef data fill:#748ffc,stroke:#4c6ef5,stroke-width:2px,color:#fff
    classDef decision fill:#ffd43b,stroke:#fab005,stroke-width:2px,color:#000
    
    class START,END startEnd
    class STAGE1,STAGE2,STAGE3,STAGE4,STAGE5,STAGE6,STAGE7,STAGE8,STAGE9 stage
    class CONF_GATE,STAGE5,STAGE6 critical
    class NORM_DATA,HIGH_CONF,RISK_PASS,PRED_ARTIFACT pass
    class SKIP_STOCKS,SKIP_SYM,FALLBACK,LOW_CONF,ACCURACY_FAIL,VOL_FAIL,REGIME_FAIL,SUPPRESS fail
    class RAW_DATA,FEATURE_VEC,RAW_PRED,DB data
    class MKT_CHECK,CRYPTO_CHECK,VAL_COMPLETE,VAL_LATENCY,VAL_QUORUM,MODEL_AGREE,CONF_GATE,ACC_CHECK,VOL_CHECK,REGIME_CHECK,FALSE_SIG,MANUAL_OVERRIDE decision
```

---

## 🔑 Legend & Key Concepts

### Symbol Types

| Symbol | Meaning | Example |
|--------|---------|---------|
| `🚀` | System start/trigger | Scheduler, cron jobs |
| `{Diamond}` | **Decision gate** | Confidence ≥ 70%? |
| `[Rectangle]` | **Process/Stage** | Feature engineering |
| `[(Database)]` | **Data storage** | Raw data, predictions |
| `⭐` | **Critical gate** | Must pass to continue |
| `✅` | **Pass condition** | Quality threshold met |
| `⚠️` | **Warning/degraded** | Fallback mode active |
| `❌` | **Failure/skip** | Symbol rejected |
| `🔄` | **Feedback loop** | Learning updates models |
| `-.->` | **Feedback arrow** | Updates flow back |

### Color Coding

- **Green** = Start/End points, successful outcomes
- **Blue** = Processing stages
- **Red** = Critical gates (70% confidence, accuracy filters)
- **Light Green** = Passed quality checks
- **Orange** = Warnings, fallbacks, skipped symbols
- **Purple** = Data storage/artifacts
- **Yellow** = Decision points

---

## ⭐ Critical Gates & Thresholds

### STAGE 5: Confidence Gate (⭐ PRIMARY FILTER)
```
IF Confidence < 70%:
    → Route to MONITOR ONLY (no trade signal)
    → Log for learning loop
    → Track false negatives

IF Confidence ≥ 70%:
    → Continue to STAGE 6 (Risk Filter)
    → High-confidence prediction
    → Eligible for trade signals
```

**Enforcement Point**: `CONF_GATE` decision diamond  
**Target**: 70%+ confidence for all trade signals  
**Fallback**: Low-confidence predictions monitored but not acted upon

### STAGE 6: Accuracy Filter (⭐ QUALITY GATE)
```
Historical Accuracy Check:
    MAP (Mean Absolute % Error) < 15% = PASS
    MAP ≥ 15% = Downgrade confidence or skip

Volatility Risk Check:
    ATR within normal range = PASS
    Extreme volatility = Reduce position size

Regime Alignment:
    Model trained for current regime = PASS
    Regime mismatch = Skip or hedge

False Signal Filter:
    Clean signal (low noise) = PASS
    Likely noise = SUPPRESS
```

**Enforcement Point**: Multiple checks in STAGE 6  
**Purpose**: Prevent low-quality predictions from reaching users  
**Outcome**: Only high-confidence + high-accuracy signals distributed

---

## 🔄 Feedback Loops (Learning System)

### Primary Feedback Loop (STAGE 9 → STAGES 4-6)

1. **Capture Outcome** (24h/48h after prediction)
   - Actual price vs predicted price
   - Direction correct? (UP/DOWN)
   - Within confidence bands?

2. **Calculate Accuracy** (STAGE 9)
   - MAP (Mean Absolute Percentage Error)
   - Hit rate (% correct direction)
   - Confidence calibration (overconfidence penalty)

3. **Update System** (Feedback to earlier stages)
   - **STAGE 4**: Boost weights of accurate models, penalize poor models
   - **STAGE 5**: Adjust confidence thresholds based on calibration
   - **STAGE 6**: Refine accuracy filters, update risk parameters

4. **Store Learning** (STAGE 9)
   - Feature importance scores
   - Error patterns (when/why predictions fail)
   - Regime-specific performance

### Secondary Feedback Loops

- **STAGE 2 → STAGE 1**: Failed validation triggers re-fetch
- **STAGE 6 → STAGE 5**: Poor accuracy downgrades confidence
- **MONITOR → STAGE 9**: Low-confidence predictions tracked for learning

---

## 🎯 Data Flow Summary

### Input → Output Transformation

```
Market Data (Live prices, OHLCV, sentiment)
    ↓
Validated Dataset (Quality gates passed)
    ↓
Feature Vector (150+ technical/sentiment features)
    ↓
Raw Model Predictions (Direction probabilities, forecast curves)
    ↓
High-Confidence Prediction (≥70% confidence, ensemble agreement)
    ↓
Risk-Filtered Signal (Accuracy verified, volatility checked)
    ↓
Final Prediction Artifact (Symbol, direction, target, stop-loss)
    ↓
User-Facing Outputs (Cockpit UI, Telegram alerts, logs)
    ↓
Learning Loop (Compare actual vs predicted, update models)
```

---

## 🚀 System Performance Targets

| Metric | Target | Enforcement |
|--------|--------|-------------|
| **Prediction Confidence** | ≥ 70% | STAGE 5 gate |
| **Historical Accuracy (MAP)** | < 15% | STAGE 6 gate |
| **Hit Rate (Direction)** | ≥ 60% | STAGE 6 gate |
| **Data Completeness** | ≥ 80% | STAGE 2 gate |
| **Provider Quorum** | ≥ 2 sources | STAGE 2 gate |
| **Pipeline Latency** | < 5 seconds | STAGE 2 validation |
| **Feature Availability** | ≥ 85% | STAGE 3 output |
| **Model Agreement** | ≥ 60% consensus | STAGE 5 logic |

---

## 📦 Prediction Artifact Schema

Every prediction that passes all gates produces this standardized artifact:

```json
{
  "prediction_id": "uuid-12345",
  "symbol": "BTC",
  "asset_type": "crypto",
  "timestamp": 1702454400,
  "direction": "UP",
  "confidence": 75.2,
  "horizon_hours": 48,
  "current_price": 43251.50,
  "target_price": 45200.00,
  "stop_loss": 42800.00,
  "take_profit": 45500.00,
  "supporting_signals": 12,
  "total_signals": 15,
  "model_votes": {
    "statistical": "UP",
    "ml_ensemble": "UP",
    "momentum": "UP"
  },
  "risk_level": "MEDIUM",
  "volatility_atr": 0.048,
  "regime": "BULL",
  "accuracy_map_7d": 4.2,
  "accuracy_map_30d": 8.7
}
```

---

## 🔧 Operational Modes

### WEB Mode (API Server)
- Serves HTTP requests
- Returns cached predictions
- No background prediction loops
- Reads from shared prediction store

### WORKER Mode (Prediction Engine)
- Runs STAGES 0-9 continuously
- Executes auto-prediction loops
- Writes to shared prediction store
- Updates learning artifacts

### Architecture Split (Railway Deployment)
```
┌─────────────────┐         ┌──────────────────┐
│   WEB Process   │         │  WORKER Process  │
│  (HTTP Server)  │◄────────┤ (Prediction Loop)│
│                 │  Shared │                  │
│ • Cockpit UI    │  Store  │ • STAGES 0-9     │
│ • API endpoints │◄────────┤ • Learning loop  │
│ • Alerts        │ (Redis) │ • Model updates  │
└─────────────────┘         └──────────────────┘
```

---

## 🏆 Success Criteria

A prediction is considered **production-ready** when:

1. ✅ **Confidence ≥ 70%** (STAGE 5 gate)
2. ✅ **Historical MAP < 15%** (STAGE 6 gate)
3. ✅ **Model agreement ≥ 60%** (STAGE 5 logic)
4. ✅ **Data quality passed** (STAGE 2 validation)
5. ✅ **Volatility within limits** (STAGE 6 risk check)
6. ✅ **Regime alignment confirmed** (STAGE 6 regime check)
7. ✅ **False signal filter passed** (STAGE 6 noise suppression)

**Only predictions meeting ALL criteria reach users.**

---

## 📊 Monitoring & Observability

### Key Metrics Tracked

- **Pipeline Health**: Success rate per stage
- **Data Quality**: Validation pass rate, provider uptime
- **Model Performance**: Per-model accuracy, confidence calibration
- **Prediction Quality**: Distribution of confidence scores, hit rate
- **Latency**: Time spent in each stage
- **Feedback Loop**: Learning rate, model weight adjustments

### Alerting Thresholds

- Pipeline failure rate > 10%
- Data validation failure rate > 20%
- Confidence calibration drift > 15%
- Historical accuracy (MAP) > 15%
- Provider downtime > 30s

---

**Document Version**: 1.0  
**Last Updated**: December 13, 2025  
**Status**: ✅ Production Architecture
