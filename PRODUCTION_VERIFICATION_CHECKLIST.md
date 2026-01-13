# 🔍 PRODUCTION VERIFICATION CHECKLIST

**Status:** ⏳ AWAITING VERIFICATION  
**Deployed:** January 13, 2026  
**Commit:** f3ac7f4

---

## ❗ NEW RULE: "DONE" = WORKING IN PRODUCTION

**NOT done:**
- ❌ Code exists
- ❌ Local tests pass
- ❌ Deployed to Railway

**DONE means:**
- ✅ Deployed to production
- ✅ Real data flowing
- ✅ Railway logs show activity
- ✅ Database queries confirm functionality
- ✅ PROOF with screenshots/logs

---

## 📋 VERIFICATION STEPS

### Step 1: Wait for Railway Deployment

```bash
# Check deployment status
railway status

# Watch deployment logs
railway logs --tail 50
```

**What to look for:**
- ✅ "Deployment successful"
- ✅ No errors during startup
- ✅ All services initialized

**PROOF REQUIRED:** Screenshot of successful deployment

---

### Step 2: Run Production Verification Script

```bash
# SSH into Railway production
railway run bash

# Run verification script
./VERIFY_PRODUCTION.sh
```

**What to look for:**
- ✅ All tests pass with green checkmarks
- ✅ Real data (not NULL, not dummy values)
- ✅ Timestamps are recent (< 60 min)

**PROOF REQUIRED:** Full output of VERIFY_PRODUCTION.sh

---

### Step 3: Verify Ghost News Brain

#### Check if News Brain is running:

```bash
railway logs --tail 500 | grep "News Analysis"
```

**Expected output:**
```
📰 News Analysis Loop: STARTING (every 30 min)
📰 Running automatic news analysis...
📰 News analysis complete: 3 events, 0 predictions at risk
```

**PROOF REQUIRED:** 
- [ ] Screenshot showing "News Analysis Loop: STARTING"
- [ ] Screenshot showing recent analysis run (< 30 min ago)

#### Check news_analysis table:

```bash
railway run psql -c "
SELECT 
    analysis_id,
    analysis_time,
    headlines_fetched,
    events_found,
    EXTRACT(EPOCH FROM (NOW() - analysis_time))/60 as age_minutes
FROM news_analysis 
ORDER BY analysis_time DESC 
LIMIT 10;
"
```

**Expected output:**
```
analysis_id | analysis_time       | headlines_fetched | events_found | age_minutes
------------|---------------------|-------------------|--------------|-------------
123         | 2026-01-13 12:30:00 | 47                | 3            | 15
122         | 2026-01-13 12:00:00 | 51                | 2            | 45
...
```

**PROOF REQUIRED:**
- [ ] Records exist (not empty)
- [ ] Latest record < 60 min old
- [ ] headlines_fetched > 0
- [ ] Screenshot of query results

---

### Step 4: Verify Sentiment Engine

#### Test with known crypto symbol:

```bash
railway run python3 <<'EOF'
from core.data_pillars.sentiment_engine import SentimentEngine

engine = SentimentEngine()
response = engine.get_signals("BTC")

print(f"Signals: {response.signal_count()}")
print(f"Available: {response.available_signal_count()}")
print(f"Execution time: {response.execution_time_ms}ms")
print()

for signal in response.signals:
    print(f"  {signal.name}: {signal.value} (source: {signal.source})")
EOF
```

**Expected output:**
```
Signals: 3
Available: 3
Execution time: 234ms

  NEWS_SENTIMENT_SCORE: 0.4 (source: ghost_news_brain)  ← NOT 0.0 dummy!
  NEWS_COUNT_24H: 5 (source: ghost_news_brain)          ← NOT 0!
  BULLISH_RATIO: 0.7 (source: ghost_news_brain)         ← NOT 0.5!
```

**PROOF REQUIRED:**
- [ ] All 3 signals available (not NULL)
- [ ] Source is "ghost_news_brain" or "rss_scan" (NOT "unavailable")
- [ ] Values are NOT dummy (0.0, 0, 0.5)
- [ ] Screenshot of output

#### Check Railway logs for sentiment activity:

```bash
railway logs --tail 200 | grep -i sentiment
```

**Expected output:**
```
[SENTIMENT] BTC: Using Ghost News Brain cached analysis
[SENTIMENT] ETH: Using RSS feed scan (3 articles)
[SENTIMENT] RNDR: No recent news, returning neutral
```

**PROOF REQUIRED:**
- [ ] Logs show sentiment engine running
- [ ] Using Ghost News Brain or RSS (not Alpha Vantage)
- [ ] Screenshot of logs

---

### Step 5: Verify World Context (SPY/VIX)

#### Test SPY/VIX prices:

```bash
railway run python3 <<'EOF'
from core.world_context import get_world_context

context = get_world_context()

print(f"SPY Price: {context['spy']['price']}")
print(f"SPY Provider: {context['spy']['provider']}")
print(f"VIX Level: {context['vix']['level']}")
print(f"VIX Status: {context['vix']['status']}")
print(f"Market Mood: {context['market_mood']['sentiment']}")
EOF
```

**Expected output:**
```
SPY Price: 580.45          ← NOT NULL!
SPY Provider: yfinance_fallback  ← OR price_quorum
VIX Level: 14.8            ← NOT NULL!
VIX Status: calm           ← NOT unknown!
Market Mood: bullish       ← NOT neutral (unless market is actually neutral)
```

**PROOF REQUIRED:**
- [ ] SPY price is NOT NULL
- [ ] VIX level is NOT NULL
- [ ] Provider shown (yfinance_fallback or price_quorum)
- [ ] Screenshot of output

#### Check Railway logs for world context:

```bash
railway logs --tail 200 | grep -E "SPY|VIX|yfinance"
```

**Expected output (if fallback used):**
```
SPY price_quorum returned NULL, trying yfinance fallback...
✅ SPY yfinance fallback: $580.45 (+1.2%)
VIX price_quorum returned NULL, trying yfinance fallback...
✅ VIX yfinance fallback: 14.8 (calm)
```

**PROOF REQUIRED:**
- [ ] Logs show SPY/VIX being fetched
- [ ] Either price_quorum works OR yfinance fallback succeeds
- [ ] Screenshot of logs

---

### Step 6: Verify Full Integration

#### Test feature orchestrator:

```bash
railway run python3 <<'EOF'
from core.data_pillars.feature_orchestrator import get_feature_orchestrator

orchestrator = get_feature_orchestrator()
features = orchestrator.get_all_features("RNDR")

print(f"Total features: {features['feature_count']}")
print(f"Available: {features['available_count']}")
print()

for pillar, stat in features['feature_availability'].items():
    print(f"  {pillar}: {stat}")

print()
print("Key Features:")
for key in ["NEWS_SENTIMENT_SCORE", "NEWS_COUNT_24H", "SPY_PRICE", "VIX_LEVEL"]:
    value = features['features'].get(key)
    print(f"  {key}: {value}")
EOF
```

**Expected output:**
```
Total features: 75
Available: 73

  price_engine: 1/1
  technical_engine: 66/66
  volume_engine: 5/5
  sentiment_engine: 3/3      ← NOT "DISABLED"!
  world_context_engine: 4/4  ← NOT "DISABLED"!
  flow_engine: 0/1

Key Features:
  NEWS_SENTIMENT_SCORE: 0.3    ← Real value (or 0.0 if no news)
  NEWS_COUNT_24H: 2            ← Real count (or 0 if no news)
  SPY_PRICE: 580.45            ← NOT None!
  VIX_LEVEL: 14.8              ← NOT None!
```

**PROOF REQUIRED:**
- [ ] sentiment_engine shows X/3 (not "DISABLED")
- [ ] world_context_engine shows X/4 (not "DISABLED")
- [ ] SPY_PRICE is NOT None
- [ ] VIX_LEVEL is NOT None
- [ ] Screenshot of output

---

### Step 7: Test with Real Prediction

#### Make a prediction and check logs:

```bash
# Make prediction
railway run python3 <<'EOF'
from wolf_app import run_single_prediction
result = run_single_prediction("BTC")
print(result)
EOF

# Check logs for news integration
railway logs --tail 100 | grep -E "BTC.*sentiment|BTC.*news|BTC.*world"
```

**Expected in logs:**
```
[SENTIMENT] BTC: Using Ghost News Brain cached analysis
[WORLD_CONTEXT] SPY: $580.45, VIX: 14.8 (calm)
Prediction for BTC: UP (confidence: 0.82)
```

**PROOF REQUIRED:**
- [ ] Prediction uses sentiment data
- [ ] Prediction uses world context data
- [ ] Logs show real values (not dummy)
- [ ] Screenshot of prediction output + logs

---

## ✅ COMPLETION CRITERIA

### Ghost News Brain:
- [ ] `news_analysis` table has records < 60 min old
- [ ] Railway logs show "News Analysis Loop: STARTING"
- [ ] Railway logs show periodic analysis runs (every 30 min)
- [ ] Database shows headlines_fetched > 0

### Sentiment Engine:
- [ ] Returns real sentiment scores (not always 0.0)
- [ ] Source is "ghost_news_brain" or "rss_scan" (not "unavailable")
- [ ] Railway logs show sentiment engine using news data
- [ ] Works for at least 3 test symbols (BTC, ETH, RNDR)

### World Context:
- [ ] SPY price is NOT NULL
- [ ] VIX level is NOT NULL
- [ ] Provider shown (price_quorum or yfinance_fallback)
- [ ] Railway logs show successful price fetches

### Integration:
- [ ] Feature orchestrator shows all 6 pillars active
- [ ] sentiment_engine: 3/3 (not "DISABLED")
- [ ] world_context_engine: 4/4 (not "DISABLED")
- [ ] Real predictions use news + world context data

---

## 📸 PROOF REQUIRED

Upload to issue/PR:
1. Screenshot: Railway deployment success
2. Screenshot: VERIFY_PRODUCTION.sh full output
3. Screenshot: news_analysis table query results
4. Screenshot: Sentiment engine test output (BTC)
5. Screenshot: World context test output (SPY/VIX)
6. Screenshot: Feature orchestrator output (RNDR)
7. Screenshot: Railway logs showing news activity
8. Screenshot: Real prediction with news integration

---

## ❌ IF ANY TEST FAILS

### Ghost News Brain not running:
```bash
# Check environment
railway variables | grep NEWS_ANALYSIS

# Expected:
# NEWS_ANALYSIS_ENABLED=1
# ANTHROPIC_API_KEY=sk-ant-...

# Manually trigger analysis
railway run python3 -c "from core.intelligence.ghost_news_brain import test_news_brain; import asyncio; asyncio.run(test_news_brain())"
```

### Sentiment engine returning dummy data:
```bash
# Check if Ghost News Brain has cached data
railway run python3 -c "from core.intelligence.ghost_news_brain import get_news_brain; brain = get_news_brain(); print(brain.get_cached_analysis('BTC'))"
```

### World context returning NULL:
```bash
# Test yfinance directly
railway run python3 -c "import yfinance as yf; spy = yf.Ticker('SPY'); print(spy.history(period='1d'))"
```

---

## 📊 STATUS

**Deployment:** ⏳ In Progress  
**Ghost News Brain:** ⏳ Awaiting Verification  
**Sentiment Engine:** ⏳ Awaiting Verification  
**World Context:** ⏳ Awaiting Verification  
**Integration:** ⏳ Awaiting Verification

**DONE when ALL checkboxes above are ✅ and PROOF screenshots uploaded.**
