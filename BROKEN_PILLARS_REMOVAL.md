# Broken Pillars Removal - Deployment Summary
**Date:** January 13, 2026  
**Status:** ✅ COMPLETE - Ready for Production Deployment

---

## Changes Made

### 1. ✅ Disabled sentiment_engine Pillar

**File:** `core/data_pillars/feature_orchestrator.py`

**Changes:**
- Commented out `SentimentEngine` import
- Commented out `self.sentiment_engine = SentimentEngine()` initialization
- Disabled sentiment feature extraction in `get_all_features()`
- Updated pillar stats to show `"DISABLED (dummy data)"`
- Removed from health check loop

**Reason:**
- Alpha Vantage News Sentiment API timing out (10 second timeout)
- Returns constant 0.0 neutral sentiment for ALL predictions
- Provides NO predictive value (constant = no signal)
- Slows down prediction pipeline with failed API calls

**Impact:**
- ✅ Faster predictions (no 10 second API timeout)
- ✅ Cleaner feature dict (no dummy 0.0 values)
- ✅ No loss of accuracy (pillar was returning constant values)
- ✅ Fewer errors logged

---

### 2. ✅ Disabled world_context_engine Pillar

**File:** `core/data_pillars/feature_orchestrator.py`

**Changes:**
- Commented out `WorldContextEngine` import
- Commented out `self.world_context_engine = WorldContextEngine()` initialization
- Disabled world context feature extraction in `get_all_features()`
- Updated pillar stats to show `"DISABLED (dummy data)"`
- Removed from health check loop

**Reason:**
- SPY/VIX price fetching fails (returns NULL)
- Returns constant 50.0 neutral market mood for ALL predictions
- Provides NO predictive value (constant = no signal)
- Adds complexity with no benefit

**Impact:**
- ✅ Simpler prediction pipeline
- ✅ Fewer failure points (no SPY/VIX fetch attempts)
- ✅ No loss of accuracy (pillar was returning constant neutral values)
- ✅ Cleaner logs

---

### 3. Updated Health Check

**Changes:**
- Updated health check to expect 4 pillars instead of 6
- Changed minimum healthy threshold from 4/6 to 3/4
- Updated total count from 6 to 4 in summary

**Remaining Active Pillars:**
1. ✅ **price_engine** - Multi-source price data (WORKING)
2. ✅ **technical_engine** - RSI, MACD, indicators (WORKING)
3. ✅ **volume_engine** - Volume analysis (WORKING)
4. ✅ **flow_engine** - Orderbook/on-chain (WORKING)

**Disabled Pillars:**
- ❌ **sentiment_engine** - Dummy 0.0 data (Alpha Vantage timeout)
- ❌ **world_context_engine** - Dummy null/50 data (SPY/VIX fails)

---

## Ghost News Brain Verification

**Status:** ⚠️ **REQUIRES MANUAL VERIFICATION IN PRODUCTION**

### Created Verification Script

**File:** `verify_ghost_news_brain.sh`

**What it checks:**
1. Railway logs for News Brain activity (`📰 News Analysis`)
2. Environment variables (NEWS_ANALYSIS_ENABLED, ANTHROPIC_API_KEY)
3. `news_analysis` table for recent records
4. `guardian_alerts` table for alert history
5. Telegram channel for News Brain alerts

### To Run in Production:

```bash
# SSH into Railway production environment
railway run bash

# Run verification script
./verify_ghost_news_brain.sh
```

### What to Look For:

**IF NEWS BRAIN IS RUNNING:**
- Logs show: `"📰 News Analysis Loop: STARTING"`
- Logs show: `"📰 Running automatic news analysis..."` every 30 minutes
- `news_analysis` table has records every 30 minutes
- Telegram shows News Brain alerts for HIGH/CRITICAL events
- **Action:** Monitor alert quality and Anthropic API costs

**IF NEWS BRAIN IS NOT RUNNING:**
- No logs with `📰` emoji
- `news_analysis` table is empty or doesn't exist
- No Telegram alerts from News Brain
- **Action:** Either FIX IT or DISABLE IT (to save API credits)

### Decision Matrix:

| Scenario | Action |
|----------|--------|
| Running + Useful alerts | ✅ Keep enabled, monitor costs |
| Running + Noisy alerts | 🟡 Reduce frequency (30min → 60min) |
| Running + Expensive | 🟡 Disable or reduce frequency |
| NOT running | 🔴 Fix OR disable to save complexity |

---

## How to Disable Ghost News Brain (If Not Running)

If verification shows News Brain is NOT running or NOT useful:

**Option 1: Environment Variable (Preferred)**
```bash
# In Railway production environment
railway variables set NEWS_ANALYSIS_ENABLED=0
```

**Option 2: Code Change**
Edit `wolf_app.py` lines 4880-4950:
```python
# Set to False to disable
NEWS_ANALYSIS_ENABLED = False  # os.getenv("NEWS_ANALYSIS_ENABLED", "1") == "1"
```

---

## Testing Before Deployment

### 1. Test Feature Orchestrator Locally

```bash
cd /workspaces/ghost-protocol

python3 <<EOF
from core.data_pillars.feature_orchestrator import get_feature_orchestrator

orchestrator = get_feature_orchestrator()

# Test with whitelisted symbol
features = orchestrator.get_all_features("RNDR")

print("Feature Orchestrator Test:")
print(f"✅ Features extracted: {features['feature_count']}")
print(f"✅ Available: {features['available_count']}")
print(f"✅ Unavailable: {features['unavailable_count']}")
print(f"✅ Execution time: {features['execution_time_ms']}ms")
print(f"\n📊 Pillar Stats:")
for pillar, stat in features['feature_availability'].items():
    print(f"   {pillar}: {stat}")
print(f"\n⚠️  Errors: {len(features['errors'])}")
for err in features['errors']:
    print(f"   - {err}")
EOF
```

**Expected Output:**
- sentiment_engine: `DISABLED (dummy data)`
- world_context_engine: `DISABLED (dummy data)`
- No sentiment/world context features in features dict
- Faster execution time (no API timeouts)

### 2. Test Health Check

```bash
python3 <<EOF
from core.data_pillars.feature_orchestrator import get_feature_orchestrator

orchestrator = get_feature_orchestrator()
health = orchestrator.health_check()

print("Health Check:")
print(f"✅ Overall OK: {health['ok']}")
print(f"📊 Summary: {health['summary']}")
print(f"\n🔍 Pillar Status:")
for pillar, status in health['pillars'].items():
    ok_status = "✅" if status.get('ok') else "❌"
    print(f"   {ok_status} {pillar}: {status}")
EOF
```

**Expected Output:**
- Total: 4 pillars (down from 6)
- Healthy: 3-4 pillars
- No sentiment_engine or world_context_engine in results

---

## Deployment Steps

### 1. Commit Changes

```bash
cd /workspaces/ghost-protocol

git add core/data_pillars/feature_orchestrator.py
git add verify_ghost_news_brain.sh
git add BROKEN_PILLARS_REMOVAL.md

git commit -m "Remove broken sentiment and world_context pillars

- Disabled sentiment_engine (Alpha Vantage API timeout, returns 0.0 dummy data)
- Disabled world_context_engine (SPY/VIX price fetch fails, returns null/50 dummy data)
- Updated health check to expect 4 pillars instead of 6
- Created verification script for Ghost News Brain status
- No loss of accuracy: pillars returned constant values (no signal)
- Benefit: Faster predictions, fewer errors, simpler pipeline"

git push origin main
```

### 2. Deploy to Production

```bash
# Deploy via Railway
railway up

# Or via Railway dashboard
# git push triggers automatic deployment
```

### 3. Verify Deployment

```bash
# Check Railway logs for successful startup
railway logs --tail 100

# Look for:
# - "Feature orchestrator initialized"
# - No sentiment_engine errors
# - No world_context_engine errors
# - Faster prediction times
```

### 4. Run Ghost News Brain Verification

```bash
railway run bash
./verify_ghost_news_brain.sh
```

### 5. Monitor Win Rate

```sql
-- Run after 24 hours
SELECT 
    COUNT(*) as total_predictions,
    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
FROM predictions
WHERE created_at > NOW() - INTERVAL '24 hours'
AND symbol IN ('RLC', 'RNDR', 'ICP', 'CHZ', 'EGLD', 'ILV', 'OCEAN', 'T', 'TURBO', 'ZEC');
```

**Expected:** Win rate stays at 70%+ (or improves)  
**If drops:** Investigate, but unlikely (pillars returned constant dummy values)

---

## Rollback Plan (If Needed)

If win rate drops unexpectedly:

```bash
# Revert changes
git revert HEAD
git push origin main
railway up
```

Then investigate:
- Were sentiment/world_context pillars providing ANY signal?
- Check correlation: sentiment scores vs. win rate
- Fix Alpha Vantage API timeout (increase to 30s)
- Fix SPY/VIX price fetching

---

## Summary

### What Was Changed ✅
- ✅ Disabled sentiment_engine (dummy 0.0 data)
- ✅ Disabled world_context_engine (dummy null/50 data)
- ✅ Updated health check (4 pillars instead of 6)
- ✅ Created Ghost News Brain verification script

### What Remains Active ✅
- ✅ price_engine (multi-source price data)
- ✅ technical_engine (RSI, MACD, indicators)
- ✅ volume_engine (volume analysis)
- ✅ flow_engine (orderbook/on-chain)
- ✅ V2 quality filter (10 symbol whitelist)

### Expected Impact 🚀
- ⚡ **Faster predictions** (no 10s API timeout)
- 🧹 **Cleaner logs** (fewer errors)
- 📊 **Same or better win rate** (removed noise, not signal)
- 🎯 **Simpler system** (fewer failure points)

### Next Steps 📋
1. ✅ Deploy changes to production
2. ⏳ Run `verify_ghost_news_brain.sh` in production
3. ⏳ Decide: Keep News Brain enabled or disable it
4. ⏳ Monitor win rate for 24-48 hours
5. ⏳ If win rate stays 70%+: Success! Keep pillars disabled
6. ⏳ If win rate drops: Investigate (unlikely)

---

**Status:** Ready for production deployment  
**Risk Level:** 🟢 LOW (removing noise, not signal)  
**Rollback Time:** < 5 minutes if needed  
**Expected Outcome:** Same or better win rate, faster predictions
