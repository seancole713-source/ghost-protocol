# Ghost News Brain Verification Guide

**CRITICAL: Ghost News Brain loop is NOT running in production**

## ✅ What We Fixed

### 1. V2 Filter - **VERIFIED WORKING** ✅
- DASH: Blocked ✅
- LRC: Blocked ✅  
- ZEC: Allowed ✅
- Fix: Removed non-whitelisted from DEFAULT_CRYPTO_SYMBOLS, added cache filtering

### 2. Sentiment Engine - **CODE DEPLOYED, NOT VERIFIED** ⚠️
- Rewritten to use Ghost News Brain → RSS → Neutral fallback
- File: `core/data_pillars/sentiment_engine.py`
- Commit: f3ac7f4
- **Needs verification:** Are sentiment scores real (>0.0) or dummy (0.0)?

### 3. World Context - **CODE DEPLOYED, NOT VERIFIED** ⚠️
- Added yfinance fallback for SPY/VIX when price_quorum returns NULL
- File: `core/world_context.py`
- Commit: f3ac7f4
- **Needs verification:** Are SPY/VIX values real or NULL?

### 4. Ghost News Brain - **LOOP NOT RUNNING** ❌
- Code exists: wolf_app.py lines 4910-4967
- Integration exists: ghost_news_brain.py get_cached_analysis()
- Commit: f3ac7f4
- **PROBLEM:** Loop never starts, no logs found

---

## 🔴 CRITICAL ISSUE: Ghost News Brain Loop Not Starting

### Expected Logs (NOT FOUND):
```
📰 News Analysis Loop: STARTING (every 30 min)
📰 Running automatic news analysis...
```

### Actual Result:
```bash
railway logs --tail 200 | grep "News Analysis"
# EMPTY - No output
```

### Root Cause Investigation:

**Location:** `wolf_app.py` lines 4910-4967

**Initialization Code:**
```python
NEWS_ANALYSIS_ENABLED = os.getenv("NEWS_ANALYSIS_ENABLED", "1") == "1"

if NEWS_ANALYSIS_ENABLED:
    try:
        from core.intelligence.ghost_news_brain import get_news_brain
        asyncio.create_task(_news_analysis_loop())
        LOGGER.info("📰 News Analysis Loop: STARTING (every 30 min)")
    except Exception as e:
        LOGGER.error(f"news_analysis_start_failed: {e}", exc_info=True)
```

**Possible Causes:**
1. ❓ `NEWS_ANALYSIS_ENABLED` set to "0" in Railway environment
2. ❓ `ANTHROPIC_API_KEY` not set (Claude AI won't work)
3. ❓ Import error: `get_news_brain` fails to import
4. ❓ Database tables missing: `news_analysis`, `guardian_alerts`
5. ❓ Exception caught silently (line 4967)

---

## 🔍 Verification Commands (Run These in Railway)

### 1. Check for Startup Error
```bash
railway logs --tail 1000 | grep "news_analysis_start_failed"
```
**Expected:** Error message if loop failed to start  
**If empty:** Loop is disabled or not being initialized

### 2. Check Environment Variables
```bash
railway variables list | grep -E "NEWS_ANALYSIS|ANTHROPIC"
```
**Must have:**
- `ANTHROPIC_API_KEY` = `sk-ant-...` (Claude AI key)
- `NEWS_ANALYSIS_ENABLED` = "1" or unset (defaults to "1")

### 3. Check Current Deployment
```bash
railway status
```
**Should show:** Commit 878044f or later (includes debug endpoint)

### 4. Test New Debug Endpoint
```bash
curl https://ghost-protocol-production.up.railway.app/api/v3/debug/features/ZEC
```
**Expected output:**
```json
{
  "ok": true,
  "symbol": "ZEC",
  "sentiment": {
    "status": "success",
    "signals": [
      {"name": "news_sentiment_score", "value": 0.0 or >0.0, "source": "ghost_news_brain or rss"},
      {"name": "social_sentiment", "value": 0.0, "source": "neutral"}
    ],
    "working": true,
    "has_real_data": true or false
  },
  "world_context": {
    "spy_price": 598.45,  // Should be >$400, not null
    "vix_level": 14.23,   // Should be >10, not null
    "market_regime": "normal",
    "working": true
  },
  "orchestrator_health": {
    "total_pillars": 6,
    "healthy_pillars": 6,
    "pillar_status": {...}
  },
  "verdict": {
    "sentiment_engine_working": true,
    "world_context_working": true,
    "all_pillars_healthy": true
  }
}
```

### 5. Test Verification Script (When Deployed)
```bash
railway run python3 /app/check_zec_features.py
```
**Note:** Script committed (b98a747) but not deployed yet

---

## 🔧 Fix Ghost News Brain Loop

### Option A: Environment Variables Missing

**If `ANTHROPIC_API_KEY` not set:**
```bash
railway variables set ANTHROPIC_API_KEY=sk-ant-...
```

**If `NEWS_ANALYSIS_ENABLED` is "0":**
```bash
railway variables set NEWS_ANALYSIS_ENABLED=1
```

**Then redeploy:**
```bash
railway up
```

### Option B: Database Tables Missing

**Check if tables exist:**
```bash
railway run psql -c "\dt news_analysis"
railway run psql -c "\dt guardian_alerts"
```

**If tables missing, create them:**
- Tables are auto-created by Ghost News Brain on first run
- Or manually create via SQL migration

### Option C: Import/Code Error

**Check for Python errors:**
```bash
railway logs --tail 1000 | grep "Traceback" -A 10
railway logs --tail 1000 | grep "Error" | head -20
```

**Common errors:**
- `ModuleNotFoundError: core.intelligence`
- `AttributeError: get_news_brain`
- `asyncpg.exceptions.UndefinedTableError`

### Option D: Loop Not Being Called

**Add more logging (already committed in 878044f):**
- Debug endpoint shows what's actually working
- Will prove if sentiment/world context return real data

---

## ✅ Success Criteria (PROOF REQUIRED)

### Sentiment Engine Working:
- [ ] `news_sentiment_score` > 0.0 (not dummy data)
- [ ] Source = "ghost_news_brain" (not "neutral")
- [ ] Railway logs show: `📰 News Analysis Loop: STARTING`
- [ ] Database has records: `SELECT COUNT(*) FROM news_analysis;` > 0

### World Context Working:
- [ ] `spy_price` > $400 (not NULL)
- [ ] `vix_level` > 10 (not NULL)
- [ ] Source = "yfinance" or "price_quorum"
- [ ] Debug endpoint shows `working: true`

### Ghost News Brain Working:
- [ ] Railway logs show: `📰 Running automatic news analysis...`
- [ ] Database has analysis: `SELECT * FROM news_analysis ORDER BY created_at DESC LIMIT 5;`
- [ ] get_cached_analysis() returns data (not empty)
- [ ] Predictions integrate news (not news-blind)

### All Pillars Healthy:
- [ ] `/api/dev/features/diagnostic` shows 6/6 pillars
- [ ] No "DISABLED" warnings in orchestrator
- [ ] Feature count = 75+ (was 73 with 4 pillars)

---

## 🚀 Next Steps

### Immediate (Priority 1):
1. **Run verification commands above** - Get logs and environment variables
2. **Check Railway dashboard** - What commit is deployed?
3. **Test debug endpoint** - Does it work? What does it show?
4. **Fix missing environment variables** - Set ANTHROPIC_API_KEY if needed

### Short-Term (Priority 2):
5. **Wait for Railway redeploy** - Should pick up commits b98a747, 878044f
6. **Run check_zec_features.py** - Once deployed
7. **Verify sentiment scores** - Real or dummy?
8. **Verify SPY/VIX values** - Real or NULL?

### Medium-Term (Priority 3):
9. **Get Ghost News Brain running** - Fix whatever is blocking it
10. **Verify news integration** - Show predictions using news data
11. **Full end-to-end test** - Create prediction, show all features

---

## 📊 Current Deployment Status

**Commits:**
- ✅ f3ac7f4: Sentiment/world context fixes
- ✅ 583a1f2, 6922792: V2 filter fixes (VERIFIED WORKING)
- ⚠️ b98a747: Verification script (NOT DEPLOYED YET)
- ⚠️ 878044f: Debug endpoint (NOT DEPLOYED YET)

**Railway Status:**
- Auto-deploy from GitHub: Enabled
- Current commit: Unknown (check Railway dashboard)
- Deployment lag: Yes (b98a747 not accessible yet)

**Production Status:**
- V2 filter: ✅ WORKING (user verified)
- Sentiment engine: ⚠️ Code deployed, not verified
- World context: ⚠️ Code deployed, not verified
- Ghost News Brain: ❌ Loop not running
- Predictions: 251 in memory, database connected

---

## 🎯 What User Needs to Provide

**Required Information:**
1. Output of `railway logs --tail 1000 | grep "news_analysis_start_failed"`
2. Output of `railway variables list | grep -E "NEWS_ANALYSIS|ANTHROPIC"`
3. Output of `railway status` (what commit is deployed?)
4. Output of debug endpoint: `/api/v3/debug/features/ZEC`

**After Providing Info:**
- Agent can diagnose exact cause of Ghost News Brain failure
- Agent can provide specific fix (set env var, create tables, etc.)
- Agent can verify sentiment/world context with real data
- Agent can mark all issues as RESOLVED with proof

---

## 📝 Summary

**Fixed & Verified:**
- ✅ V2 filter (DASH/LRC blocked, ZEC allowed)

**Fixed but Not Verified:**
- ⚠️ Sentiment engine (needs proof of real data)
- ⚠️ World context (needs proof of real SPY/VIX)

**Not Working:**
- ❌ Ghost News Brain loop (not starting, needs investigation)

**Next Actions:**
- 🔴 User: Run verification commands above
- 🔴 User: Provide logs and environment variables
- 🔴 Agent: Diagnose Ghost News Brain failure
- 🔴 Agent: Fix Ghost News Brain loop
- 🔴 Agent: Verify sentiment/world context with PROOF
