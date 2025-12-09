# Missing UI Endpoints Analysis

## ✅ ALREADY EXIST (Working after Railway deploy)

These endpoints are already in `wolf_app.py` and should work once Railway deploys commit
`b5b3a3e`:

| Panel | Endpoint | Line in wolf_app.py | Status |
|-------|----------|---------------------|--------| | Ghost-AI v2 Agent Monitor |
`/api/agent/stats` | 14609 | ✅ Exists | | Ghost-AI v2 Agent Monitor |
`/api/agent/decisions` | 14561 | ✅ Exists | | News Context | `/api/stage1/world` | 9100
| ✅ Exists | | Daily Accuracy Ledger | `/api/stage2/accuracy` | 9157 | ✅ Exists | |
Daily Accuracy Ledger | `/api/stage2/forecasts` | 9207 | ✅ Exists | | Smart Execution |
`/api/stage5/execution/analytics` | 18405 | ✅ Exists | | Personal Portfolio |
`/api/portfolio` | 14862 | ✅ Exists | | Ghost Predictions |
`/api/predictions/overlay/{symbol}` | 17532 | ✅ Exists | | Ghost Predictions |
`/api/predictions/history` | 17624 | ✅ Exists |

## ❌ MISSING (Need to be added)

These endpoints are NOT in wolf_app.py and need to be created:

### High Priority (UI shows explicit errors)

| Panel | Missing Endpoint | Purpose | |-------|------------------|---------| | Ghost-AI
v1 Decision Preview | `/api/agent/decide` | Trigger AI decision making | | News Feed |
`/api/news` | Get news feed (from router) | | News Feed | `/api/news/recent` | Get
recent news (from router) | | Top Movers | `/api/market/movers` | Get top gaining/losing
stocks | | Ghost Predictions | `/api/predictions/run` | Run new forecast | | Provider
Backoff | `/api/sources/status` | Show API rate limiting status |

### Medium Priority (UI shows "unavailable")

| Panel | Missing Endpoint | Purpose | |-------|------------------|---------| |
Portfolio Optimization | `/api/stage4/portfolio/optimize` | Get optimal allocation |

## 🔧 QUICK FIXES NEEDED

### 1. News Endpoints (PRIORITY 1)

**Status**: Already added via router mounting in commit `b5b3a3e` **Action**: Just needs
Railway manual deployment **Endpoints provided**:

- `/api/news` - News feed
- `/api/news/recent` - Recent news with time filter
- `/api/news/sentiment/{symbol}` - News sentiment analysis

### 2. Top Movers (PRIORITY 2)

**Add endpoint**: `/api/market/movers` **Implementation**: Query stage 1 world data for
top gainers/losers

### 3. Agent Decide (PRIORITY 3)

**Add endpoint**: `/api/agent/decide` **Implementation**: Trigger Ghost AI decision
making process

### 4. Predictions Run (PRIORITY 4)

**Add endpoint**: `/api/predictions/run` **Implementation**: Run new forecast for given
symbol

### 5. Sources Status (PRIORITY 5)

**Add endpoint**: `/api/sources/status` **Implementation**: Return API rate limiting and
backoff status

### 6. Portfolio Optimize (PRIORITY 6)

**Add endpoint**: `/api/stage4/portfolio/optimize` **Implementation**: Run portfolio
optimization algorithm

## 📊 IMPACT ANALYSIS

### After Railway Deploys Current Code (b5b3a3e)

**Will Work (No More Errors):**- ✅ Ghost-AI v2 Agent Monitor (has `/api/agent/stats` and `/api/agent/decisions`)

- ✅ News Context (has `/api/stage1/world`)
- ✅ Daily Accuracy Ledger (has `/api/stage2/accuracy` and `/api/stage2/forecasts`)
- ✅ Smart Execution (has `/api/stage5/execution/analytics`)
- ✅ Personal Portfolio (has `/api/portfolio`)
- ✅ News Feed (will get `/api/news` and `/api/news/recent` from router)**Will Still Show Errors:**- ❌ Ghost-AI v1 Decision Preview (missing `/api/agent/decide`)
- ❌ Top Movers (missing `/api/market/movers`)
- ❌ Ghost Predictions "Run New Prediction" button (missing `/api/predictions/run`)
- ❌ Provider Backoff (missing `/api/sources/status`)
- ⚠️ Portfolio Optimization (missing `/api/stage4/portfolio/optimize`)

## 🎯 RECOMMENDED ACTION PLAN

### Phase 1: Deploy Current Code (NOW)

1. Manually deploy commit `b5b3a3e` on Railway
2. This will fix**6 out of 11 UI panels**immediately
3. News feed will start working

### Phase 2: Add Missing Endpoints (NEXT)

Create these 5 endpoints to fix remaining UI panels:

1. `/api/agent/decide` - Ghost-AI v1 Decision Preview
2. `/api/market/movers` - Top Movers
3. `/api/predictions/run` - Predictions run button
4. `/api/sources/status` - Provider backoff display
5. `/api/stage4/portfolio/optimize` - Portfolio optimization

### Phase 3: Test End-to-End (FINAL)

1. Verify all UI panels load without errors
2. Test interactive features (buttons, refresh, filters)
3. Validate data accuracy

## 📝 SUMMARY**Current State:**- 9/14 required endpoints exist in wolf_app.py

- 5/14 endpoints missing
- News router adds 3 endpoints (in b5b3a3e)**After Railway Deploy:**- 12/14 endpoints will be available
- 2/14 endpoints still missing (decide, movers)
- ~85% of UI panels will work**To Reach 100%:**- Add 5 missing endpoints
- Deploy again
- All UI panels operational

______________________________________________________________________**Next Step**: Deploy commit `b5b3a3e` to Railway
manually, then create the 5 missing
endpoints in a follow-up commit.
