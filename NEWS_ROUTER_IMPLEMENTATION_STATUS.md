# News API Routes - Implementation Summary

## ✅ What Was Accomplished

### 1. Created Modular Router Structure

- **File**: `routes/news_routes.py`
- **Router**: `news_router = APIRouter()`
- **Endpoints Added**:
  - `GET /api/news` - Get aggregated news feed (symbol filter, limit 1-200)
  - `GET /api/news/recent` - Get news within time window (symbol filter, minutes 1-1440)
  - `GET /api/news/sentiment/{symbol}` - Get sentiment analysis for specific symbol

### 2. Mounted Router in wolf_app.py

- **Location**: Lines 188-200
- **Code**:

```python
_NEWS_ROUTER_MOUNTED = False
_NEWS_ROUTER_ERROR = None
try:
    from routes.news_routes import news_router
    APP.include_router(news_router, prefix="/api/news", tags=["news"])
    _NEWS_ROUTER_MOUNTED = True
except Exception as e:
    _NEWS_ROUTER_ERROR = str(e)
    # Logs error
```

### 3. Local Verification ✅

- **Tool**: `inspect_routes.py`
- **Result**: Successfully imported wolf_app.APP
- **Routes Registered**: 256 total (added 3 news routes)
- **Confirmed Routes**:
  - `GET /api/news`
  - `GET /api/news/recent`
  - `GET /api/news/sentiment/{symbol}`

### 4. Git Commits

1. **Commit 2e98537**: "feat(api): add and mount /api/news routes with modular router
   pattern"

   - Created routes/news_routes.py
   - Mounted router in wolf_app.py
   - Added inspect_routes.py tool

2. **Commit 388efd8**: "debug: add /debug/router_status endpoint to diagnose news router
   mounting"

   - Added `/debug/router_status` endpoint
   - Returns: news_router_mounted status, errors, file existence, route count

Both commits pushed to `origin/main` successfully.

______________________________________________________________________

## ❌ Current Issue: Railway Not Deploying Latest Code

### Symptoms

- News routes return **HTTP 404** on production
- Debug endpoint (`/debug/router_status`) returns **HTTP 404**
- Other endpoints still work:
  - `/api/agent/decisions` → HTTP 200 ✅
  - `/api/snapshot` → HTTP 200 ✅
  - `/health` → HTTP 200 ✅

### OpenAPI Schema Check

```bash
curl -s https://web-production-8e9a0.up.railway.app/openapi.json
```

- Only shows 1 news-related route: `/api/watcher/ticker_news` (old endpoint)
- Does NOT show our new `/api/news` routes

### Possible Causes

1. **Railway hasn't picked up latest commits** (most likely)

   - Git shows latest commit 388efd8 is on `origin/main`
   - Railway may still be serving an older build

2. **routes/ directory not deployed**

   - Check if `routes/` is in `.gitignore`
   - Check Railway build logs

3. **Import error on Railway**

   - Router mounting may be failing silently
   - Need to check Railway runtime logs

______________________________________________________________________

## 🔧 Next Steps to Fix

### Immediate Actions

1. **Check Railway Dashboard** (https://railway.app)

   - Go to GHOST project
   - Check "Deployments" tab
   - Verify latest commit (388efd8) was deployed
   - If not deployed, click "Redeploy" button

2. **Check Build Logs**

   - Look for errors during `pip install` or file copying
   - Verify `routes/news_routes.py` was included in build

3. **Check Runtime Logs**

   - Look for import errors: `Failed to mount news router`
   - Check if routes/ directory exists on Railway filesystem

4. **Force Rebuild** (if needed)

   ```bash
   echo "# Force rebuild $(date)" >> railway.toml
   git add railway.toml
   git commit -m "force: trigger Railway rebuild"
   git push origin main
   ```

### Verification Commands (After Railway Deploys)

```bash
BASE=https://web-production-8e9a0.up.railway.app

# 1. Check debug endpoint
curl -s $BASE/debug/router_status | jq

# 2. Test news endpoints
curl -s "$BASE/api/news?limit=5" | jq
curl -s "$BASE/api/news/recent?minutes=120" | jq
curl -s "$BASE/api/news/sentiment/WOLF" | jq

# 3. Verify OpenAPI schema
curl -s $BASE/openapi.json | jq '.paths | keys | .[] | select(contains("news"))'
```

______________________________________________________________________

## 📊 Current State

| Component | Status | Notes | |-----------|--------|-------| | routes/news_routes.py |
✅ Created | 3 endpoints, proper error handling | | wolf_app.py mounting | ✅ Added |
Lines 188-200, with try/except | | Local import test | ✅ Pass | 256 routes registered
including 3 news | | Git commits | ✅ Pushed | 2e98537, 388efd8 on origin/main | |
Railway deployment | ❌ Stale | Still serving old code | | Production /api/news | ❌ 404 |
Not found | | Production /debug/router_status | ❌ 404 | Not found |

______________________________________________________________________

## 🎯 Success Criteria

When fixed, you should see:

- ✅ `/debug/router_status` returns `{"news_router_mounted": true}`
- ✅ `/api/news` returns `{"news": [...], "count": N, "status": "live"}`
- ✅ `/api/news/recent` returns recent articles within time window
- ✅ `/api/news/sentiment/WOLF` returns sentiment metrics
- ✅ OpenAPI schema includes all 3 news routes
- ✅ Ghost Cockpit UI news panels load data

______________________________________________________________________

## 📁 Files Modified

1. **routes/__init__.py** (new)

   - Package init file for routes module

2. **routes/news_routes.py** (new)

   - 368 lines
   - Comprehensive news feed aggregation
   - Fallback to RSS if database unavailable

3. **wolf_app.py** (modified)

   - Added router mounting code (lines 188-200)
   - Added debug endpoint (lines 202-213)
   - Net change: +14 lines

4. **inspect_routes.py** (new)

   - Debug tool to verify route registration
   - Works locally, confirms 256 routes with 3 news routes

______________________________________________________________________

## 🔍 Debug Endpoint Response (Expected)

Once deployed, `/debug/router_status` should return:

```json
{
  "news_router_mounted": true,
  "news_router_error": null,
  "routes_dir_exists": true,
  "news_routes_file_exists": true,
  "total_routes": 256,
  "news_routes": [
    "/api/news",
    "/api/news/recent",
    "/api/news/sentiment/{symbol}"
  ]
}
```

______________________________________________________________________

**Last Updated**: October 14, 2025 **Current Railway URL**:
https://web-production-8e9a0.up.railway.app **Latest Commit**: 388efd8 (on origin/main,
waiting for Railway deployment)
