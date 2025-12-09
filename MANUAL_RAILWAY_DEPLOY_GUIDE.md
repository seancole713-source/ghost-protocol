# 🚨 MANUAL RAILWAY DEPLOYMENT REQUIRED

## Current Situation

- ✅ Code is **ready and working**(tested locally with 259 routes including 5 news

  routes)

- ✅ Latest commit**`b5b3a3e`**pushed to GitHub with router mounting
- ❌ Railway**auto-deployment is NOT working**- still serving old code (231 routes)

## ✅ Code Changes (Commit b5b3a3e)

Added module-level router mounting in `wolf_app.py`:

```python

# Line 187-191

from routes.news_routes import news_router
APP.include_router(news_router, prefix="/api/news", tags=["news"])

```text

This provides 3 news endpoints via the router:

- `GET /api/news` - Get news feed with optional symbol filtering
- `GET /api/news/recent` - Get news within time window (default 120 min)
- `GET /api/news/sentiment/{symbol}` - Get news sentiment for specific symbol


## 📋 STEP-BY-STEP MANUAL DEPLOYMENT

### Step 1: Access Railway Dashboard

1. Go to:**<<<<<https://railway.app/dashboard**>>>>>
2. Click on **GHOST**project
3. Click on**web**service


### Step 2: Force Clean Redeploy

1. Click**"Deployments"**tab
2. Find commit**`b5b3a3e`**(feat: mount news_router from routes/news_routes.py)
3. Click the**3 dots menu**(⋮) next to that deployment
4. Select**"Redeploy"**5.**✅ IMPORTANT**: Check the box **"Clear build cache"**6. Click**"Redeploy"**button to confirm


### Step 3: Verify Configuration (while build is running)

Go to**Settings**tab and confirm:**Start Command:**```text

python3 wolf_app.py

```text**Healthcheck Settings:**- Path: `/health`

- Timeout: `300` seconds (or 15 seconds as suggested)**Environment Variables:**- PORT should be provided automatically by Railway
- All other vars (OPENAI_API_KEY, POLYGON_API_KEY, etc.) should be set


### Step 4: Watch Build Logs

Click on**"Build Logs"**tab and wait for:

```text

✅ Successfully installed ... feedparser-6.0.11 ...
✅ === Successfully Built! ===
✅ Build time: ~100-120 seconds

```text

### Step 5: Watch Deploy Logs

Click on**"Deploy Logs"**tab and verify:

```text

✅ Starting Container
✅ [GHOST INIT] ...
✅ INFO: Started server process [1]
✅ INFO: Application startup complete
✅ INFO: Uvicorn running on <<<<<http://0.0.0.0:8080>>>>>

```text**Note**: Railway provides PORT via environment variable, so it might be 8080 or another

port (NOT 8444).

### Step 6: Watch HTTP Logs

Click on **"HTTP Logs"**tab and look for:

```text

✅ INFO: GET /health HTTP/1.1 200 OK

```text

If you see healthcheck failures:

```text

❌ INFO: GET /health HTTP/1.1 404 Not Found (or timeout)

```text

Then temporarily disable healthcheck:

1. Settings → Deploy → Healthcheck Path → Set to `/` or leave empty
2. Redeploy again


### Step 7: Verify Deployment (after "Active" status)

Run these commands locally to test:

```bash

BASE=<<<<<https://web-production-8e9a0.up.railway.app>>>>>

# Check OpenAPI schema

curl -s $BASE/openapi.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Total routes:
{len(d[\"paths\"])}'); news=[p for p in d['paths'] if p.startswith('/api/news')]; print(f'News routes: {len(news)}');
[print(f' {p}') for p in news]"

# Test endpoints

for p in /api/news /api/news/recent; do echo $p; curl -s -o /dev/null -w "%{http_code}\n" $BASE$p; done

# Or use the check script

./check_railway_status.sh

```text**Expected Results:**```text

✅ Total routes: 259 (not 231)
✅ News routes: 3
    /api/news
    /api/news/recent
    /api/news/sentiment/{symbol}
✅ /api/news → HTTP 200
✅ /api/news/recent → HTTP 200

```text

## 🔧 Troubleshooting

### If Build Fails**Check Build Logs for errors:**- Missing dependency: Check `requirements.txt`

- Python syntax error: Check latest commit
- Import error: Verify `routes/news_routes.py` exists in repo**Solutions:**- If missing file: Ensure `routes/news_routes.py` is in git


  (`git ls-files | grep routes/`)

- If import fails: Check Python version compatibility
- If dependency missing: Update `requirements.txt` and push


### If Deploy Fails (healthcheck timeout)**Problem**: Application starts but healthcheck fails

**Solutions:**1.**Temporarily disable healthcheck**:

   - Settings → Deploy → Healthcheck Path → **Leave empty**- Redeploy


1.**Check if app is listening on correct port**:

   - Deploy Logs should show: `Uvicorn running on <<<<<http://0.0.0.0:$PORT`>>>>>
   - Railway provides PORT via environment variable
   - wolf_app.py reads it: `port = int(os.getenv("PORT", "8444"))`

1. **Test health endpoint manually**:


   ```bash

   curl -v <<<<<https://web-production-8e9a0.up.railway.app/health>>>>>

   ```text

### If Still Shows 231 Routes After "Active"

**Problem**: Build succeeds, deploy succeeds, but old code is still running

**This means Docker image cache is stuck**:

1. Settings → Service → **"Clear Build Cache"**(if button exists)
2. Deployments → Latest →**"Redeploy"**again with cache cleared
3. If still fails: Try deploying from a different commit, then back to latest


### If GitHub Integration is Broken**Problem**: Commits `b5b3a3e`, `a31ce8b`, etc. don't appear in Deployments list

**Solution - Reconnect GitHub**:

1. Settings → Integrations → GitHub
2. Click **"Disconnect"**3. Click**"Connect GitHub"**4. Authorize Railway app
3. Select repository: `seancole713-source/GHOST`
4. Select branch: `main`
5. Enable**"Deploy on push"**8. Manually trigger**"New Deployment"**→**"From GitHub"**→ `main` branch


## 📊 Verification Checklist

After deployment is "Active", verify:

- [ ] OpenAPI schema shows 259 total routes (not 231)
- [ ] OpenAPI schema lists `/api/news`, `/api/news/recent`,


  `/api/news/sentiment/{symbol}`

- [ ] `GET /api/news` returns HTTP 200 with news items
- [ ] `GET /api/news/recent` returns HTTP 200 with time-filtered news
- [ ] `GET /health` returns HTTP 200 with `{"ok": true, "ts": ...}`
- [ ] Ghost Cockpit UI news panels load data (no "error loading data")


## 📄 Files to Reference

-**`routes/news_routes.py`**- Modular news router (exists, tracked by git)
-**`wolf_app.py`**- Main app with router mounting at lines 187-191
-**`railway.toml`**- Deployment config (healthcheck, start command)
-**`requirements.txt`**- Python dependencies (feedparser, dateutil, etc.)
-**`check_railway_status.sh`**- Quick verification script


## 🎯 Success Criteria

Deployment is successful when:

1. Railway dashboard shows commit `b5b3a3e` as**"Active"**2. Build logs show successful pip install
2. Deploy logs show "Application startup complete"
3. HTTP logs show "GET /health HTTP/1.1 200 OK"
4. OpenAPI schema returns 259 routes with 3 news endpoints
5. All news endpoints return HTTP 200 with valid JSON


______________________________________________________________________**Last Updated**: October 14, 2025, 3:15 PM CDT\
**Latest Commit**: `b5b3a3e` (feat: mount news_router from routes/news_routes.py at
module level)\
**Status**: ⏳ Awaiting manual Railway deployment with cache cleared
