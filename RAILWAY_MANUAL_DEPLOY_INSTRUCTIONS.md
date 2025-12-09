# 🚨 RAILWAY MANUAL DEPLOYMENT REQUIRED

## Current Situation

**Railway Auto-Deploy is BROKEN**- Despite successful builds, Railway keeps deploying
old code.

-**Current Railway Deployment**: `05d4b89e` (231 routes) ❌

- **Latest GitHub Commit**: `f03e4b4` (263 routes) ✅
- **Health Check**: Passing ✅
- **Problem**: Railway webhook not pulling latest commits from GitHub

______________________________________________________________________

## 🎯 MANUAL DEPLOYMENT STEPS

### Option 1: Railway Dashboard (RECOMMENDED)

1. **Go to Railway Dashboard**- URL: <<<<<https://railway.app/dashboard>>>>>
   - Project:**tender-benevolence**- Service:**web**1.**Navigate to Settings**- Click**Settings**tab (not Deployments)
   - Scroll to**GitHub Repo**- Current: `seancole713-source/GHOST` on branch `main`

1.**Disconnect & Reconnect GitHub**- Click**Disconnect**button

- Confirm disconnect
- Click**Connect GitHub Repo**- Select: `seancole713-source/GHOST`
- Branch: `main`
- Root Directory: `/` (leave empty)

1.**Trigger New Deployment**- This should automatically start a new deployment

- Watch Build Logs for commit hash - should show `f03e4b4`
- Build should take ~2 minutes

1.**Verify Deployment**```bash
   curl -s <<<<<https://web-production-8e9a0.up.railway.app/openapi.json>>>>> | \
     python3 -c "import sys,json; print(f\"Routes: {len(json.load[sys.stdin]('paths'))}\")"

   ```text**Expected**: `Routes: 263`

______________________________________________________________________

### Option 2: Force Deploy from Specific Commit

1. **Go to Deployments Tab**- Click**Deployments**- You'll see list of past deployments

1.**Look for Commit `f03e4b4`**- Look for: "feat: add UI alias endpoints for missing panels"

- If you see it: Click**⋮**→**Redeploy**→ ✅**Clear build cache**→ Confirm
- If you DON'T see it: Railway hasn't fetched latest commits (proceed to Option 3)

______________________________________________________________________

### Option 3: Force Railway to Fetch Latest Commits

If Railway isn't showing commit `f03e4b4` in the deployments list:

1.**Make a tiny change to force webhook**```bash

   cd /Users/studio713/Desktop/GHOST

# Add a comment to railway.toml

   echo "# Force Railway sync: $(date)" >> railway.toml

# Commit and push

   git add railway.toml
   git commit -m "trigger: force Railway to sync latest commits"
   git push origin main

   ```text

1.**Wait 30 seconds**, then check Railway dashboard

   - Should see new deployment starting
   - Check Build Logs for commit message "trigger: force Railway to sync"

1. **If STILL not deploying**:

   - Go to Settings → GitHub Repo → Click **"Sync Now"**(if available)
   - Or disconnect/reconnect GitHub (Option 1)


______________________________________________________________________

## 🔍 VERIFICATION COMMANDS

After deployment shows**"Active"**status:

```bash

BASE=<<<<<https://web-production-8e9a0.up.railway.app>>>>>

# 1. Check total routes (should be 263)

echo "Total Routes:"
curl -s $BASE/openapi.json | python3 -c "import sys,json; print(len(json.load(sys.stdin)['paths']))"

# 2. Test alias endpoints

echo -e "\nAlias Endpoints:"
curl -s $BASE/api/agent/decide | jq .message
curl -s $BASE/api/sources/status | jq .throttled
curl -s $BASE/api/market/movers -w " (HTTP %{http_code})\n" -o /dev/null

# 3. Test news endpoints

echo -e "\nNews Endpoints:"
curl -s $BASE/api/news?limit=2 | jq 'length'
curl -s $BASE/api/news/recent?minutes=120 | jq 'length'

# 4. Check which commit is deployed

echo -e "\nDeployed Endpoints:"
curl -s $BASE/openapi.json | python3 -c "
import sys,json
paths = json.load(sys.stdin)['paths']
news = [p for p in paths if '/news' in p]
alias = [p for p in paths if p in ['/api/agent/decide', '/api/sources/status', '/api/market/movers']]
print(f'News endpoints: {len(news)}')
print(f'Alias endpoints: {len(alias)}')
if len(alias) == 3 and len(news) >= 3:
    print('✅ NEW CODE DEPLOYED (f03e4b4)')
else:
    print('❌ OLD CODE STILL DEPLOYED')
"

```text**Success Criteria:**- ✅ Total routes:**263**(not 231)

- ✅ News endpoints:**3+**- ✅ Alias endpoints:**3**(decide, sources/status, market/movers)
- ✅ All test commands return HTTP 200


______________________________________________________________________

## 📊 EXPECTED VS ACTUAL

| Metric | Old Code (Current) | New Code (Expected) |
|--------|-------------------|---------------------| | Total Routes | 231 ❌ | 263 ✅ | |
News Endpoints | 0 ❌ | 3+ ✅ | | Alias Endpoints | 0 ❌ | 4 ✅ | | UI Panels Working |
~7/12 ❌ | 12/12 ✅ |

______________________________________________________________________

## 🐛 TROUBLESHOOTING

### If deployment keeps failing healthcheck

1.**Check Deploy Logs**for Python errors:


   ```python

   ImportError: cannot import name 'news_router'
   ModuleNotFoundError: No module named 'routes'

   ```text

1.**Verify routes/ directory exists in Railway**:

   - Railway should have copied `/app/routes/news_routes.py`
   - Check Build Logs for "COPY . /app"

1. **Check railway.toml start command**:


   ```toml

   startCommand = "python3 wolf_app.py"  # Should NOT have PORT=8444

   ```text

   Railway provides `$PORT` via environment variable automatically.

### If alias endpoints return 404

This means Railway deployed a commit BEFORE `f03e4b4`. You need to:

1. Manually redeploy commit `f03e4b4` (Option 2 above)
2. Or reconnect GitHub to force latest fetch (Option 1 above)


### If you see "Application startup complete" but still 404s

The code loaded successfully but doesn't have the new endpoints. This confirms Railway
is deploying old code. Follow Option 1 or Option 2 above.

______________________________________________________________________

## 📝 WHY IS THIS HAPPENING

Railway's GitHub webhook integration sometimes stops working:

- **Possible causes**:

  - GitHub webhook delivery failure
  - Railway service temporarily down
  - Repository permissions changed
  - Too many rapid commits (rate limiting)

- **Solution**: Manual intervention required (disconnect/reconnect GitHub)


______________________________________________________________________

**Current Status**: Railway is healthy but deploying wrong code\
**Required Action**: Manual redeploy of commit `f03e4b4` OR reconnect GitHub\
**Expected Outcome**: All 12 UI panels working with live data

Last Updated: October 14, 2025, 3:35 PM CDT
