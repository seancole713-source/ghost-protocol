# ⚡ Railway Cache Busting & Deployment Guide

## Problem

Railway may be caching old static assets (HTML, JS, CSS), causing the UI to show
outdated panel names even after code updates.

## Solutions

### 1. Force Railway to Rebuild (Recommended)

```bash
# Option A: Use Railway CLI
railway up --detach

# Option B: Git push to trigger rebuild
git add -A
git commit -m "🚀 UI updates + intelligence enhancements"
git push origin main
```

### 2. Add Version Query Parameters

Update static file references to include timestamps:

```html
<!-- Before -->
<link rel="stylesheet" href="/static/ghost.css" />
<script src="/static/ghost.js"></script>

<!-- After -->
<link rel="stylesheet" href="/static/ghost.css?v=20251012" />
<script src="/static/ghost.js?v=20251012" />
```

### 3. Update Railway Environment Variables

```bash
# Add a VERSION variable that updates on each deploy
railway variables set APP_VERSION=$(date +%Y%m%d%H%M%S)

# Force cache headers in FastAPI
railway variables set CACHE_CONTROL="no-cache, no-store, must-revalidate"
```

### 4. Clear Railway Cache

```bash
# SSH into Railway service and clear cache
railway run rm -rf /app/.cache /tmp/*

# Restart the service
railway restart
```

### 5. Update Nginx/Caddy Cache Headers (If Applicable)

If using a reverse proxy, ensure proper cache headers:

```
# Caddyfile
header /static/* {
    Cache-Control "no-cache, no-store, must-revalidate"
    Pragma "no-cache"
    Expires "0"
}
```

### 6. Browser Cache Busting

For immediate testing, bypass browser cache:

- **Chrome/Edge**: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
- **Firefox**: Ctrl+F5 (Windows/Linux) or Cmd+Shift+R (Mac)
- **Safari**: Cmd+Option+R
- **Or**: Open in Incognito/Private mode

## Verification Steps

After deploying:

1. **Check Railway logs**:

```bash
railway logs --tail 100
```

2. **Run verification script**:

```bash
./verify_railway_deployment.sh
```

3. **Test in browser**:

   - Open https://web-production-8e9a0.up.railway.app
   - Hard refresh (Ctrl+Shift+R)
   - Check browser DevTools → Network tab
   - Verify 200 OK for all assets
   - Check "Size" column - should show actual bytes, not "(disk cache)"

4. **Verify panel names**:

   - Look for: "🧠 Ghost Intelligence Engine" (not "Ghost-AI v1")
   - Look for: "🏛️ Market Pulse" (not "Market Status")
   - Look for: "🔮 Predictive Analytics" (not "48h Forecast")

## Permanent Fix

Add cache-busting middleware to wolf_app.py:

```python
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse
import time

@APP.middleware("http")
async def cache_control_middleware(request: Request, call_next):
    response = await call_next(request)
    
    # Add no-cache headers for static assets
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    
    return response
```

## Monitoring

Check if cache is being used:

```bash
# Check response headers
curl -I https://web-production-8e9a0.up.railway.app/static/index.html

# Should see:
# Cache-Control: no-cache
# X-Version: <timestamp>
```

## Rollback Plan

If deployment breaks:

```bash
# Rollback to previous deployment
railway rollback

# Or revert git commit
git revert HEAD
git push origin main
```

______________________________________________________________________

**Remember**: Always test in incognito/private browsing mode to avoid local browser
cache issues!
