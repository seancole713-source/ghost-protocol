P# 🚀 Railway Deployment Guide for GHOST

## Quick Deploy (Manual - Recommended for First Deploy)

Since Railway CLI requires browser auth (which doesn't work smoothly in Codespaces), use
the Railway dashboard:

### Step 1: Push to GitHub

```bash
git add -A
git commit -m "Railway deployment ready"
git push origin main
```

### Step 2: Deploy via Railway Dashboard

1. **Go to Railway**: https://railway.app/dashboard
2. **New Project** → **Deploy from GitHub repo**
3. **Select**: `seancole713-source/GHOST`
4. **Branch**: `main`
5. **Settings**:
   - Builder: **Dockerfile** (auto-detected from `railway.json`)
   - Start Command: `python wolf_app.py`
   - Healthcheck Path: `/health`
   - Port: Railway auto-detects from `PORT` env var

### Step 3: Verify Variables Are Set

Go to **Variables** tab and confirm these exist (already configured):

- ✅ POLYGON_API_KEY
- ✅ ALPHAVANTAGE_API_KEY
- ✅ OPENAI_API_KEY
- ✅ TELEGRAM_BOT_TOKEN (if using alerts)
- ✅ TELEGRAM_CHAT_ID (if using alerts)

### Step 4: Deploy!

Railway will:

1. Clone your repo
2. Build using Dockerfile
3. Deploy to a public URL
4. Run healthcheck against `/health`

### Step 5: Get Your URL

After deployment completes:

- **Settings** → **Networking** → **Generate Domain**
- Copy the URL (format: `ghost-production-xxxx.up.railway.app`)

### Step 6: Test Live

```bash
# Replace with your actual Railway URL
RAILWAY_URL="https://your-app.up.railway.app"

curl $RAILWAY_URL/health
curl $RAILWAY_URL/api/cockpit
curl "$RAILWAY_URL/api/top_movers?threshold=7.0"
```

______________________________________________________________________

## Alternative: CLI Deploy (if you have Railway token)

If you have a `RAILWAY_TOKEN`:

```bash
export RAILWAY_TOKEN="your_token_here"
railway link  # Link to existing project
railway up    # Deploy
railway open  # Open dashboard
```

______________________________________________________________________

## Files Created for Railway

- ✅ `Dockerfile` - Multi-stage build optimized for Railway
- ✅ `railway.json` - Railway config (builder, healthcheck, restart policy)
- ✅ `requirements.txt` - Python dependencies (existing)

______________________________________________________________________

## Post-Deploy Checks

Once deployed, verify:

```bash
# Health
curl https://your-app.up.railway.app/health

# Price provider (should NOT be "prev-close" during market hours)
curl https://your-app.up.railway.app/api/price/diagnostics | jq .provider

# Trigger forecast generation
curl -X POST https://your-app.up.railway.app/api/advisor_refresh

# Scan watchlist for movers
curl -X POST "https://your-app.up.railway.app/api/watchlist/scan?threshold=7.0&limit=30"

# Check top movers
curl https://your-app.up.railway.app/api/top_movers
```

______________________________________________________________________

## Troubleshooting

### Build fails

- Check Railway logs: **Deployments** → Click latest → **View Logs**
- Common: missing dependencies in `requirements.txt`

### Health check fails

- Increase timeout in `railway.json` (`healthcheckTimeout: 300`)
- Check `/health` endpoint returns `{"ok": true}`

### Environment variables not working

- Verify in Railway **Variables** tab
- Restart deployment after adding new vars

### Port issues

- Railway sets `PORT` automatically - don't hardcode 5000
- `wolf_app.py` reads `PORT` from env (already configured)

______________________________________________________________________

## Next Steps After Deploy

1. **Custom Domain** (optional): Settings → Domains → Add Custom Domain
2. **Monitoring**: Use Railway metrics dashboard
3. **Auto-deploys**: Settings → Enable "Auto-deploy on push to main"
4. **Scale**: Settings → Adjust resources if needed

______________________________________________________________________

**Your Railway vars are already set!** Just push to GitHub and deploy via dashboard.
