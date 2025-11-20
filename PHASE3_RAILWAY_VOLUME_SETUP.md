# Phase 3: Railway Persistent Storage Setup

## Problem
The prediction database (`data/ghost_predictions.db`) lives in `/app/data/` which is **ephemeral** on Railway - it gets deleted on every redeploy.

## Solution
Add a Railway persistent volume for the `/data` directory.

---

## Setup Instructions

### Option 1: Railway Dashboard (Recommended)

1. **Open your Railway project**: https://railway.app/project/YOUR_PROJECT_ID

2. **Go to your service** (Ghost Sniper Bot)

3. **Click "Variables" tab**

4. **Add Volume Mount**:
   - Click "+ New Volume"
   - **Mount Path**: `/app/data`
   - **Size**: Start with 1GB (can increase later)
   - Click "Add"

5. **Verify Environment Variable**:
   - Check that `GHOST_PREDICT_DB=/app/data/ghost_predictions.db` is set
   - Already configured in `.env.railway` ✅

6. **Redeploy**:
   - Click "Deploy" → "Redeploy"
   - Watch logs to confirm database persists

---

### Option 2: Railway CLI

```bash
# Install Railway CLI if not already installed
npm i -g @railway/cli

# Login to Railway
railway login

# Link to your project (if not already linked)
railway link

# Add volume mount
railway volume add /app/data

# Verify volume is mounted
railway run env | grep RAILWAY_VOLUME

# Deploy
git push origin main
```

---

## Verification Steps

After deployment with volume mounted:

1. **Trigger a prediction**:
   ```bash
   curl -X POST https://YOUR_APP.up.railway.app/api/predict/force \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

2. **Check logs** for database write:
   ```
   [GHOST] Created prediction 123 for WOLF with 25 forecast points
   ```

3. **Redeploy the service** (trigger rebuild):
   ```bash
   railway up --detach
   ```

4. **Verify predictions survived**:
   ```bash
   curl https://YOUR_APP.up.railway.app/api/cockpit | jq '.ghost_2x.latest_predictions'
   ```

5. **Expected**: Should show predictions with timestamps from BEFORE redeploy

---

## Database Paths

| Environment | Path | Persistence |
|-------------|------|-------------|
| **Local** | `./data/ghost_predictions.db` | ✅ Git-ignored |
| **Railway (Before)** | `/app/data/ghost_predictions.db` | ❌ Ephemeral |
| **Railway (After)** | `/app/data/ghost_predictions.db` | ✅ **Persistent Volume** |

---

## Backup Strategy

Railway volume is backed up by the existing cron job:

```toml
# railway.toml
[[deploy.cron]]
schedule = "0 3 * * *"  # Daily at 3 AM UTC
command = "python scripts/railway_backup.py"
```

This backs up:
- `ghost_predictions.db` ← Prediction history
- `wolf.db` ← Main application database
- `watchlist.db` ← User watchlists

Keeps last 7 days, auto-cleans old backups.

---

## Troubleshooting

### Volume not mounting
```bash
# Check Railway logs
railway logs

# Look for:
# "Volume mounted at /app/data"
```

### Database permission errors
```bash
# Railway containers run as root by default
# Ensure write permissions in Dockerfile:
RUN mkdir -p /app/data && chmod 777 /app/data
```

### Database still resets
```bash
# Verify volume mount path matches DB path
railway run env | grep GHOST_PREDICT_DB
# Should output: GHOST_PREDICT_DB=/app/data/ghost_predictions.db

# Check volume mount point
railway run ls -la /app/data
# Should show: drwxr-xr-x ... /app/data (not empty after first prediction)
```

---

## Cost Impact

Railway pricing for volumes:
- **Free Tier**: 1GB included
- **Paid Tier**: $0.25/GB/month

For Ghost predictions:
- ~10KB per prediction
- 100 predictions/day = 1MB/day
- **1GB volume = ~3 years of predictions** 📈

---

## Next Steps After Setup

Once volume is mounted:

1. ✅ Predictions survive redeploys
2. ✅ GPS score shows accurate data
3. ✅ Cockpit displays historical predictions
4. 🚀 **Ready for Phase 4: Price Refresh Optimization**
