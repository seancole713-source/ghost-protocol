# 🚂 GHOST RAILWAY DEPLOYMENT GUIDE

## 🚀 One-Command Deploy

Run this to set up everything automatically:

```bash
./deploy_ghost.sh

```text

This script will:

1. ✅ Install Railway CLI (if needed)
2. ✅ Authenticate with Railway
3. ✅ Link to ghost-protocol project
4. ✅ Set environment variables
5. ✅ Deploy Ghost
6. ✅ Test health endpoint
7. ✅ Show deployment URL


**First-time setup: ~5 minutes**______________________________________________________________________

## 📋 Available Scripts

###**1. Initial Deployment**```bash

./deploy_ghost.sh

```text

Complete setup + deployment. Use this the first time.

###**2. Quick Redeploy**```bash

./redeploy.sh "Your commit message"

```text

Git commit + push + Railway deploy in one command.

###**3. Railway Management**```bash

./railway_manage.sh [command]

```text

Commands:

- `deploy` - Deploy to Railway
- `logs` - View live logs
- `status` - Check deployment status
- `url` - Get deployment URL
- `health` - Test health endpoint
- `vars` - List environment variables
- `restart` - Restart service
- `open` - Open in browser
- `restore` - Restore position data


______________________________________________________________________

## 🎯 Common Workflows

###**Deploy After Code Changes**```bash

./redeploy.sh "Fixed WOLF price provider"

```text

###**Check If Ghost Is Running**```bash

./railway_manage.sh health

```text

###**Watch Live Logs**```bash

./railway_manage.sh logs

```text

###**Get Your URL**```bash

./railway_manage.sh url

```text

###**Restore Position Data**```bash

./railway_manage.sh restore

```text

______________________________________________________________________

## 🔧 Manual Railway Commands

If you prefer using Railway CLI directly:

```bash

# Deploy

railway up --detach

# Watch logs

railway logs

# Check status

railway status

# Get URL

railway domain

# Open in browser

railway open

# Set variable

railway variables set KEY="value"

# List variables

railway variables

# Restart

railway restart

```text

______________________________________________________________________

## 🌐 Your Deployment

After running `./deploy_ghost.sh`, you'll get:**URL**: `https://ghost-protocol-production.up.railway.app`\
(or similar)

**Endpoints**:

- UI: `https://[your-url]/`
- Health: `https://[your-url]/health`
- Cockpit: `https://[your-url]/api/cockpit`
- AI Memory: `https://[your-url]/ai/memory/stats`


______________________________________________________________________

## 🔐 Environment Variables

These are automatically set by `deploy_ghost.sh`:

```bash

GHOST_API_TOKEN              # API authentication
POLYGON_API_KEY              # Stock price data
ALPHAVANTAGE_API_KEY         # Backup price provider
TELEGRAM_BOT_TOKEN           # Alert notifications
TELEGRAM_CHAT_ID             # Your Telegram chat
GHOST_FOCUS_TICKER           # Trading symbol (WOLF)
WOLF_PERSIST_MODE            # Database mode (sqlite)
SIM_MODE                     # Live trading (0)

```text

To update:

```bash

railway variables set KEY="new_value"

```text

______________________________________________________________________

## 📊 Monitoring

### **Live Logs**```bash

railway logs

```text

###**Health Check**```bash

curl <<<<<https://[your-url]/health>>>>>

```text

Expected response:

```json

{"ok": true, "ts": 1759543749}

```text

###**Detailed Health**```bash

curl <<<<<https://[your-url]/health/detailed>>>>>

```text

Shows:

- Position data
- AI memory stats
- Price provider status
- Database health


______________________________________________________________________

## 🔄 Auto-Deploy

Railway automatically deploys when you push to `main` branch:

```bash

git add .
git commit -m "Updated feature"
git push origin main

```text

Railway detects the push and redeploys within 3-5 minutes.

Or use the quick script:

```bash

./redeploy.sh "Updated feature"

```text

______________________________________________________________________

## 💾 Database Persistence

Ghost uses Railway's persistent storage for:

- `wolf.db` - Position data
- `ai_memory.db` - AI decisions
- `ghost_ai.db` - Trading history**Automatic backups**: Daily at 3 AM UTC (configured in `railway.toml`)


______________________________________________________________________

## 🐛 Troubleshooting

### **Deployment fails with "ModuleNotFoundError"**✅ Already fixed! `Procfile` and `nixpacks.toml` ensure dependencies install

###**Health check fails**```bash

# Check logs

railway logs --tail 100

# Check status

railway status

# Restart

railway restart

```text

###**Can't get URL**```bash

railway domain

```text

###**Environment variables missing**```bash

# List current

railway variables

# Set missing ones

railway variables set KEY="value"

```text

###**Position data lost**```bash

# Restore WOLF position

./railway_manage.sh restore

```text

______________________________________________________________________

## 📁 Deployment Files

These files control Railway deployment:

-**`Procfile`**- Tells Railway to run `python main.py`
-**`nixpacks.toml`**- Installs Python dependencies
-**`railway.toml`**- Health checks, restarts, cron jobs
-**`requirements.txt`**- Python packages


Don't delete these!

______________________________________________________________________

## 🎯 Quick Reference

| Task | Command | |------|---------| |**First deploy**| `./deploy_ghost.sh` | |**Redeploy**| `./redeploy.sh "message"`
| |**Watch logs**| `./railway_manage.sh logs`
| |**Check health**| `./railway_manage.sh health` | |**Get URL**|
`./railway_manage.sh url` | |**Restart**| `./railway_manage.sh restart` | |**Restore
position**| `./railway_manage.sh restore` |

______________________________________________________________________

## ✅ Success Checklist

After deployment, verify:

- [ ] `./railway_manage.sh health` returns `{"ok": true}`
- [ ] `./railway_manage.sh url` shows your URL
- [ ] Visit URL in browser - UI loads
- [ ] `/api/cockpit` returns JSON data
- [ ] `/ai/memory/stats` shows decision count
- [ ] `./railway_manage.sh restore` succeeds


If all checked,**Ghost is running 24/7 on Railway!**🎉

______________________________________________________________________

## 🆘 Need Help

1. Check logs: `./railway_manage.sh logs`
2. Check status: `./railway_manage.sh status`
3. Test health: `./railway_manage.sh health`
4. Restart service: `./railway_manage.sh restart`


______________________________________________________________________**Made with ❤️ for Ghost Trading System**
