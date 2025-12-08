# 🚀 Railway Deployment Fix Guide

## 🔴 Critical Issue: 502 Bad Gateway

Your Railway deployment is failing because the application cannot start properly.

---

## 📋 Step-by-Step Fix

### STEP 1: Check Railway Logs (MOST IMPORTANT)

1. Open Railway dashboard: <<<<<https://railway.app>>>>>
2. Select your `ghost-protocol` project
3. Click on your service
4. Go to **Deployments**tab
5. Click on the latest deployment
6. View**Build Logs**and**Deploy Logs**


**Look for:**- ❌ Python errors

- ❌ Missing dependencies
- ❌ Port binding issues
- ❌ Environment variable errors


---

### STEP 2: Add Environment Variables

Railway needs these environment variables set:

1. In Railway dashboard, go to your service
2. Click**Variables** tab
3. Add these variables:


```bash

# ============================================================================

# CRITICAL - Required for app to start

# ============================================================================

OPENAI_API_KEY=sk-proj-...
TELEGRAM_BOT_TOKEN=8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw
TELEGRAM_CHAT_ID=940596997

# ============================================================================

# DATA PROVIDERS - Required for trading

# ============================================================================

POLYGON_API_KEY=8VIvELVXiLG30K2l1348RzSurffLM0jR
ALPHAVANTAGE_API_KEY=3WNNLA81KS7BG4AK
COINGECKO_API_KEY=your-coingecko-key

# ============================================================================

# BROKER - Alpaca Trading

# ============================================================================

BROKER=alpaca
ALPACA_KEY_ID=PKVUMLL1V91W9Y5QCG77
ALPACA_SECRET_KEY=sw09z6TdIeXrs9G6fE5Lo9AayM44UmSWiEYcuXyk
ALPACA_PAPER=1
APCA_API_BASE_URL=<<<<<https://paper-api.alpaca.markets/v2>>>>>

# ============================================================================

# REDIS CACHE

# ============================================================================

CACHE_MODE=redis
REDIS_URL=rediss://default:AVriAAIncDJmNmUyNjFmMDRkMDE0YzE2OWNiOTY0MmYxZjcxMWYxNXAyMjMyNjY@comic-hookworm-23266.upstash.io:6379/0

# ============================================================================

# GHOST API

# ============================================================================

GHOST_API_TOKEN=edaa4eac-6455-4693-a745-142cb6deef03
GHOST_BASE_URL=<<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app>>>>>

# ============================================================================

# AI CONFIGURATION

# ============================================================================

AI_ON=1
AI_MODEL=gpt-4o-mini
AI_PROVIDER=openai
OPENAI_ORG_ID=org:jgG9PhOvU5uFkEPWKwe8Moa0
VECTOR_SOURCE=openai
VECTOR_STORE_ID=vs_68e8621d71e48191a85c49cb95bef3a0

# ============================================================================

# AGENTS

# ============================================================================

AGENTS_ENABLED=1
AGENTKIT_ENABLED=true
AGENT_MODEL=gpt-4o-mini
AGENT_POLICY=hybrid
AGENT_ROLE=diag_orchestrator
AUTO_FIXER_ENABLED=true

# ============================================================================

# MARKET DATA & PRICING

# ============================================================================

STOCKS_ENABLED=1
CRYPTO_ENABLED=1
STOCK_PRICE_SOURCE=polygon
CRYPTO_PRICE_SOURCE=coingecko
CRYPTO_QUORUM=coingecko,binance,coinbase
PRICE_TTL_S=120
PRICE_TTL_OPEN_S=300
PRICE_CACHE_TTL=60

# ============================================================================

# PREDICTIONS

# ============================================================================

PREDICT_STOCKS_ENABLED=1
PREDICT_STOCKS_ALLOW=AAPL,*
PRICE_REFRESH_ALLOW=AAPL,WOLF
CRYPTO_FORECAST_H=48
CRYPTO_LOOKBACK_H=96

# ============================================================================

# RISK MANAGEMENT

# ============================================================================

MAX_RISK_DRAWDOWN=0.05
RISK_MAX_DAILY_DD_PCT=5
RISK_MAX_POS_PCT=5
RISK_SL_PCT=3
RISK_TP_PCT=6
TARGET_WEEKLY_PROFIT_USD=300

# ============================================================================

# FEATURES

# ============================================================================

FUSE_DECISION_ON=1
MACRO_BRAIN_ON=1
NEWS_SENTIMENT_ON=1
FOCUS_WOLF_ONLY=0

# ============================================================================

# SECURITY

# ============================================================================

ADMIN_IP_ALLOWLIST=127.0.0.1,::1,0.0.0.0/0
TRUSTED_HOSTS=*.railway.app,localhost,127.0.0.1
ALLOWED_ORIGINS=<<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app,http://localhost:8444,http://127.0.0.1:8444>>>>>
FORWARDED_ALLOW_IPS=*
CSP_MODE=prod

# ============================================================================

# SYSTEM

# ============================================================================

PORT=${{PORT}}
PYTHONUNBUFFERED=1
GHOST_TZ=America/Chicago
LOG_LEVEL=INFO
LOG_JSON=1
DATABASE_URL=sqlite:///data/ghost.db

```text

1. Click **Deploy**to restart with new variables


---

### STEP 3: Verify Dockerfile

Your current Dockerfile should work, but verify it has:

```dockerfile

# Final CMD should be

CMD uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080}

```text

If not, update the Dockerfile and push:

```bash

git add Dockerfile
git commit -m "fix: ensure Dockerfile uses correct CMD"
git push origin main

```text

---

### STEP 4: Test Locally First

Before deploying to Railway, test locally:

```bash

# Set environment variables

export OPENAI_API_KEY=sk-proj-...
export TELEGRAM_TOKEN=...
export PORT=8444

# Run the app

uvicorn wolf_app:APP --host 0.0.0.0 --port 8444

```text

If it starts locally, the issue is Railway configuration.
If it fails locally, there's a code issue to fix.

---

## 🔍 Common Failure Patterns

### Pattern 1: Missing API Keys**Logs show:**`KeyError: 'OPENAI_API_KEY'` or similar**Fix:**Add all environment variables in Railway dashboard**IMPORTANT:**Use correct variable names

- ✅ `TELEGRAM_BOT_TOKEN` (NOT `TELEGRAM_TOKEN`)
- ✅ `ALPHAVANTAGE_API_KEY` (NOT `ALPHAVANTAGE_KEY`)
- ✅ `POLYGON_API_KEY` (NOT `POLYGON_KEY`)


### Pattern 2: Import Errors**Logs show:**`ModuleNotFoundError: No module named 'xyz'`**Fix:**Add missing package to `requirements.txt`

### Pattern 3: Port Binding**Logs show:**`Address already in use` or port errors**Fix:**Ensure CMD uses `${PORT}` variable

### Pattern 4: Database Path**Logs show:**`sqlite3.OperationalError: unable to open database`**Fix:**Add volume mount or use Railway database service

### Pattern 5: Redis Connection**Logs show:**`redis.exceptions.ConnectionError`**Fix:**Verify `REDIS_URL` is correct and Upstash Redis is active

---

## 🧪 Testing Checklist

After fixing, test these:

```bash

# Health check

curl <<<<<https://your-app.up.railway.app/ui/health>>>>>

# Should return

{"status":"ok","uptime":123,...}

```text

```bash

# World context

curl <<<<<https://your-app.up.railway.app/api/world/context>>>>>

# Should return market data

```text

```bash

# Browser test

# Open: <<<<<https://your-app.up.railway.app>>>>>

# Should load cockpit UI

```text

---

## 📞 Get Railway Logs

Run this to see what Railway sees:

```bash

# Install Railway CLI (if not installed)

npm install -g @railway/cli

# Login

railway login

# Link to your project

railway link

# View logs

railway logs

```text

---

## 🎯 Quick Debug Script

Save this as `test_local.sh`:

```bash

#!/bin/bash

echo "🧪 Testing Ghost Local Startup..."

# Load environment variables

source .env 2>/dev/null || echo "⚠️ No .env file found"

# Check critical vars

echo ""
echo "Environment Variables:"
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
echo "  TELEGRAM_TOKEN: ${TELEGRAM_TOKEN:0:10}..."
echo "  PORT: ${PORT:-8444}"

# Test import

echo ""
echo "Testing Python import..."
python3 -c "from wolf_app import APP; print('✅ Import successful')" || exit 1

# Try to start

echo ""
echo "Starting uvicorn..."
uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8444}

```text

Run with: `chmod +x test_local.sh && ./test_local.sh`

---

## 💡 If Still Failing

1.**Check Railway status**: <<<<<https://railway.statuspage.io/>>>>>

1. **Check resource limits**: Your plan might be out of resources
2. **Try smaller deployment**: Disable non-critical features
3. **Check build time**: If > 10 minutes, may timeout


---

## 📚 Useful Railway Commands

```bash

# Restart deployment

railway up

# View environment variables

railway vars

# Open project in browser

railway open

# SSH into container (if running)

railway shell

```text

---

*Last Updated: November 17, 2025*
*Ghost Protocol - Railway Deployment Guide*
