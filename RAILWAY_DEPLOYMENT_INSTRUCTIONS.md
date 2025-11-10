# 🚂 RAILWAY DEPLOYMENT - GHOST OMNIBRAIN v10.3

**Quick Deploy**: Copy environment variables → Deploy → Verify

______________________________________________________________________

## ⚡ RAPID DEPLOYMENT (5 Minutes)

### Step 1: Set Environment Variables in Railway Dashboard

Go to your Railway project → Variables tab → Add these:

```bash
# === REQUIRED FOR CRYPTO ===
CRYPTO_ENABLED=1
CRYPTO_SYMBOLS=BTC,ETH,SOL,BNB,DOGE,SHIB,PEPE,AVAX,DOT,MATIC,LINK,UNI,AAVE
CRYPTO_LOOKBACK_H=96
CRYPTO_FORECAST_H=48
CRYPTO_PRICE_SOURCE=coingecko
CRYPTO_QUORUM=coingecko,binance,coinbase

# === FEATURES ===
NEWS_SENTIMENT_ON=1
FUSION_AI_ON=1
MACRO_BRAIN_ON=1

# === AI (Required if AI_ON=1) ===
AI_ON=1
AI_PROVIDER=openai
AI_MODEL=gpt-4o-mini
OPENAI_API_KEY=<paste-your-key-here>

# === CORE CONFIG ===
SIM_MODE=0
LOG_LEVEL=INFO
LOG_JSON=1
PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
ADMIN_IP_ALLOWLIST=127.0.0.1

# === OPTIONAL: Stock Providers ===
ALPHA_VANTAGE_API_KEY=<your-key>
POLYGON_API_KEY=<your-key>
POLYGON_KEY=<your-key>

# === OPTIONAL: Security ===
GHOST_API_TOKEN=<generate-a-random-string>

# === OPTIONAL: Telegram ===
TELEGRAM_BOT_TOKEN=<your-bot-token>
TELEGRAM_CHAT_ID=<your-chat-id>

# === DATABASE ===
WOLF_SQLITE_PATH=data/wolf.db

# === UI PREFERENCES ===
GHOST_TZ=America/New_York
GHOST_CLOCK_24H=0
```

### Step 2: Deploy

#### Option A: Railway CLI

```bash
cd /workspaces/GHOST
railway up
```

#### Option B: Railway Dashboard

1. Click "Deploy" button
2. Wait for build to complete (~2-3 minutes)
3. Railway will automatically start the service

### Step 3: Verify Deployment

```bash
# Replace with your Railway URL
export RAILWAY_URL="https://your-app.railway.app"

# 1. Health Check
curl $RAILWAY_URL/health
# Expected: {"ok": true}

# 2. Crypto Price
curl "$RAILWAY_URL/api/crypto/price/BTC"
# Expected: {price: <number>, provider: "coingecko"}

# 3. Cockpit
curl "$RAILWAY_URL/api/cockpit" | jq '.predictions.crypto'
# Expected: Array of crypto predictions

# 4. Metrics
curl "$RAILWAY_URL/metrics" | grep ghost_up
# Expected: ghost_up 1.0
```

______________________________________________________________________

## 🔑 ENVIRONMENT VARIABLES REFERENCE

### Critical (Must Have)

| Variable | Value | Purpose | |----------|-------|---------| | `CRYPTO_ENABLED` | `1` |
Enable crypto module | | `CRYPTO_SYMBOLS` | `BTC,ETH,SOL,BNB,...` | Coins to track | |
`AI_ON` | `1` | Enable AI features | | `OPENAI_API_KEY` | `sk-proj-...` | OpenAI API key
| | `SIM_MODE` | `0` | Live mode (not simulation) |

### Important (Recommended)

| Variable | Value | Purpose | |----------|-------|---------| | `NEWS_SENTIMENT_ON` |
`1` | News sentiment scoring | | `FUSION_AI_ON` | `1` | AI fusion scoring | |
`MACRO_BRAIN_ON` | `1` | Macro indicators | | `GHOST_API_TOKEN` | `<random-string>` |
API security | | `ALPHA_VANTAGE_API_KEY` | `<your-key>` | Stock prices | |
`POLYGON_API_KEY` | `<your-key>` | Stock news |

### Optional (Enhanced Features)

| Variable | Value | Purpose | |----------|-------|---------| | `TELEGRAM_BOT_TOKEN` |
`<bot-token>` | Telegram integration | | `TELEGRAM_CHAT_ID` | `<chat-id>` | Your
Telegram chat | | `GHOST_TZ` | `America/New_York` | Display timezone | | `LOG_LEVEL` |
`INFO` | Logging verbosity |

______________________________________________________________________

## 🔧 POST-DEPLOYMENT CHECKLIST

### Immediate (First 5 Minutes)

- [ ] Health endpoint returns `{"ok": true}`
- [ ] Metrics endpoint returns `ghost_up 1.0`
- [ ] Crypto price endpoint works (BTC, ETH, SOL)
- [ ] Cockpit shows crypto predictions array
- [ ] No errors in Railway logs

### First Hour

- [ ] Monitor CPU/Memory usage (should be \<50% / \<200 MB)
- [ ] Verify price updates every 5 minutes
- [ ] Check prediction generation works
- [ ] Test API endpoints from Postman/curl
- [ ] Verify database writes (check Railway volumes)

### First Day

- [ ] Set up Grafana dashboard (optional)
- [ ] Configure alerts (PagerDuty/Slack)
- [ ] Test Telegram integration (if enabled)
- [ ] Verify all 45+ coins updating
- [ ] Monitor for rate limiting issues

______________________________________________________________________

## 🚨 TROUBLESHOOTING

### Issue: "Crypto module not enabled" Error

**Symptom**: API returns `{"detail": "Crypto module not enabled"}`

**Cause**: `CRYPTO_ENABLED` not set to `1`

**Fix**:

1. Go to Railway dashboard → Variables
2. Add/Update: `CRYPTO_ENABLED=1`
3. Redeploy (Railway auto-restarts)

### Issue: "OpenAI API Error"

**Symptom**: AI features don't work, errors in logs

**Cause**: Missing or invalid `OPENAI_API_KEY`

**Fix**:

1. Get API key from https://platform.openai.com/api-keys
2. Add to Railway: `OPENAI_API_KEY=sk-proj-...`
3. Alternatively, disable AI: `AI_ON=0`

### Issue: High Memory Usage

**Symptom**: Railway shows >400 MB memory usage

**Cause**: Too many coins or inefficient caching

**Fix**:

1. Reduce `CRYPTO_SYMBOLS` to fewer coins
2. Increase cache TTL (less frequent updates)
3. Upgrade Railway plan to 1GB RAM

### Issue: Rate Limiting from CoinGecko

**Symptom**: Errors like "429 Too Many Requests"

**Cause**: Exceeding 50 calls/min free tier

**Fix**:

- Already mitigated with 2-min cache TTL
- Reduce update frequency
- Or upgrade to CoinGecko Pro ($129/month)

### Issue: Stale UI / Old Version Showing

**Symptom**: Changes not visible in browser

**Cause**: Railway CDN caching

**Fix**:

1. Hard refresh browser: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
2. Add cache buster: `?v=20251012` to URL
3. Clear Railway cache: Redeploy

______________________________________________________________________

## 📊 MONITORING & OBSERVABILITY

### Railway Metrics (Built-in)

- **CPU Usage**: Monitor in Railway dashboard
- **Memory**: Should stay \<200 MB
- **Network**: ~1 MB/hour
- **Deploy Time**: ~2-3 minutes

### Prometheus Metrics (Custom)

Access at `https://your-app.railway.app/metrics`:

```bash
# Key metrics to monitor
ghost_up                                    # Should be 1.0
ghost_crypto_price_fetch_total              # Increments with each fetch
ghost_crypto_predict_seconds_bucket         # Prediction latency histogram
ghost_prediction_mape{asset_class="crypto"} # Prediction accuracy
ghost_snapshot_asof                         # Data freshness (Unix timestamp)
```

### Set Up Grafana (Optional)

1. Create Grafana Cloud account (free tier)
2. Add Prometheus data source: `https://your-app.railway.app/metrics`
3. Import dashboard template (coming soon)
4. Set alerts for:
   - `ghost_up == 0` (service down)
   - `rate(ghost_crypto_price_fetch_total{result="error"}[5m]) > 10` (provider errors)
   - `ghost_prediction_mape > 10` (poor accuracy)

______________________________________________________________________

## 🔄 UPDATES & ROLLBACKS

### Deploy New Version

```bash
# From local machine
git add .
git commit -m "feat: your changes"
git push origin main

# Railway automatically deploys from main branch
```

### Rollback to Previous Version

#### Option A: Railway Dashboard

1. Go to Deployments tab
2. Click on previous successful deployment
3. Click "Redeploy"

#### Option B: Git Revert

```bash
git revert HEAD
git push origin main
```

### Feature Flag Rollback (Instant)

If crypto module causes issues:

```bash
# In Railway Variables, change:
CRYPTO_ENABLED=0

# Service restarts automatically
# Crypto endpoints return "not enabled" but system stays up
```

______________________________________________________________________

## 🎯 SUCCESS CRITERIA

### Deployment Successful When:

- [x] Health check: `200 OK`
- [x] Metrics: `ghost_up = 1.0`
- [x] Crypto prices: All returning valid data
- [x] Predictions: Generating successfully
- [x] Logs: No critical errors
- [x] Memory: \<200 MB
- [x] CPU: \<20% idle

### Ready for Production When:

- [x] All acceptance tests passing
- [x] Contract tests: 100% pass rate
- [x] Monitoring: Grafana dashboard set up
- [x] Alerts: PagerDuty/Slack configured
- [x] Runbooks: Team trained
- [x] Rollback: Tested and confirmed working

______________________________________________________________________

## 📞 SUPPORT

### Self-Service Diagnostics

```bash
# 1. Check all systems
curl https://your-app.railway.app/health

# 2. Verify crypto module
curl https://your-app.railway.app/api/crypto/price/BTC

# 3. Check metrics
curl https://your-app.railway.app/metrics | grep ghost_

# 4. Run contract tests (from local)
python tests/contract_tests.py
```

### Common Commands

```bash
# View Railway logs
railway logs

# SSH into Railway container
railway run bash

# Check environment variables
railway variables

# Restart service
railway restart
```

______________________________________________________________________

## 📝 CHANGELOG

### v10.3.0 (October 12, 2025)

- ✅ Crypto prediction module
- ✅ Multi-provider quorum
- ✅ 45+ cryptocurrencies
- ✅ Prometheus metrics
- ✅ Contract testing
- ✅ System dependency map

______________________________________________________________________

## 🎉 READY TO DEPLOY!

**Estimated Time**: 5 minutes\
**Difficulty**: Easy\
**Cost**: $0-5/month\
**Uptime**: 99.9% target

**Run**: `railway up`

______________________________________________________________________

**Questions?** See `DEPLOYMENT_GUIDE_V10.3.md` for full documentation.
