# Ghost Protocol - Deployment Monitoring Guide

## Quick Health Check

```bash
# Run health monitor
chmod +x railway_health_monitor.sh
./railway_health_monitor.sh

# Or with verbose output
./railway_health_monitor.sh --verbose
```

## Manual Health Checks

### 1. Check if Server is Running
```bash
curl https://ghost-protocol-production.up.railway.app/health
# Expected: {"status": "ok"}
```

### 2. Verify Predictions are Generating
```bash
curl https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest | jq
# Expected: {"ok": true, "predictions": [...], "count": N}
```

### 3. Check Prediction Loop Status
```bash
# View Railway logs (requires Railway CLI)
railway logs --tail 100

# Look for these messages:
# ✅ Auto-Prediction Loop: STARTED (ASYNC, non-blocking, 60-min interval)
# [AUTO-PREDICT] Processing 150+ crypto symbols
# [AUTO-PREDICT] ✅ Async cycle complete: X/150+ predictions
```

### 4. Verify No Errors
```bash
# Check for crypto provider errors
railway logs | grep "Failed to get ticker"
# Should NOT see Yahoo Finance 429 errors for crypto

# Check for table errors
railway logs | grep "no such table"
# Should NOT see "no such table: ghost_predictions"
```

## Monitoring Checklist

### Every Hour (Automated)
- [ ] Auto-prediction loop running (check logs for cycle completion)
- [ ] No Yahoo Finance rate limit errors
- [ ] Predictions being saved to database

### Daily (Manual)
- [ ] Telegram reports showing prediction counts
- [ ] Accuracy metrics updating
- [ ] No sustained errors in Railway logs

### Weekly (Manual)
- [ ] Review accuracy trends (target: 70%+)
- [ ] Check database size (SQLite + PostgreSQL)
- [ ] Review API endpoint performance

## Key Metrics to Monitor

### Prediction Generation
```bash
# Count predictions in last 24 hours
curl https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?limit=100 | \
  jq '[.predictions[] | select(.run_at > (now - 86400))] | length'
```

### Accuracy Tracking
```bash
# Get current accuracy
curl https://ghost-protocol-production.up.railway.app/api/v3/accuracy/summary | \
  jq '{daily: .daily_accuracy_pct, weekly: .weekly_accuracy_pct, monthly: .monthly_accuracy_pct}'
```

### System Health
```bash
# Check health metrics
curl https://ghost-protocol-production.up.railway.app/api/v3/health/metrics | jq
```

## Troubleshooting

### Issue: No Predictions Generating

**Symptoms:**
- `/api/v3/predictions/latest` returns empty array
- Hunter feed shows "no predictions"
- Telegram reports show "0 predictions"

**Check:**
```bash
railway logs | grep "Auto-Prediction Loop"
# Should see: "✅ Auto-Prediction Loop: STARTED"
```

**Fix:**
1. Check if auto_prediction_loop started in logs
2. Verify HUNTER_CRYPTO_SYMBOLS is populated (150+ symbols)
3. Wait 60 minutes for first prediction cycle

### Issue: Yahoo Finance 429 Errors

**Symptoms:**
```
Failed to get ticker 'DOT' reason: 429 Client Error: Too Many Requests
```

**Check:**
```bash
railway logs | grep "429"
```

**Fix:**
- Should be fixed in commit `f31f6ea` (crypto symbols now use CoinGecko)
- Verify crypto symbols route to `turbo_crypto_price` not `yfinance`

### Issue: SQLite Table Not Found

**Symptoms:**
```
[ACCURACY] Evaluator error: no such table: ghost_predictions
```

**Check:**
```bash
railway logs | grep "Prediction store tables initialized"
# Should see: "✅ Prediction store tables initialized"
```

**Fix:**
- Should be fixed in commit `f31f6ea` (tables initialize on startup)
- If still occurring, check for permission issues in `/app/data/`

## Railway CLI Commands

```bash
# View live logs
railway logs

# Tail logs (like tail -f)
railway logs --tail 50

# View environment variables
railway variables

# Trigger manual deployment
railway up

# Check service status
railway status
```

## Alerts Setup (Optional)

### Telegram Notifications
Set up Telegram bot to receive alerts:

```bash
# Add to Railway environment
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### External Monitoring (Recommended)
- **UptimeRobot**: Monitor `/health` endpoint every 5 minutes
- **Better Uptime**: Monitor multiple endpoints with status page
- **Cronitor**: Monitor prediction cycle timing

## Performance Benchmarks

### Expected Response Times
- `/health`: < 50ms
- `/api/v3/cockpit/status`: < 100ms
- `/api/v3/predictions/latest`: < 200ms (cached)
- `/api/v3/watchlist/user`: < 500ms (crypto price fetches)

### Prediction Generation
- **Crypto (24/7)**: 150+ predictions every 60-120 minutes
- **Stocks (market hours)**: ~20 predictions every 60 minutes
- **Total**: 170+ predictions per cycle

### Accuracy Targets
- **Daily**: 70%+ (short-term volatility)
- **Weekly**: 70%+ (medium-term trends)
- **Monthly**: 70%+ (strategic accuracy goal)

## Next Steps

1. ✅ Deploy commit `f31f6ea` to Railway
2. ⏱️ Wait 10 minutes for deployment
3. 🧪 Run health monitor script
4. ⏱️ Wait 60 minutes for first prediction cycle
5. 📊 Verify predictions generating in logs
6. 📱 Check Telegram report (next scheduled time)

---

**Last Updated:** December 14, 2025  
**Version:** Post-Critical-Fix (commits `ec902a2`, `f31f6ea`)
