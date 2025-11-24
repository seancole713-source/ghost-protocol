# GHOST FAILURE POINTS DIAGRAM

**Purpose**: Identify responsibility for failures across the stack with testable symptoms.

---

## FAILURE CATEGORIES

### 1. 🐳 RAILWAY / DEPLOYMENT FAILURES

**Responsibility**: Railway platform, container orchestration, environment configuration

| Symptom | Root Cause | How to Test | Who Fixes |
|---------|------------|-------------|-----------|
| HTTP 502 Bad Gateway | Container crashed/restarting | `curl -I https://ghost-protocol-production.up.railway.app/health` | Railway/DevOps |
| HTTP 503 Service Unavailable | Container not running | Check Railway dashboard → Deployments tab | Railway/DevOps |
| Environment variables missing | `.env` not synced to Railway | `curl /api/v3/cockpit/status` returns errors about missing keys | DevOps |
| Port mismatch | Container listening on wrong port | Railway logs show "address already in use" | DevOps |
| Memory/CPU limits hit | Container OOMKilled or throttled | Railway Metrics tab shows 100% usage | DevOps/Optimization |
| Build failures | Dockerfile errors, missing dependencies | Railway Build Logs show errors | DevOps |
| Database file missing | Volume not mounted correctly | `curl /api/v3/predictions/latest` returns 0 predictions | DevOps |

**Test Command**:
```bash
# Check if Railway container is up and responding
curl -I https://ghost-protocol-production.up.railway.app/health
# Expected: HTTP 200

# Check environment variables loaded
curl -s https://ghost-protocol-production.up.railway.app/api/v3/cockpit/status | jq '.live'
# Expected: true

# Check Railway deployment status
# Go to: https://railway.app/project/ghost-protocol/service/ghost-protocol/deployments
# Expected: Green "Active" badge
```

---

### 2. 🔑 PROVIDER / API KEY FAILURES

**Responsibility**: External API providers (Polygon, Alpha Vantage), API key configuration

| Symptom | Root Cause | How to Test | Who Fixes |
|---------|------------|-------------|-----------|
| HTTP 403 Forbidden | Rate limit hit, bad API key | Railway logs show "403 Client Error" | DevOps (upgrade plan or fix key) |
| HTTP 401 Unauthorized | Invalid/expired API key | Direct test: `curl "https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey=BAD_KEY"` | DevOps (update key) |
| HTTP 429 Too Many Requests | Rate limit exceeded | Railway logs show "rate_limited": true | DevOps (upgrade plan or reduce frequency) |
| Price data = 0.0 | All providers failed, fell back to placeholder | `curl /api/v3/hunter/feed` returns movers with `price: 0.0` | Provider issue (wait) or DevOps (check keys) |
| Timeout errors | Provider API slow/down | Railway logs show timeout after 30s | Provider issue (wait) or increase timeout |
| Empty provider response | Provider returning no data | Railway logs show "Insufficient data" warnings | Provider issue (wait) or check symbol availability |

**Test Commands**:
```bash
# Test Polygon API directly
curl "https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey=8VIvELVXiLG30K2l1348RzSurffLM0jR"
# Expected: JSON with price data (not "status": "ERROR")

# Test Alpha Vantage directly
curl "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=3WNNLA81KS7BG4AK"
# Expected: JSON with "Global Quote" object

# Test Ghost provider fallback
curl -s https://ghost-protocol-production.up.railway.app/api/price/AAPL | jq '.provider'
# Expected: "polygon" or "alpha_vantage" (not "yfinance" or "prev-close")

# Check Railway logs for rate limit warnings
# Go to: https://railway.app → Observability → Logs
# Search: "rate_limited":true
# Expected: No recent matches (if matches found, rate limits are active)
```

---

### 3. 🌐 UI / BROWSER FAILURES

**Responsibility**: JavaScript code, browser compatibility, network, CORS

| Symptom | Root Cause | How to Test | Who Fixes |
|---------|------------|-------------|-----------|
| Panel shows "--" or blank | JavaScript fetch failed | Browser DevTools → Console → Look for `[GHOST V3] Error loading` | Frontend Dev |
| HTTP 499 Client Cancelled | Browser cancelled request (timeout, user navigated away) | Railway HTTP Logs show 499 status | Frontend Dev (increase timeout) or user behavior |
| CORS errors | Missing CORS headers | Browser Console shows "blocked by CORS policy" | Backend Dev (add CORS headers) |
| Wrong base URL | Hardcoded localhost in production | Check `cockpit_v3.js` for `http://localhost` strings | Frontend Dev |
| JavaScript errors | Syntax error, undefined variable | Browser Console shows red error messages | Frontend Dev |
| Panel never updates | Interval not set up correctly | Check `cockpit_v3.js` for `setInterval()` calls | Frontend Dev |
| Data mismatch | UI expects different JSON structure | Compare API response vs JS code field access | Frontend Dev |

**Test Commands**:
```bash
# Open production Cockpit V3
open https://ghost-protocol-production.up.railway.app/cockpit

# Open browser DevTools (F12 or Cmd+Option+I)
# Go to Console tab
# Look for errors (red text)

# Check Network tab
# Reload page
# Look for failed requests (red status codes)
# Click on failed request → Headers → Check status code
# Expected: All requests should be 200 OK

# Test API endpoints directly
curl -s https://ghost-protocol-production.up.railway.app/api/v3/watchlist | jq .
curl -s https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?limit=5 | jq .
curl -s https://ghost-protocol-production.up.railway.app/api/v3/hunter/feed | jq .
# Compare JSON structure to what UI expects
```

---

### 4. 📱 TELEGRAM / ALERT FAILURES

**Responsibility**: Telegram Bot API, alert queue, message templates

| Symptom | Root Cause | How to Test | Who Fixes |
|---------|------------|-------------|-----------|
| No alerts sent | Telegram bot token invalid/missing | Check Railway env vars for `TELEGRAM_BOT_TOKEN` | DevOps |
| Alert queue full | Too many alerts, queue not draining | Railway logs show "alert_queue_full" warnings | Backend Dev (increase queue size) |
| Fake accuracy claims | Hardcoded "85%+ Accuracy" text | Check Telegram messages for accuracy claim, compare to `/api/v3/accuracy/summary` | **FIXED** (commit in this audit) |
| Alerts delayed | Alert worker slow or stuck | Check Railway logs for alert worker thread activity | Backend Dev |
| Malformed messages | HTML escaping issues | Telegram shows raw HTML tags like `<b>` | Backend Dev (fix template) |
| Chat ID invalid | Wrong chat ID in env vars | Telegram API returns "chat not found" | DevOps |

**Test Commands**:
```bash
# Check if Telegram bot token is set
curl -s "https://api.telegram.org/bot$(grep TELEGRAM_BOT_TOKEN .env | cut -d'=' -f2)/getMe"
# Expected: JSON with bot username

# Check current accuracy data
curl -s https://ghost-protocol-production.up.railway.app/api/v3/accuracy/summary | jq '{total_predictions, correct, wrong, pending}'
# Compare to Telegram message claims

# Search Railway logs for Telegram errors
# Go to: Railway → Observability → Logs
# Search: "telegram" OR "alert"
# Look for error messages

# Manually test alert sending (if admin endpoint exists)
# curl -X POST https://ghost-protocol-production.up.railway.app/api/test/alert \
#   -H "Content-Type: application/json" \
#   -d '{"text": "Test alert from Ghost Truth Audit"}'
```

---

### 5. 💾 DATABASE FAILURES

**Responsibility**: SQLite database, file system, data integrity

| Symptom | Root Cause | How to Test | Who Fixes |
|---------|------------|-------------|-----------|
| Empty predictions array | Database file missing | `curl /api/v3/predictions/latest` returns `predictions: []` | DevOps (check volume mount) |
| Database locked | Write conflict, WAL mode not enabled | Railway logs show "database is locked" | Backend Dev (enable WAL) |
| Prediction IDs not incrementing | Auto-increment broken | Check prediction IDs for gaps or duplicates | Backend Dev (check schema) |
| Outcomes never updated | Reconciler worker not running | `curl /api/v3/accuracy/summary` shows `pending: 100, correct: 0` after 48h | Backend Dev (check worker thread) |
| Disk full | Container storage limit hit | Railway logs show "no space left on device" | DevOps (increase storage) |
| Corrupted database | Crash during write, bad shutdown | SQLite queries return "malformed" errors | DevOps (restore from backup) |

**Test Commands**:
```bash
# Check if predictions are being stored
curl -s https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?limit=10 | jq '.count'
# Expected: >= 1 (if auto-prediction loop is running)

# Check if database is accessible
curl -s https://ghost-protocol-production.up.railway.app/api/v3/watchlist | jq '.count'
# Expected: >= 20 (should have 26 symbols)

# Check if outcomes are being reconciled
curl -s https://ghost-protocol-production.up.railway.app/api/v3/accuracy/summary | jq '{total_predictions, correct, wrong, pending}'
# Expected: If total_predictions > 10 and all predictions > 48h old, some should be correct/wrong (not all pending)

# Check Railway logs for database errors
# Search: "database" OR "sqlite" OR "locked"
# Expected: No recent errors

# Check container disk usage (if SSH access available)
# df -h /app/data
# Expected: < 80% usage
```

---

### 6. 🔄 AUTO-PREDICTION LOOP FAILURES

**Responsibility**: Background worker thread, prediction generation logic

| Symptom | Root Cause | How to Test | Who Fixes |
|---------|------------|-------------|-----------|
| No predictions generated | Loop not started | Railway logs show NO "[AUTO-PREDICT] Running batch" messages | Backend Dev |
| Loop stuck | Thread deadlock or exception | Railway logs show last batch started hours ago | Backend Dev |
| Only 1-2 symbols predicted | Loop iterating too slowly | Railway logs show "Batch complete: 2/26" repeatedly | Backend Dev (optimize) |
| Predictions all FLAT | Direction logic broken | `curl /api/v3/predictions/latest?limit=20` shows ALL directions = "FLAT" | Backend Dev (check direction calculation) |
| Predictions all 40% confidence | Confidence logic broken | `curl /api/v3/predictions/latest?limit=20` shows ALL confidence = 0.40 | Backend Dev (check confidence calculation) |
| Batch takes > 5 min | Feature extraction too slow | Railway logs show batch completion time > 300s | Backend Dev (optimize feature extraction) |

**Test Commands**:
```bash
# Check if auto-prediction loop is running
curl -s https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?limit=30 | jq '.predictions | map(.symbol) | unique | length'
# Expected: >= 10 different symbols (if loop ran at least once)

# Check last prediction time (should be < 5 minutes ago)
curl -s https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?limit=1 | jq '.predictions[0].run_at' | xargs -I {} date -r {}
# Expected: Recent timestamp (within last 5-10 minutes)

# Check Railway logs for loop activity
# Search: "[AUTO-PREDICT]"
# Expected: See "Running batch" messages every 5 minutes
# Expected: See "Batch complete: X/26" messages

# Check prediction diversity (not all FLAT)
curl -s https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?limit=20 | jq '.predictions | group_by(.direction) | map({direction: .[0].direction, count: length})'
# Expected: Mix of UP/DOWN/FLAT (not 100% FLAT)

# Check confidence variation (not all 40%)
curl -s https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?limit=20 | jq '.predictions | [.[].confidence] | {min: min, max: max, avg: (add/length)}'
# Expected: min ~40%, max ~70-85%, avg ~55%
```

---

## QUICK DIAGNOSTIC DECISION TREE

```
UI panel is empty/broken
│
├─ Browser console has errors?
│  ├─ YES → UI/Browser failure (Category 3)
│  └─ NO → Continue
│
├─ Network tab shows HTTP 50x errors?
│  ├─ YES → Railway/Deployment failure (Category 1)
│  └─ NO → Continue
│
├─ API returns empty array [] but HTTP 200?
│  ├─ YES → Database or Auto-Loop failure (Category 5 or 6)
│  └─ NO → Continue
│
├─ API returns data with `price: 0.0` or `confidence: 0`?
│  ├─ YES → Provider/API Key failure (Category 2)
│  └─ NO → Continue
│
└─ Data exists but UI shows "--" or wrong values?
   └─ YES → UI wiring mismatch (Category 3 - check GHOST_UI_WIRING.md)
```

---

## TELEGRAM ALERT DIAGNOSTIC

```
Telegram alert claims "85%+ Accuracy"
│
└─ Check actual accuracy:
   curl /api/v3/accuracy/summary | jq '.daily_accuracy_pct'
   │
   ├─ Returns 0% but alert says 85%+?
   │  └─ **TEMPLATE LIE** (Category 4) - **FIXED in this audit**
   │
   ├─ Returns 75-90% and alert says 85%+?
   │  └─ OK (claim is accurate)
   │
   └─ Returns null/error?
      └─ Database or API failure (Category 5)
```

---

## RESPONSIBILITY MATRIX

| Category | Symptom Keywords | Responsible Team | Escalation Path |
|----------|------------------|------------------|-----------------|
| **Railway** | 502, 503, crashed, OOMKilled, build failed | DevOps | Railway support ticket |
| **Providers** | 403, 401, 429, rate_limited, timeout | DevOps → Provider support | Upgrade API plan or change providers |
| **UI** | console.error, CORS, fetch failed, 499 | Frontend Dev | Backend Dev if API issue |
| **Telegram** | alert_queue_full, bot token, template lie | Backend Dev | DevOps if token/config issue |
| **Database** | empty array, locked, corrupted | Backend Dev | DevOps if disk full |
| **Auto-Loop** | no predictions, stuck, slow | Backend Dev | DevOps if CPU/memory limits hit |

---

**Generated**: November 24, 2025  
**Auditor**: Ghost Truth Squad  
**Use**: Quick diagnosis when panels are dead or alerts are lying
