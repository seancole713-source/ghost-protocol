# 🚀 Quick Start: Activate Phase 4 Fixes

## TL;DR - What You Need to Do NOW

### 1️⃣ Restart the Server (REQUIRED)

```bash
# Stop current server
pkill -f "uvicorn wolf_app"

# Start with new config
cd /workspaces/GHOST
source .venv/bin/activate
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
rm -rf "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload
```

### 2️⃣ Verify Everything Works

```bash
./verify_sprint_phase4.sh
```

Expected output:

- ✅ Server is responding
- ✅ Market open/close scheduler is ENABLED
- ✅ Telegram is configured and ready
- ✅ Corporate actions API working
- ✅ Price diagnostics endpoint responding

### 3️⃣ Wait for Market Open Alert

**When:** Next trading day (Mon-Fri) at 9:30-9:40 AM ET\
**Where:** Your Telegram chat\
**What:** 🟢 OPEN message with WOLF price + portfolio snapshot

______________________________________________________________________

## What Was Fixed

| Issue | Status | Action Required | |-------|--------|-----------------| | 📱 Missing
market open Telegram | ✅ FIXED | Restart server | | 📊 Empty /metrics endpoint | ⚠️
DOCUMENTED | Optional: Add eager init | | 🎨 Cockpit.html refactoring | 📋 PLANNED |
Future work (~10 hrs) | | 🧪 Throttling test suite | ✅ CREATED | Run after restart |

______________________________________________________________________

## Files to Read

1. **MARKET_OPEN_ALERT_FIX.md** - Why you weren't getting alerts (required reading)
2. **PROMETHEUS_METRICS_DEBUG.md** - Why metrics are empty (optional, for
   troubleshooting)
3. **COCKPIT_REFACTORING_PLAN.md** - How to clean up the UI code (future work)
4. **SPRINT_SUMMARY_PHASE4.md** - Complete technical details (comprehensive)

______________________________________________________________________

## Quick Verification Commands

```bash
# Is scheduler enabled?
curl -s http://localhost:5000/api/config | jq '.alerts.schedule_open_close'
# Should show: true

# Is Telegram working?
curl -X POST "http://localhost:5000/api/telegram/test?send=false" | jq '.can_send'
# Should show: true

# Are corporate actions visible?
curl -s http://localhost:5000/api/corporate_actions | jq '.symbols'
# Should show: ["WOLF"]

# Is price diagnostics enhanced?
curl -s http://localhost:5000/api/price/diagnostics | jq '.backoff_active'
# Should show: {} (empty object if no throttling active)
```

______________________________________________________________________

## What to Expect Next

### Tomorrow Morning (9:30-9:40 AM ET)

You will receive a Telegram message like:

```
🟢 OPEN — WOLF

Portfolio
• Qty: 909.43
• Avg Cost: $3.30
• Price: $XX.XX (live)
• Market Value: $X,XXX.XX
• PnL: -$X,XXX.XX (-XX.XX%)
• Note: Adjusted for 120.0:1 reverse split (2025-10-01)

NAV / Cash
• NAV: $XXX,XXX.XX
• Cash: $176,000.00

...
```

### Tomorrow Afternoon (4:00-4:10 PM ET)

Similar message with:

```
🔴 CLOSE — WOLF
...
```

______________________________________________________________________

## If Something Goes Wrong

### No Telegram Alert Tomorrow?

1. Check server logs: `tail -f ghost_server.log | grep schedule`
2. Look for: `"schedule_open_send"` or `"schedule_loop_failed"`
3. Verify scheduler is running: `ps aux | grep python | grep uvicorn`

### Metrics Still Empty?

1. Generate some activity: `curl http://localhost:5000/api/cockpit`
2. Check again: `curl http://localhost:5000/metrics | head -20`
3. If still empty, see `PROMETHEUS_METRICS_DEBUG.md` for code fix

### Server Won't Start?

1. Check if port is in use: `lsof -i :5000`
2. Kill process: `kill -9 <PID>`
3. Try again with fresh start

______________________________________________________________________

## Support Resources

- **Documentation:** All `.md` files in `/workspaces/GHOST/`
- **Tests:** `tests/test_provider_backoff.py` (run with `pytest`)
- **Verification:** `./verify_sprint_phase4.sh`
- **Logs:** `tail -f ghost_server.log`

______________________________________________________________________

## ONE Command to Rule Them All

```bash
# Run this after server restart:
./verify_sprint_phase4.sh && echo "✅ All systems ready for market open!"
```

______________________________________________________________________

**Created:** 2025-10-07 16:20 UTC\
**Priority:** 🔴 HIGH - Restart server before market open!\
**Impact:** 📱 Enables automated daily market open/close alerts
