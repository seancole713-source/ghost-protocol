# Market Open Telegram Alert - Resolution Report

## Issue

**User Report:** "The market is open but I never got a telegram message with the open
price and my holding that should be sent 10 mins after opening"

## Root Cause

The scheduled market open/close alert feature was **disabled by default** in the
environment configuration.

### Technical Details

- Feature controlled by env var: `ALERT_SCHEDULE_OPEN_CLOSE`
- Default value in `wolf_app.py` line 2126:
  `int(os.getenv("ALERT_SCHEDULE_OPEN_CLOSE", "0"))`
- Was set to `"0"` (disabled)
- Conditional startup in `wolf_app.py` line 1816:
  ```python
  if SCHEDULE_OPEN_CLOSE:
      _start_schedule_worker()
  ```

## Solution Implemented

### 1. **Environment Configuration Updated**

Added to `/workspaces/GHOST/secrets.env`:

```bash
# Enable market open/close scheduled alerts (10 min window)
ALERT_SCHEDULE_OPEN_CLOSE=1
SCHEDULE_WINDOW_S=600
```

**Why 600 seconds?**

- User specified "10 mins after opening"
- Market opens at 9:30 AM ET
- Window of ±10 minutes (600 seconds) catches 9:30-9:40 AM window
- Prevents duplicate sends on the same day via `_SCHED_LAST_OPEN_DAY` tracking

### 2. **Scheduler Behavior**

The scheduler thread (`_schedule_loop` in `wolf_app.py:4610-4648`) runs every 30 seconds
and:

- Checks if today is weekday (Mon-Fri)
- Calculates time distance from 9:30 AM (open) and 4:00 PM (close)
- Fires alert if within `SCHEDULE_WINDOW_S` and not already sent today
- Alert format:
  ```
  🟢 OPEN — WOLF
  Portfolio
  • Qty: 909.43
  • Avg Cost: $3.30
  • Price: $XX.XX (live)
  • PnL: ...
  ```

### 3. **Telegram Verification**

Verified Telegram bot configuration:

- ✅ `TELEGRAM_BOT_TOKEN`: SET
- ✅ `TELEGRAM_CHAT_ID`: SET
- ✅ Test endpoint works: `/api/telegram/test?send=false` returns valid card

### 4. **Activation Steps**

**IMPORTANT:** Server must be restarted to pick up new environment variables:

```bash
# Kill existing server
pkill -f "uvicorn wolf_app"

# Start with new config (scheduler will now be enabled)
source .venv/bin/activate
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload
```

**Verification:**

```bash
# Check scheduler is enabled in runtime config
curl -s http://localhost:5000/api/config | jq '.alerts.schedule_open_close'
# Should return: true
```

## Expected Behavior (After Restart)

- **9:30-9:40 AM ET (Mon-Fri):** Market open alert sent automatically with live WOLF
  price + portfolio snapshot
- **4:00-4:10 PM ET (Mon-Fri):** Market close alert sent automatically
- **One alert per day** (deduplicated by date key)

## Testing

To manually test the scheduler logic without waiting for market open:

```python
# Temporarily modify wolf_app.py line 4622 for testing:
open_dt = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
# Change to current time + 1 minute for immediate trigger
```

## Related Code References

- Scheduler initialization: `wolf_app.py:1816-1817`
- Scheduler loop: `wolf_app.py:4610-4648`
- Open window check: `wolf_app.py:4627-4637`
- Close window check: `wolf_app.py:4638-4648`
- Telegram send: `wolf_app.py:4406-4485`

## Status

✅ **RESOLVED** - Configuration updated, scheduler enabled. Requires server restart to
activate.

## Next Market Open Test

- **Date:** Next trading day (Mon-Fri)
- **Time:** 9:30-9:40 AM ET
- **Expected:** Telegram message with open price + portfolio snapshot
- **Verification:** Check Telegram chat for 🟢 OPEN message

______________________________________________________________________

**Updated:** 2025-10-07 15:55 UTC\
**Author:** GitHub Copilot\
**Sprint:** Deep Scrub + Full Fix - Phase 4
