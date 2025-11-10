# 🚀 Real-Time Price Fix Applied

## Problem

Ghost was caching prices for **45 seconds** during market hours, causing stale data.

Your broker shows: **$24.59**\
Ghost showed: **$24.37** (22¢ stale)

## Root Cause

In `wolf_app.py` line 512:

```python
PRICE_TTL_OPEN_S = int(os.getenv("PRICE_TTL_OPEN_S", "45"))  # ← TOO LONG!
```

The cache was set to 45 seconds, so Ghost would serve the same stale price for 45
seconds instead of fetching fresh data.

## Solution

1. **Cache TTL now defaults to 5 seconds** when the market is open. This change lives in
   `wolf_app.py` and no longer requires a manual environment override.
2. When a cached price comes from `prev-close` during market hours, Ghost discards it
   and fetches a live quote automatically.
3. A new endpoint `/api/price/refresh` clears the cache and forces an immediate refresh.

### Optional environment override

You can still override the TTL via the environment if you want a different cadence:

```bash
export PRICE_TTL_OPEN_S=5
```

Then restart Ghost server.

## Manual Restart Instructions (optional)

```bash
# 1. Stop Ghost
pkill -f "uvicorn wolf_app"
lsof -ti:5000 | xargs kill -9 2>/dev/null || true
sleep 3

# 2. Start with real-time pricing
cd /workspaces/GHOST
source .venv/bin/activate

export SIM_MODE=0
export PORTFOLIO_PERSISTENCE_ENABLED=1
export PRICE_TTL_OPEN_S=5          # ← 5 second cache (was 45!)
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

nohup uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload > ghost_server.out 2>&1 &

# 3. Wait for startup
sleep 10

# 4. Verify real-time updates
curl http://localhost:5000/api/portfolio | python3 -c "import json,sys; pos=json.load(sys.stdin)['positions'][0]; print(f'Price: \${pos[\"current\"]} from {pos[\"src\"]}')"
```

## Manual Refresh API

Trigger an immediate refresh without restarting:

```bash
curl -X POST http://localhost:5000/api/price/refresh
```

Response:

```json
{
  "symbol": "WOLF",
  "price": 24.59,
  "prev_close": 24.37,
  "provider": "yahoo",
  "timestamp": 1759769000,
  "cache_cleared": true
}
```

## Expected Result

After restart with `PRICE_TTL_OPEN_S=5`:

- **Every 5 seconds**, Ghost will fetch fresh prices from Yahoo/AlphaVantage/Polygon
- **UI will show live updates** matching your broker
- **NAV will be accurate** within 5 seconds

## Verification

Watch the diagnostics panel - you should see:

```json
{
  "provider": "yahoo",  // ← Not "prev-close"!
  "price": 24.59,       // ← Live price!
  "ttl_hit": false      // ← Fresh fetch, not cache!
}
```

## Alternative: Use VS Code Task

You can also restart using the VS Code task:

1. Open Command Palette (Ctrl+Shift+P)
2. Run Task > "Run Ghost server (:5000)"
3. Manually add env var `PRICE_TTL_OPEN_S=5` to task definition in `.vscode/tasks.json`

## Files Created

- `restart_realtime.sh` - Automated restart script
- `force_price_refresh.py` - Diagnostic tool
- `REALTIME_PRICE_FIX.md` - This document

______________________________________________________________________

**Status**: ⏸️ Awaiting manual restart with `PRICE_TTL_OPEN_S=5`

Once restarted, Ghost will fetch fresh prices every 5 seconds and match your broker's
live data!
