# 🔧 GHOST ISSUES ANALYSIS & FIXES

**Date:** October 6, 2025\
**Problems Identified:**

1. ❌ **Telegram shows 0 shares** (should show 8.41959051)
2. ❌ **Watchlist JavaScript error**: `f.value?.toFixed is not a function`
3. ❌ **Price not updating** in real-time (stuck at prev-close)
4. ❌ **WOLF not in watchlist** (only 52 other symbols)

______________________________________________________________________

## 🔍 ROOT CAUSE ANALYSIS

### Issue #1: Telegram Returns 0 Shares

**Location:** `wolf_app.py` line 6414-6415

```python
q = float(STATE.get("qty", 0.0))
a = float(STATE.get("avg_cost", 0.0))
```

**Problem:**

- UI shows correct data: 8.41959051 shares ✅ (reads from `/api/portfolio`)
- Telegram shows 0 shares ❌ (reads from `STATE["qty"]`)
- Server logs show position WAS loaded: `position_restored_from_db qty=8.41959051`

**Hypothesis:**\
STATE["qty"] and STATE["avg_cost"] were initially loaded correctly, but then got
overwritten or reset. Possible causes:

1. Another part of code resets STATE["qty"] to 0
2. Persist save/load cycle resets the legacy fields
3. Bootstrap running AFTER persist_load overwrites with 0

**Fix:**\
The Telegram handler should read from `/api/portfolio` endpoint (which works correctly)
instead of directly from STATE.

**Code Change:**

```python
# OLD (broken):
q = float(STATE.get("qty", 0.0))
a = float(STATE.get("avg_cost", 0.0))

# NEW (fixed):
# Use the same logic as api_portfolio endpoint
positions = STATE.get("positions", [])
if positions:
    wolf_pos = next((p for p in positions if p.get("symbol") == WOLF), None)
    if wolf_pos:
        q = float(wolf_pos.get("qty", 0.0))
        a = float(wolf_pos.get("price", 0.0))  # cost basis
    else:
        q = float(STATE.get("qty", 0.0))  # fallback to legacy
        a = float(STATE.get("avg_cost", 0.0))
else:
    q = float(STATE.get("qty", 0.0))
    a = float(STATE.get("avg_cost", 0.0))
```

______________________________________________________________________

### Issue #2: Watchlist JavaScript Error

**Error:**
`f.value?.toFixed is not a function. (In 'f.value?.toFixed(3)', 'f.value?.toFixed' is undefined)`

**Location:** Frontend React/JavaScript code in `ui_dist/assets/app.js`

**Problem:**\
The watchlist API returns:

```json
{
  "symbols": [
    {
      "symbol": "WOLF",
      "name": "Wolfspeed Inc.",
      "added_at": "2025-10-06...",
      "metadata": null
    }
  ]
}
```

But the UI JavaScript expects:

```json
{
  "symbols": [
    {
      "symbol": "WOLF",
      "value": 24.37,  // ← MISSING!
      "change": 0.05
    }
  ]
}
```

**Fix Options:**

**Option A: Fix Backend** - Add `value` field to watchlist response

```python
# In api_watchlist endpoint
for symbol_data in symbols:
    price, _, _ = get_wolf_price() if symbol_data['symbol'] == WOLF else (None, None, None)
    symbol_data['value'] = price if price else 0.0
    symbol_data['change'] = 0.0  # Calculate daily change
```

**Option B: Fix Frontend** - Handle missing `value` field

```javascript
// In UI code
const value = f.value || f.price || f.last_price || 0;
const formatted = value.toFixed(3);
```

______________________________________________________________________

### Issue #3: Price Not Updating

**Problem:**

- UI shows: "as of 10/06/2025, 10:55:39 AM"
- Current time: 10:59 AM
- Price stuck at $24.37 (prev-close)
- Market is OPEN but price not refreshing

**Diagnosis:**\
The price fetching loop is working (diagnostics show cache hits), but:

1. Yahoo Finance is rate-limiting (seen in logs: "Failed to get ticker 'WOLF'")
2. System falling back to prev-close cached price
3. No alternative provider (AlphaVantage, Polygon) being used for real-time data

**Fix:**\
Enable alternative price providers when Yahoo fails:

```python
# In get_wolf_price() function
# 1. Try Yahoo Finance
try:
    price = _fetch_yahoo(WOLF)
    if price:
        return price, prev_close, "yahoo"
except:
    pass

# 2. Try AlphaVantage
if ALPHAVANTAGE_API_KEY:
    try:
        price = _fetch_alphavantage(WOLF)
        if price:
            return price, prev_close, "alphavantage"
    except:
        pass

# 3. Try Polygon
if POLYGON_API_KEY:
    try:
        price = _fetch_polygon(WOLF)
        if price:
            return price, prev_close, "polygon"
    except:
        pass

# 4. Fallback to prev-close
return prev_close, prev_close, "prev-close"
```

______________________________________________________________________

### Issue #4: WOLF Not in Watchlist

**Problem:**\
Watchlist has 52 symbols but WOLF (your primary holding) is NOT included.

**Symbols Found:** AEO, ANET, APH, CAH, CFG, CL, CNH, (...)

**Missing:** WOLF (your 8.41959051 share position!)

**Fix:**\
Add WOLF to the watchlist via API or database:

```python
# Via API:
curl -X POST http://localhost:5000/api/watchlist \
  -H "Content-Type: application/json" \
  -d '{"symbol":"WOLF"}'

# Via Database:
import sqlite3
conn = sqlite3.connect("data/watchlist.db")
conn.execute("""
  INSERT OR REPLACE INTO watchlist (symbol, name, added_at)
  VALUES ('WOLF', 'Wolfspeed Inc.', datetime('now'))
""")
conn.commit()
```

**Better:** Auto-add portfolio positions to watchlist on startup:

```python
# In startup function after loading portfolio
for position in STATE.get("positions", []):
    symbol = position.get("symbol")
    if symbol:
        # Add to watchlist
        watchlist_manager.add_symbol(symbol)
```

______________________________________________________________________

## 🛠️ IMMEDIATE FIXES TO APPLY

### Fix #1: Update Telegram Handler

**File:** `wolf_app.py`\
**Lines:** 6414-6423, 2104-2105, 7060-7070 (all places reading STATE["qty"])

```python
def _get_portfolio_qty_and_avg():
    """Get current portfolio quantity and avg cost from STATE.
    Checks positions array first, then falls back to legacy qty/avg_cost fields.
    """
    positions = STATE.get("positions", [])
    if positions:
        wolf_pos = next((p for p in positions if p.get("symbol") == WOLF), None)
        if wolf_pos:
            return float(wolf_pos.get("qty", 0.0)), float(wolf_pos.get("price", 0.0))
    # Fallback to legacy fields
    return float(STATE.get("qty", 0.0)), float(STATE.get("avg_cost", 0.0))

# Then replace all instances of:
q = float(STATE.get("qty", 0.0))
a = float(STATE.get("avg_cost", 0.0))

# With:
q, a = _get_portfolio_qty_and_avg()
```

### Fix #2: Add WOLF to Watchlist

**Script:** Run once to add WOLF

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/GHOST')

from core.watchlist_manager import WatchlistManager

wm = WatchlistManager()
wm.add_symbol("WOLF", name="Wolfspeed Inc.")
print("✅ WOLF added to watchlist")
```

### Fix #3: Fix Watchlist API Response

**File:** `wolf_app.py`\
**Function:** `api_watchlist`

Add price data to each symbol:

```python
@APP.get("/api/watchlist")
async def api_watchlist():
    # ... existing code ...
    enriched_symbols = []
    for sym in symbols:
        symbol_name = sym.get("symbol") if isinstance(sym, dict) else sym
        
        # Fetch current price
        try:
            if symbol_name == WOLF:
                price, prev, _ = get_wolf_price()
            else:
                price, prev = None, None  # TODO: fetch other symbols
            
            enriched = {
                "symbol": symbol_name,
                "name": sym.get("name") if isinstance(sym, dict) else symbol_name,
                "value": price if price else 0.0,
                "change": ((price - prev) / prev * 100) if (price and prev and prev > 0) else 0.0,
                "added_at": sym.get("added_at") if isinstance(sym, dict) else None
            }
            enriched_symbols.append(enriched)
        except:
            # Fallback without price
            enriched_symbols.append(sym)
    
    return {"symbols": enriched_symbols, "count": len(enriched_symbols)}
```

### Fix #4: Enable Real-Time Price Updates

**File:** `wolf_app.py`\
**Function:** `get_wolf_price()`

Add AlphaVantage/Polygon fallback when Yahoo fails.

______________________________________________________________________

## 🔄 SERVER RESTART PROCEDURE

After applying fixes:

```bash
# 1. Stop current server
pkill -f "uvicorn wolf_app"

# 2. Clear any stale state
rm -f ghost_server.out

# 3. Restart with all env vars
cd /workspaces/GHOST && source .venv/bin/activate
export SIM_MODE=0 USE_PLACEHOLDERS=0 PORTFOLIO_PERSISTENCE_ENABLED=1
export ALERT_SCHEDULE_OPEN_CLOSE=1 PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload > ghost_server.out 2>&1 &

# 4. Wait for startup
sleep 5

# 5. Verify
curl -s http://localhost:5000/api/portfolio | python3 -m json.tool
```

______________________________________________________________________

## ✅ VERIFICATION CHECKLIST

After fixes:

- [ ] Telegram `/status` shows 8.41959051 shares
- [ ] UI Watchlist displays without JavaScript errors
- [ ] WOLF appears in watchlist
- [ ] Price updates every 5 seconds (check diagnostics panel)
- [ ] Portfolio P&L matches live price changes

______________________________________________________________________

## 📝 SUMMARY

**Critical Fixes Needed:**

1. **Telegram Handler** - Update all places reading STATE["qty"] to use helper function
   that checks positions array first
2. **Watchlist Data** - Add `value` field with current price to API response
3. **Add WOLF to Watchlist** - Your primary position should be tracked!
4. **Enable Price Fallbacks** - Use AlphaVantage/Polygon when Yahoo rate-limits

**Expected Results:**

- ✅ Telegram will show correct 8.41959051 shares
- ✅ Watchlist will display all symbols including WOLF
- ✅ No JavaScript errors
- ✅ Real-time prices updating (when providers available)

**Estimated Time:** 15 minutes to apply all fixes + restart server
