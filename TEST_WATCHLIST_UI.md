# Ghost Protocol Watchlist UI - Test & Validation Guide

## Changes Summary

### Files Modified
1. **templates/cockpit_v3.html**
   - Added Personal/Market mode tabs (`#watchlist-mode-tabs`)
   - Kept existing Stocks/Crypto/All filter tabs (`#watchlist-filter-tabs`)
   - Both tab groups now visible in watchlist panel

2. **static/cockpit_v3.js**
   - Added state variables: `watchlistMode` ('personal' | 'market'), `watchlistFilter` ('all' | 'stocks' | 'crypto')
   - Added `loadWatchlistByMode()` - master router function
   - Refactored `loadMarketWatchlist()` - existing market watchlist (now with filter support)
   - Updated `switchTab()` - handles both mode tabs and filter tabs
   - Updated event listeners - distinguishes `data-mode` vs `data-tab` attributes
   - Updated interval loaders - calls `loadWatchlistByMode()` every 15s

3. **static/personal_watchlist_ui.js**
   - Removed auto-override of `loadWatchlist()`
   - Now works alongside cockpit_v3.js (called when mode='personal')
   - All CRUD functions intact: add, remove, toggle ownership, view history

## API Endpoints Used

### Personal Watchlist
- **GET** `/api/v3/watchlist/user` - Fetch enriched personal watchlist with predictions
- **POST** `/api/v3/watchlist/add` - Add symbol (body: symbol, asset_type, owns_position, notes, alert_threshold_pct, priority)
- **POST** `/api/v3/watchlist/remove` - Remove symbol (body: symbol, asset_type)
- **POST** `/api/v3/watchlist/update-position` - Update owns_position flag
- **GET** `/api/v3/watchlist/history/{symbol}` - Get prediction history

### Market Watchlist
- **GET** `/api/v3/watchlist/enriched` - Fetch default Ghost market watchlist with live prices
- **GET** `/api/v3/predictions/latest?limit=100` - Fetch prediction data to enrich watchlist

## User Workflow

### Initial Load (Mode: Personal, Filter: All)
1. User opens `/cockpit/v3`
2. Cockpit loads with "Personal" tab active by default
3. `loadWatchlistByMode()` → detects mode='personal' → calls `loadPersonalWatchlist()`
4. If personal watchlist empty: Shows "Your watchlist is empty" + "Add Symbol" button
5. If personal watchlist has items: Displays with add/remove controls

### Adding a Symbol to Personal Watchlist
1. User clicks "➕ Add Symbol" button
2. Modal appears with form:
   - Symbol (text input, uppercased)
   - Asset Type (dropdown: Stock | Crypto)
   - Owns Position (checkbox)
   - Alert Threshold (number input, default 5.0%)
   - Notes (textarea, optional)
3. User fills form and clicks "Add Symbol"
4. POST to `/api/v3/watchlist/add`
5. On success: Modal closes, watchlist reloads, toast notification shows
6. Symbol now appears in personal watchlist with prediction data

### Removing a Symbol from Personal Watchlist
1. User clicks "✖" button on a watchlist row
2. Confirmation dialog: "Remove {SYMBOL} from watchlist?"
3. If confirmed: POST to `/api/v3/watchlist/remove`
4. On success: Watchlist reloads, toast notification shows

### Switching to Market Watchlist
1. User clicks "📊 Market" tab
2. `switchTab()` → sets `watchlistMode = 'market'` → calls `loadWatchlistByMode()`
3. `loadMarketWatchlist()` executes:
   - Fetches `/api/v3/watchlist/enriched` (default Ghost symbols)
   - Fetches `/api/v3/predictions/latest` (prediction data)
   - Enriches watchlist with Ghost predictions
   - Applies filter (stocks/crypto/all based on active filter tab)
4. Market watchlist renders with Ghost predictions (NO add/remove buttons)

### Filter Tabs (Stocks/Crypto/All)
- **When mode=personal**: Filter is applied by `personal_watchlist_ui.js` via `getFilteredWatchlistItems()`
- **When mode=market**: Filter is applied by `loadMarketWatchlist()` using array filter on `item.type`
- Both modes respect the same filter tabs

## Testing Checklist

### ✅ Backend API Tests
```bash
# Test market watchlist (should work immediately)
curl "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/enriched"

# Test personal watchlist GET (may be empty initially)
curl "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user"

# Test add symbol (replace with real values)
curl -X POST "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/add" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","asset_type":"stock","owns_position":false,"notes":"Test","alert_threshold_pct":5.0,"priority":1}'

# Test remove symbol
curl -X POST "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/remove" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","asset_type":"stock"}'
```

### ✅ UI Tests (Browser Console)

**Open**: https://ghost-protocol-production.up.railway.app/cockpit/v3

**Test 1: Check Mode Tabs Render**
```javascript
// Should see both tab groups
document.getElementById('watchlist-mode-tabs')  // Personal/Market tabs
document.getElementById('watchlist-filter-tabs')  // Stocks/Crypto/All tabs
```

**Test 2: Personal Watchlist Loads**
```javascript
// Check state
console.log(watchlistMode);  // Should be 'personal'
console.log(personalWatchlistState.items);  // Array of items or []

// Manually trigger load
loadPersonalWatchlist();
```

**Test 3: Switch to Market Watchlist**
```javascript
// Click "Market" tab or:
watchlistMode = 'market';
loadWatchlistByMode();

// Should see default Ghost symbols (BTC, ETH, ADA, DOGE, etc.)
```

**Test 4: Add Symbol to Personal Watchlist**
```javascript
// Click "Add Symbol" button or:
showAddSymbolForm();

// Fill form and submit or:
submitAddSymbol();  // (after filling form fields)
```

**Test 5: Filter Tabs Work in Both Modes**
```javascript
// Set mode to personal
watchlistMode = 'personal';
loadWatchlistByMode();

// Apply stock filter
watchlistFilter = 'stocks';
updateWatchlistTab('stocks');  // Should show only stocks

// Switch to crypto filter
watchlistFilter = 'crypto';
updateWatchlistTab('crypto');  // Should show only crypto

// Switch back to market mode and test filters
watchlistMode = 'market';
loadWatchlistByMode();
// Filters should still work
```

### ✅ Visual Regression Tests

**Market Watchlist (mode='market'):**
- ❌ NO "Add Symbol" button
- ❌ NO remove (✖) buttons on rows
- ✅ Shows Ghost predictions (direction + confidence)
- ✅ Shows price change %
- ✅ Respects Stocks/Crypto/All filter
- ✅ Updates every 15 seconds

**Personal Watchlist (mode='personal'):**
- ✅ Shows "Add Symbol" button at top
- ✅ Each row has: ownership toggle (✅/➕), history (📊), remove (✖)
- ✅ Shows asset type badge (STOCK/CRYPTO)
- ✅ Shows OWN badge if owns_position=true
- ✅ Shows Ghost predictions (direction + confidence)
- ✅ Respects Stocks/Crypto/All filter
- ✅ Updates every 15 seconds

### ✅ No-Regression Tests

**Existing Cockpit Behavior:**
- ✅ Top Movers panel still works
- ✅ VIP Coins panel still works
- ✅ Forecast panel still works
- ✅ News panel still works
- ✅ Health Score panel still works
- ✅ Goals modal still works
- ✅ All other tabs (Stocks/Crypto/All in Top Movers) still work

## Known Limitations & Future Work

### Current Scope (COMPLETE)
- ✅ Dual-mode watchlist (Personal + Market)
- ✅ Full CRUD for personal watchlist (add, remove, update ownership)
- ✅ Filter tabs work in both modes
- ✅ Postgres persistence (survives browser refresh)
- ✅ Enriched with 48h Ghost predictions
- ✅ Modal-based add/remove UI

### NOT in Scope (Intentional)
- ❌ VIP Coins module (separate feature - not touching)
- ❌ Trade execution (Ghost = signals only, no broker integration)
- ❌ Multi-user auth (single-owner system)
- ❌ Drag-to-reorder (not required by operator)

### Future Enhancements (Optional)
- 🔄 Real-time WebSocket updates for watchlist
- 🔄 Bulk import/export (CSV upload)
- 🔄 Symbol search autocomplete
- 🔄 Price alert notifications (browser push)
- 🔄 Watchlist groups/folders

## Deployment Verification

After Railway auto-deploy, verify:

1. **No Import Errors**
   ```bash
   # Check Railway logs for errors
   railway logs --tail 100 | grep -i "error\|import"
   ```

2. **Endpoints Respond**
   ```bash
   curl -I https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user
   # Should return HTTP 200 (even if empty)
   ```

3. **UI Loads Without Console Errors**
   - Open browser DevTools Console
   - Navigate to /cockpit/v3
   - Should see: `[PERSONAL WATCHLIST] UI module initialized and ready`
   - NO red errors about undefined functions

4. **Database Tables Exist**
   ```sql
   -- Via Railway Postgres plugin
   SELECT COUNT(*) FROM ghost_watchlist_items;
   -- Should return 0 or more (not "relation does not exist")
   ```

## Rollback Plan

If critical issues arise:

1. **Revert HTML changes:**
   ```bash
   git checkout HEAD~1 templates/cockpit_v3.html
   ```

2. **Revert JS changes:**
   ```bash
   git checkout HEAD~1 static/cockpit_v3.js
   git checkout HEAD~1 static/personal_watchlist_ui.js
   ```

3. **Redeploy:**
   ```bash
   git commit -m "Rollback watchlist UI changes"
   git push origin main
   ```

## Success Criteria

✅ **MISSION COMPLETE** when:
1. Personal watchlist visible with "Add Symbol" button
2. Market watchlist visible with default Ghost symbols
3. Tabs switch between Personal/Market seamlessly
4. Add/remove work without page reload
5. All existing cockpit features still functional
6. No console errors in browser DevTools
7. Database persistence confirmed (refresh page, symbols remain)

---

**Status:** ✅ READY FOR PRODUCTION TEST
**Next Step:** Push to main → Railway auto-deploy → Test in browser
