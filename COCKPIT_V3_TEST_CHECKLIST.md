# Cockpit v3 Quick Test Checklist
**Test After Hard Refresh:** `Cmd+Shift+R` or `Ctrl+Shift+R`

## 1. Major Caps (CRITICAL FIX) ✅
```
Expected: BTC and ETH show live prices and 24h change
Current:  BTC ~$91,000 (-3.6%), ETH ~$3,100 (-1.6%)

Test:
□ Major Caps BTC price = Watchlist BTC price
□ Major Caps ETH price = Watchlist ETH price
□ Major Caps BTC 24h% = Watchlist BTC 24h%
□ Major Caps ETH 24h% = Watchlist ETH 24h%
□ No more "--" displayed
```

## 2. XRP VIP Card ✅
```
Expected: Shows price, signal, confidence, Eye Score, 24h change

Test:
□ Price displays (around $2.06)
□ Signal shows (BULLISH/BEARISH/HOLD)
□ Confidence shows (percentage)
□ Eye Score shows numeric value (e.g., "72/100 🟡")
□ 24h change shows (e.g., "+1.06%")
```

## 3. Prediction Accuracy Chart ✅
```
Expected: Friendly waiting message (not blank)

Test:
□ Shows "⏳ Waiting for predictions to mature..."
□ Shows "(Predictions need 48 hours to reconcile)"
□ No red error text
□ Chart area visible (not blank white space)
```

## 4. Goals Modal ✅
```
Expected: Saves goals and updates Health panel

Test:
□ Click "🎯 Set Trading Goals" - modal opens
□ Input fields show values (not empty)
□ Change Daily Goal to $1000
□ Click "Save Goals"
□ See success alert
□ Health panel Daily Goal updates to match
```

## 5. Watchlist Data Consistency ✅
```
Expected: All 15 assets show complete data

Test:
□ Each row has: Symbol, Type, Price, 24h%, Direction, Confidence
□ No "--" or "undefined" values
□ BTC/ETH rows match Major Caps exactly
□ Tab switching works (Personal/Market/Stocks/Crypto)
```

## 6. Browser Console (NO ERRORS) ✅
```
Expected: Green success logs, no red errors

Test:
□ Open DevTools Console (F12)
□ See: "✅ Ghost Protocol Cockpit v3 initialized"
□ See: "[VIP] Major Caps pulled from Watchlist"
□ No red error messages
□ All API calls return 200 OK in Network tab
```

## Quick Smoke Test (30 seconds)
1. Hard refresh (`Cmd+Shift+R`)
2. Check Major Caps - should show BTC ~$91,000, ETH ~$3,100
3. Check Accuracy Chart - should show "⏳ Waiting..."
4. Open Goals Modal - should have prefilled values
5. Check console - should be green, no red errors

**If all 6 sections pass: ✅ SYSTEM FULLY OPERATIONAL**

---

## Troubleshooting

### If Major Caps still shows "--":
- Verify cache version: View page source, search for `v=2025120800`
- If wrong version: Force refresh again (`Cmd+Shift+R`)
- Check console for "[VIP] Major Caps pulled from Watchlist" log

### If Goals Modal inputs are empty:
- API may be slow - wait 5 seconds and try opening modal again
- Check Network tab for `/api/v3/goals/snapshot` response
- Should return `{ok: true, goals: {daily: 500, ...}}`

### If Accuracy Chart is blank:
- This is expected - predictions need 48h to reconcile
- Should show waiting message (not blank white space)
- If completely blank: Check console for errors

### If console has red errors:
- Take screenshot and report in issue
- Note which API endpoint is failing
- Check if error is transient (refresh and retest)

---

**Created:** December 7, 2025  
**Commit:** `724b7b0` (accuracy UX), `d50ce49` (Major Caps fix)  
**Status:** All issues resolved, ready for user testing
