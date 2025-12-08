# 🔒 GHOST BASELINE INVESTMENT DATA

**Date Locked**: October 8, 2025 **Source**: User's actual brokerage account screenshot

______________________________________________________________________

## 🎯 LOCKED BASELINE VALUES (DO NOT CHANGE)

### WOLF Position

- **Shares Owned**: `8.41959051` (EXACT)
- **Average Cost**: `$359.28` per share
- **Current Price (LOCKED)**: `$29.66`
- **Total Invested**: `$3,025.02`
- **Current Value**: `$249.73`
- **Total Loss**: `-$2,775.29`
- **Loss %**: `-91.74%`


### Portfolio

- **Cash Balance**: `$250.90` ← CORRECT!
- **Position Value**: `$249.73`
- **Total NAV**: `$500.63`


______________________________________________________________________

## 📋 RULES FOR GHOST

1. **Baseline Lock**: Use $29.66 as WOLF price until Ghost fetches a NEW confirmed price
2. **Daily Lock**: Each day at market close (4:00 PM ET), LOCK the closing price
3. **Price Update**: Only update price when:
   - Market is OPEN (9:30 AM - 4:00 PM ET, Mon-Fri)
   - Ghost successfully fetches from price provider
   - New price is confirmed valid
1. **After Market Close**: Use the locked closing price until next market open
2. **No Interpolation**: Never estimate or calculate prices between locks


______________________________________________________________________

## 🔍 VERIFICATION FORMULAS

```python

# Position Value

position_value = 8.41959051 * 29.66 = $249.73 ✓

# Total NAV

nav = position_value + cash = 249.73 + 176000 = $176,249.73 ✓

# Loss Calculation

loss = position_value - total_invested = 249.73 - 3025.02 = -$2,775.29 ✓
loss_pct = (loss / total_invested) * 100 = -91.74% ✓

```text

______________________________________________________________________

## 🐛 CURRENT GHOST ISSUES

**What Ghost is showing (WRONG)**:

- ❌ Price: $26.69 or $26.17 (incorrect old data)
- ❌ NAV: $176,224.72 (wrong calculation)
- ❌ Position Value: $224.72 (wrong - using wrong price)


**What Ghost SHOULD show (CORRECT)**:

- ✅ Price: $29.66 (locked baseline)
- ✅ NAV: $176,249.73
- ✅ Position Value: $249.73


______________________________________________________________________

## 🔧 WHERE TO FIX

1. **ghost_state.json**: Update `trading_state.positions[0].current_price` to `29.66`
2. **Price Provider**: Ensure locked price is used when market closed
3. **Agent Snapshot**: Verify agent sees correct locked price
4. **UI Display**: Cockpit should show $29.66 locked baseline


______________________________________________________________________

## 📊 EXPECTED BEHAVIOR

### Market Closed (After 4:00 PM ET)

```json

{
  "symbol": "WOLF",
  "quantity": 8.41959051,
  "entry_price": 359.28,
  "current_price": 29.66,
  "price_status": "LOCKED_CLOSE",
  "locked_at": "2025-10-08T16:00:00-04:00"
}

```text

### Market Open (9:30 AM - 4:00 PM ET)

```json

{
  "symbol": "WOLF",
  "quantity": 8.41959051,
  "entry_price": 359.28,
  "current_price": 29.66,  // or NEW live price if fetched
  "price_status": "LIVE" or "LOCKED_CLOSE"
}

```text

______________________________________________________________________

## ✅ VALIDATION CHECKLIST

- [ ] ghost_state.json updated with $29.66 baseline
- [ ] Ghost Agent sees $29.66 in snapshot
- [ ] UI Cockpit displays $249.73 position value
- [ ] NAV shows $176,249.73
- [ ] Loss shows -$2,775.29 (-91.74%)
- [ ] Price locked until new confirmed fetch


______________________________________________________________________

**BASELINE LOCKED**: Do not change these numbers until Ghost confirms a NEW price from
live market data!
