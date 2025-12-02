# Personal Watchlist Deployment Guide
**Date:** December 2, 2025  
**Status:** Ready for Deployment

---

## Quick Deployment Steps

### 1. Commit All Changes

From your **local machine** (outside dev container):

```bash
cd /path/to/ghost-protocol

# Stage all personal watchlist files
git add \
  templates/cockpit_v3.html \
  static/cockpit_v3.js \
  static/personal_watchlist_ui.js \
  api/personal_watchlist_endpoints.py \
  core/personal_watchlist.py \
  core/watchlist_prediction_scheduler.py \
  core/watchlist_telegram_alerts.py \
  wolf_app.py \
  migrations/001_personal_watchlist.sql \
  tests/test_personal_watchlist.py \
  PERSONAL_WATCHLIST_README.md \
  POSTGRES_WATCHLIST_MIGRATION_STATUS.md \
  GHOST_PROD_OPERATOR_PLAYBOOK.md \
  verify_postgres_migration.py \
  railway_migrate_watchlist.sh

# Commit with descriptive message
git commit -m "feat: Implement personal watchlist system

- Added personal watchlist CRUD endpoints (/api/v3/watchlist/*)
- Integrated personal watchlist UI into Cockpit v3
- Added prediction scheduler for watchlist symbols
- Added Telegram alerts for market events and big moves
- Migration script for 4 Postgres tables (ghost_watchlist_items, etc.)
- Comprehensive tests and operator documentation
- Tab filtering (stocks/crypto/all) in Cockpit UI
- Backwards compatible with global scanner watchlist

User can now:
- Manually add/remove stocks and crypto from watchlist
- See 48h predictions for each symbol
- Get Telegram alerts for market open/close/big moves
- Track ownership status (owns_position flag)
- Persist watchlist across browser sessions (Postgres)"

# Push to Railway (triggers auto-deploy)
git push origin main
```

### 2. Monitor Railway Deployment

```bash
# Watch deployment logs (from local machine)
railway logs --tail 100

# Look for these success indicators:
# ✅ Personal Watchlist endpoints registered (priority routing)
# ✅ Personal watchlist scheduler started
# ✅ Cockpit V3 LIVE endpoints registered
```

### 3. Apply Database Migration

Once deployment is ACTIVE (healthcheck passing):

```bash
# Apply migration from local machine
railway run psql $DATABASE_URL -f migrations/001_personal_watchlist.sql

# You should see:
# CREATE TABLE (4 times)
# CREATE INDEX (many times)
# INSERT 0 7 (seed data)
```

### 4. Verify Migration Success

```bash
# Run verification script
railway run python3 verify_postgres_migration.py

# Expected output:
# ✅ ghost_watchlist_items: EXISTS (7 rows)
# ✅ watchlist_prediction_tracking: EXISTS (0 rows)
# ✅ watchlist_price_snapshots: EXISTS (0 rows)
# ✅ watchlist_alerts_log: EXISTS (0 rows)
# ✅ ghost_predictions: EXISTS
#    - ID Range: 1 to 12+
```

### 5. Test Personal Watchlist Endpoints

```bash
# Get current watchlist (should have 7 seed symbols)
curl -sS "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user" \
  | python3 -m json.tool

# Expected: 7 items (BTC, ETH, AAPL, TSLA, XRP, NVDA, MSFT)

# Get stats
curl -sS "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/stats" \
  | python3 -m json.tool

# Expected:
# {
#   "total_items": 7,
#   "stocks": 4,
#   "crypto": 3,
#   ...
# }
```

### 6. Test Cockpit UI

1. Open Cockpit: https://ghost-protocol-production.up.railway.app/cockpit
2. Look for **WATCHLIST** panel (Panel 5)
3. Verify:
   - ✅ 7 symbols visible (seed data)
   - ✅ Each shows direction (🟢↑ or 🔴↓) and confidence
   - ✅ "➕ Add Symbol" button visible at top
   - ✅ Tabs work (Stocks / Crypto / All)
   - ✅ Each row has action buttons (✅ ➕ 📊 🗑️)

### 7. Test Add/Remove Functionality

#### Add a New Symbol (via API)

```bash
curl -X POST "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/add" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "NVDA",
    "asset_type": "stock",
    "owns_position": false,
    "notes": "NVIDIA - AI chips"
  }'

# Expected:
# {
#   "ok": true,
#   "action": "added",
#   "id": 8,
#   "symbol": "NVDA",
#   ...
# }
```

#### Add a New Symbol (via Cockpit UI)

1. Click "➕ Add Symbol" button
2. Fill form:
   - Symbol: `SOL`
   - Type: `Crypto`
   - Owns Position: `No`
   - Notes: `Solana`
3. Click "Add to Watchlist"
4. Verify SOL appears in list with prediction

#### Remove a Symbol (via Cockpit UI)

1. Find symbol in list (e.g., SOL)
2. Click 🗑️ (trash icon)
3. Confirm removal
4. Verify symbol disappears from list

### 8. Verify Scheduler is Running

```bash
# Check logs for scheduler activity
railway logs --tail 200 | grep -i "watchlist"

# Expected log patterns:
# 📅 Watchlist scheduler loop active
# ✅ Generated 3 predictions for watchlist (market open)
# 📊 Big move detected: BTC up 6.2% in 15min
```

### 9. Test Telegram Alerts (Optional)

**Prerequisites:**
- `TELEGRAM_BOT_TOKEN` set
- `TELEGRAM_CHAT_ID` set
- `WATCHLIST_ALERTS_ENABLED=1`

**Test Scenarios:**

1. **Market Open Alert** (9 AM EST on weekdays)
   - Wait for 9 AM EST or trigger manually
   - Check Telegram for "📌 WATCHLIST – MARKET OPEN" messages

2. **Big Move Alert** (price moves ±5% in 15 min)
   - Wait for volatile market conditions
   - Check Telegram for "📌 WATCHLIST – BIG MOVE" messages

3. **Manual Trigger** (immediate)
   ```bash
   curl -X POST "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/trigger-prediction" \
     -H "Content-Type: application/json" \
     -d '{"symbol":"BTC","asset_type":"crypto"}'
   ```

---

## Rollback Procedure

If issues occur, you can roll back:

### Option 1: Revert Git Commit

```bash
git revert HEAD
git push origin main
```

Railway will auto-deploy the previous version.

### Option 2: Keep Code, Disable Features

Set environment variables in Railway dashboard:

```bash
WATCHLIST_SCHEDULER_ENABLED=0
WATCHLIST_ALERTS_ENABLED=0
```

This disables scheduler and alerts but keeps endpoints available.

### Option 3: Drop Tables (Nuclear Option)

```bash
railway run psql $DATABASE_URL -c "DROP TABLE IF EXISTS watchlist_alerts_log CASCADE;"
railway run psql $DATABASE_URL -c "DROP TABLE IF EXISTS watchlist_price_snapshots CASCADE;"
railway run psql $DATABASE_URL -c "DROP TABLE IF EXISTS watchlist_prediction_tracking CASCADE;"
railway run psql $DATABASE_URL -c "DROP TABLE IF EXISTS ghost_watchlist_items CASCADE;"
```

**Warning:** This deletes all watchlist data permanently.

---

## Post-Deployment Checklist

- [ ] Deployment completed (Railway shows ACTIVE)
- [ ] Migration applied (4 tables exist)
- [ ] Endpoints return HTTP 200 (not 404)
- [ ] Cockpit UI shows watchlist panel
- [ ] Tab filtering works (stocks/crypto/all)
- [ ] Add symbol button visible
- [ ] Can add new symbol via UI
- [ ] Can remove symbol via UI
- [ ] Predictions show for each symbol
- [ ] Both UP and DOWN predictions visible (no bias)
- [ ] Scheduler logs appear (if enabled)
- [ ] Telegram alerts working (if enabled)
- [ ] Browser refresh preserves watchlist

---

## Support Resources

- **Documentation:** `PERSONAL_WATCHLIST_README.md`
- **Operator Guide:** `GHOST_PROD_OPERATOR_PLAYBOOK.md` (Appendix C)
- **Migration Status:** `POSTGRES_WATCHLIST_MIGRATION_STATUS.md`
- **Verification Script:** `verify_postgres_migration.py`
- **Tests:** `tests/test_personal_watchlist.py`

---

## Success Criteria

✅ **User can manually add/remove symbols** from Cockpit UI  
✅ **Watchlist persists across browser sessions** (Postgres)  
✅ **Each symbol shows 48h prediction** (direction + confidence)  
✅ **Both positive and negative predictions visible** (no green-only bias)  
✅ **Telegram alerts sent** for market events and big moves  
✅ **Tab filtering works** (stocks/crypto/all)  
✅ **Backwards compatible** (global scanner watchlist still works)  
✅ **No crashes or 500 errors** in production logs

---

**End of Deployment Guide**  
**Last Updated:** December 2, 2025
