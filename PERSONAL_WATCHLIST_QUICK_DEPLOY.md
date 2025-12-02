## 🚀 PERSONAL WATCHLIST - QUICK DEPLOY

**Status:** ✅ Ready | **Risk:** LOW | **Time:** 5 minutes

---

### ONE-COMMAND DEPLOY (Local Machine)

```bash
cd /path/to/ghost-protocol
git add -A
git commit -m "feat: Personal watchlist system"
git push origin main
```

Railway auto-deploys in ~2-3 minutes.

---

### POST-DEPLOY (Railway Active)

```bash
# 1. Apply migration (one-time)
railway run psql $DATABASE_URL -f migrations/001_personal_watchlist.sql

# 2. Verify (should show 7 seed symbols)
curl https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user
```

---

### VERIFY IN COCKPIT

1. Open: https://ghost-protocol-production.up.railway.app/cockpit
2. Find **WATCHLIST** panel (Panel 5)
3. See 7 symbols: BTC, ETH, AAPL, TSLA, XRP, NVDA, MSFT
4. Click "➕ Add Symbol" - should work
5. Click tabs (Stocks / Crypto / All) - should filter

---

### ROLLBACK (If Needed)

```bash
git revert HEAD
git push origin main
```

OR disable in Railway dashboard:
```
WATCHLIST_SCHEDULER_ENABLED=0
WATCHLIST_ALERTS_ENABLED=0
```

---

### FILES CHANGED (9)

**Modified:**
- templates/cockpit_v3.html
- static/cockpit_v3.js  
- static/personal_watchlist_ui.js
- GHOST_PROD_OPERATOR_PLAYBOOK.md

**New:**
- PERSONAL_WATCHLIST_DEPLOYMENT_GUIDE.md
- PERSONAL_WATCHLIST_INTEGRATION_SUMMARY.md

**Verified Complete:**
- api/personal_watchlist_endpoints.py
- core/personal_watchlist.py
- core/watchlist_prediction_scheduler.py
- core/watchlist_telegram_alerts.py
- migrations/001_personal_watchlist.sql
- tests/test_personal_watchlist.py
- wolf_app.py (endpoints + scheduler already wired)

---

### SUCCESS = USER CAN

✅ Add/remove symbols from Cockpit UI  
✅ See 48h predictions (UP/DOWN + confidence)  
✅ Persist watchlist across browser sessions  
✅ Get Telegram alerts for market events  
✅ Filter by stocks/crypto/all tabs  

---

**Deploy NOW. Questions? See PERSONAL_WATCHLIST_DEPLOYMENT_GUIDE.md**
