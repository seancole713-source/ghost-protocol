# 🎭 Ghost Protocol - Quick Test Reference

## 🚀 One-Liner Tests

```bash
# Test everything at once
python3 test_endpoints_and_accuracy.py

# Check accuracy ledger database
sqlite3 data/forecast_accuracy.db "SELECT COUNT(*) as forecasts FROM forecasts"

# Quick endpoint health check
curl -s https://ghost-sniper-bot-seancole713-production.up.railway.app/api/world/context | jq '.ok'
```

---

## 📊 Browser Console (Copy/Paste)

```javascript
// Quick test all endpoints
['world/context', 'goals/all', 'xrp/tracker', 'vip/coins', 'portfolio/positions'].forEach(ep => {
  fetch('/api/' + ep).then(r => r.json()).then(d => console.log('✅', ep, d)).catch(e => console.error('❌', ep, e));
});
```

---

## 🎯 Accuracy Check

```bash
# How many predictions today?
sqlite3 data/forecast_accuracy.db "SELECT COUNT(*) FROM forecasts WHERE date(forecast_timestamp, 'unixepoch') = date('now')"

# Latest prediction accuracy
sqlite3 data/forecast_accuracy.db "SELECT symbol, forecast_price, actual_price, percentage_error FROM forecasts WHERE actual_price IS NOT NULL ORDER BY id DESC LIMIT 1"
```

---

## 🔄 Auto-Tuning Status

```bash
# Check learning stats
curl -s https://ghost-sniper-bot-seancole713-production.up.railway.app/api/stage2/learning_stats | jq

# Check forecast accuracy report
curl -s https://ghost-sniper-bot-seancole713-production.up.railway.app/api/stage2/forecasts?limit=5 | jq '.forecasts[] | {symbol, forecast_price, actual_price, error: .percentage_error}'
```

---

## 📋 Daily Checklist

- [ ] Run `python3 test_endpoints_and_accuracy.py`
- [ ] Send `/predict` to Telegram bot
- [ ] Open cockpit browser console
- [ ] Verify ✅ checkmarks appear
- [ ] Check accuracy stats at market close

---

*Quick Reference - Ghost Protocol v2.0*
