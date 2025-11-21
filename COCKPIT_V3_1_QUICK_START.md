# 🚀 GHOST COCKPIT V3.1 - QUICK START GUIDE

## TL;DR - Get Running in 3 Minutes

```bash
# Navigate to project
cd /Users/studio713/ghost-protocol

# Rebuild container (includes V3.1 changes)
docker compose down
docker compose build --no-cache app

# Start Ghost
docker compose up

# Wait 180 seconds (3 minutes) for initialization

# Open browser
open http://localhost:8080/cockpit
```

---

## What You'll See

### Cockpit V3 Dashboard:

**Header**:
- 🟢 LIVE indicator (green dot)
- Ghost Score: `85/100` (B) - or `--` if still warming up
- Last updated: `14:23:45`
- Controls: START | STOP | RESET (simulation controls)

**Panel 1: Top Movers** (Crypto)
- Bitcoin, Ethereum, Solana... with 24h% changes
- Confidence scores (0-100)
- OR: "Scanner warming up - check back in 60 seconds"

**Panel 2: Forecast** (Symbol-specific)
- 3 cards: 24h, 2-5d, 7-14d
- Direction: ↑ BUY | ↓ SELL | → FLAT
- Probability: 75% (or `--`)
- Expected Move: +2.5% (or `--`)

**Panel 3: News Feed**
- 10 recent articles with sentiment (Bullish/Bearish/Neutral)
- Timestamps: "5m ago", "2h ago"
- OR: "No news available yet"

**Panel 4: Prediction Accuracy**
- Line chart showing historical accuracy
- OR: Empty chart with message

**Panel 5: Watchlist**
- Stocks: AAPL, NVDA, WOLF
- Crypto: BTC, ETH, SOL
- VIP: (if any)
- 24h change: `+2.5%` or `--`
- Ghost score: `85%` or `--`

**Panel 6: Ghost Health Score**
- Large number: `85` (or `--`)
- Grade: `B`
- 6 metric bars:
  - Daily Goal: 75%
  - Weekly Goal: 60%
  - Monthly Goal: 45%
  - Data Health: 85%
  - AI Activity: 75%
  - Accuracy: 70%

---

## What's Different from V3.0?

### Before (V3.0):
- News panel: Broken ❌ (called `/api/news/market` which didn't exist)
- Predictions: Broken ❌ (called `/api/predict/history` with wrong format)
- Watchlist: Messy fallback logic
- Health Score: Hardcoded 0.0, no transparency
- UX: Red "Failed to load" errors everywhere

### After (V3.1 "The Finisher"):
- News panel: ✅ Shows articles OR "No news available yet"
- Predictions: ✅ Shows history OR empty chart (no crash)
- Watchlist: ✅ Clean grouped view (stocks/crypto/vip)
- Health Score: ✅ Real calculation with penalty breakdown
- UX: ✅ Professional "temporarily unavailable" or "--" states

---

## API Testing (Optional)

### Test Ghost Score:
```bash
curl http://localhost:8080/api/v3/cockpit/status | jq
```
**Expected Output**:
```json
{
  "live": true,
  "ghost_health_score": 85,
  "ghost_health_grade": "B",
  "score_breakdown": {
    "base": 100,
    "provider_penalty": -10,
    "ai_activity_penalty": 0,
    "accuracy_penalty": -10,
    "freshness_penalty": 0
  },
  "score_components": {
    "providers_healthy": 2,
    "providers_total": 3,
    "ai_decisions_24h": 5,
    "accuracy_pct": 62.5,
    "data_age_minutes": 3
  }
}
```

### Test News:
```bash
curl "http://localhost:8080/api/v3/news/feed?limit=5" | jq
```

### Test Predictions:
```bash
curl "http://localhost:8080/api/v3/predictions/history?limit=10" | jq
```

### Test Watchlist:
```bash
curl http://localhost:8080/api/v3/watchlist | jq
```

### Test Daily Summary:
```bash
curl http://localhost:8080/api/v3/daily/summary | jq
```

---

## Understanding Ghost Score V1

### Formula:
```
Base: 100
Final = 100 + penalties (clamped to 0-100)
```

### Penalties:
| Component | Condition | Penalty |
|-----------|-----------|---------|
| Providers | All DOWN | -20 |
| | Some degraded | -10 |
| AI Activity | 0 predictions (24h) | -15 |
| Accuracy | <50% | -20 |
| | 50-65% | -10 |
| Data Freshness | >15 min old | -15 |

### Grades:
- **A** (90-100): Excellent
- **B** (80-89): Good
- **C** (70-79): Fair
- **D** (60-69): Poor
- **F** (0-59): Critical

### Example Calculation:
```
Scenario:
- 2/3 providers healthy (1 degraded) → -10
- 5 predictions in 24h → 0
- Accuracy = 62.5% → -10 (in 50-65% range)
- Data 3 minutes old → 0

Final Score: 100 - 10 - 10 = 80 (Grade B)
```

---

## Troubleshooting

### Problem: All panels show "--" or "No data yet"
**Solution**: Wait 3 full minutes after `docker compose up`. Ghost needs time to:
1. Load Python modules (30s)
2. Initialize database (30s)
3. Fetch crypto prices (60s)
4. Generate forecast grid (60s)

### Problem: Container crashes on startup
**Solution**: Rebuild without cache
```bash
docker compose build --no-cache app
```

### Problem: "Connection reset by peer"
**Solution**: Check logs for IP_ALLOWLIST config. Should see:
```
🔒 IP_ALLOWLIST config: enabled=False, ips=set()
```

### Problem: Health Score shows 0 or F
**Possible Causes**:
- Providers are DOWN → Check logs for provider errors
- No predictions in 24h → Run a prediction manually
- Data is stale (>15 min old) → Wait for next price update
- Accuracy <50% → Let system warm up, more predictions needed

---

## What's Next?

### You Can Now:
- ✅ Use Cockpit V3 daily as your Ghost Protocol dashboard
- ✅ Monitor Ghost health score in real-time
- ✅ Track opportunities, news, predictions
- ✅ See transparent score breakdown (what's affecting health)

### Future Enhancements (Not Required):
- WebSocket/SSE for live updates (no 5s polling)
- Real-time price changes in watchlist
- GPS score integration for watchlist
- Chart.js for better chart rendering
- Mobile-responsive design
- Cloud deployment (Railway)

---

## Files Modified (For Reference)

| File | Purpose | Lines Changed |
|------|---------|---------------|
| `api/cockpit_v3_live_endpoints.py` | Backend: 4 new endpoints + Ghost Score V1 | +515 |
| `static/cockpit_v3.js` | Frontend: Wire V3 + UX polish | ~50 |
| `COCKPIT_V3_1_FINISHER_COMPLETE.md` | Full documentation | +400 |
| `COCKPIT_V3_1_QUICK_START.md` | This guide | +200 |

---

## Support

If you see issues:
1. Check container logs: `docker compose logs app | tail -100`
2. Check browser console: F12 → Console tab (look for `[GHOST V3]` errors)
3. Test endpoints with curl commands above
4. Verify 3-minute warmup period completed

---

**Happy Ghost Hunting! 🤖🐺**

**Version**: V3.1 (The Finisher)  
**Status**: ✅ Production Ready  
**Date**: January 2025
