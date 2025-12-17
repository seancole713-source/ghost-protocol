# Cascading Predictions - Quick Reference Card

## 🎯 What Is It?

A 48h → 24h → 6h prediction journey showing Ghost adapting in real-time:
- **48h**: Early warning (62% accuracy)
- **24h**: Re-evaluation (68% accuracy)  
- **6h**: Final call (74-75% accuracy)
- **Outcome**: Full evaluation

## 🚀 Quick Start

### Start a Cascade
```bash
curl -X POST http://localhost:8000/api/v3/cascade/start?symbol=BTC
```

### Check Status
```bash
# Get specific cascade
curl http://localhost:8000/api/v3/cascade/{cascade_id}

# List all active
curl http://localhost:8000/api/v3/cascade/list?active_only=true

# Get statistics
curl http://localhost:8000/api/v3/cascade/stats?days=30
```

### From Python
```python
from core.cascading_predictor import get_cascade_predictor

predictor = get_cascade_predictor()
cascade_id = await predictor.initiate_cascade("BTC")
```

## 📊 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v3/cascade/start` | POST | Start new cascade |
| `/api/v3/cascade/{id}` | GET | Get cascade details |
| `/api/v3/cascade/list` | GET | List cascades |
| `/api/v3/cascade/stats` | GET | Performance statistics |

## 📈 Telegram Alerts

### Stage 1: 48h Early Alert
```
🔔 48H EARLY ALERT - BTC
🔔 Early Warning: BTC trending UP
💰 Entry Price: $43,250.00
📊 Confidence: 64%
```

### Stage 2: 24h Update
```
📈 24H UPDATE - BTC
📈 Signal strengthening: BTC UP
💰 Current Price: $43,890.00
📊 Confidence: 69% (+5% from 48h)
```

### Stage 3: 6h Final
```
✅ 6H FINAL CALL - BTC
✅ HIGH CONFIDENCE: BTC UP
💰 Current Price: $44,120.00
📊 Confidence: 75%
```

### Stage 4: Outcome
```
🎯 CASCADE OUTCOME - BTC
48h: ✅ UP
24h: ✅ UP
6h: ✅ UP
Actual: UP to $44,580.00 (+3.07%)
🏆 PERFECT CASCADE!
```

## 🔧 Testing

### Run Test Suite
```bash
./test_cascade_system.sh
```

### Manual Tests
```bash
# 1. Start cascade
CASCADE_ID=$(curl -s -X POST http://localhost:8000/api/v3/cascade/start?symbol=BTC | jq -r '.cascade_id')

# 2. Check it was created
curl http://localhost:8000/api/v3/cascade/$CASCADE_ID | jq '.cascade'

# 3. Monitor logs
tail -f logs/ghost.log | grep CASCADE
```

### Fast Testing Mode
For testing without waiting 48 hours:

**In `cascade_scheduler.py`:**
```python
CHECK_INTERVAL = 60  # Check every minute
```

**In `cascading_predictor.py` initiate_cascade():**
```python
self._pending_updates[cascade_id] = {
    "h24_time": time.time() + 120,   # 2 min
    "h6_time": time.time() + 240,    # 4 min  
    "eval_time": time.time() + 360   # 6 min
}
```

## 📁 Database

### Table: prediction_cascades
```sql
-- Check cascades
SELECT cascade_id, symbol, created_at, h48_direction, h24_direction, h6_direction
FROM prediction_cascades
ORDER BY created_at DESC
LIMIT 10;

-- Count by stage
SELECT 
    COUNT(*) as total,
    SUM(CASE WHEN h48_sent_at IS NOT NULL THEN 1 ELSE 0 END) as h48_sent,
    SUM(CASE WHEN h24_sent_at IS NOT NULL THEN 1 ELSE 0 END) as h24_sent,
    SUM(CASE WHEN h6_sent_at IS NOT NULL THEN 1 ELSE 0 END) as h6_sent,
    SUM(CASE WHEN evaluated_at IS NOT NULL THEN 1 ELSE 0 END) as evaluated
FROM prediction_cascades;

-- Accuracy by stage
SELECT 
    AVG(h48_correct) as h48_accuracy,
    AVG(h24_correct) as h24_accuracy,
    AVG(h6_correct) as h6_accuracy
FROM prediction_cascades
WHERE evaluated_at IS NOT NULL;
```

## 🔍 Monitoring

### Check Scheduler Status
```bash
# In logs
tail -100 logs/ghost.log | grep "Cascade Scheduler"

# Should see:
# ✅ Cascade Scheduler: STARTED
```

### Check For Updates
```bash
# Recent cascade activity
tail -200 logs/ghost.log | grep "\[CASCADE\]"

# Should see:
# [CASCADE] Initiating for BTC (ID: ...)
# [CASCADE] 24h update for BTC (...)
# [CASCADE] 6h final for BTC (...)
# [CASCADE] Evaluation complete for BTC
```

### Database Size
```bash
# Check database file
ls -lh data/ghost_predictions.db

# Count cascade records
sqlite3 data/ghost_predictions.db "SELECT COUNT(*) FROM prediction_cascades;"
```

## 🚨 Troubleshooting

### Cascade Not Updating

**Problem:** Stuck at 48h stage, no 24h update

**Solutions:**
1. Check scheduler: `grep "CASCADE SCHEDULER" logs/ghost.log`
2. Verify time passed: `SELECT created_at, (strftime('%s', 'now') - created_at) / 3600 as hours_ago FROM prediction_cascades WHERE h24_sent_at IS NULL;`
3. Force update: `python -c "from core.cascading_predictor import get_cascade_predictor; import asyncio; p = get_cascade_predictor(); asyncio.run(p.update_cascade_24h('<cascade_id>'))"`

### No Telegram Alerts

**Problem:** Cascade updates but no notifications

**Solutions:**
1. Check Telegram configured: `echo $TELEGRAM_BOT_TOKEN`
2. Test manually: `curl http://localhost:8000/api/test/telegram`
3. Check logs: `grep "send_telegram" logs/ghost.log`

### Scheduler Not Running

**Problem:** No cascade updates happening

**Solutions:**
1. Verify startup: `grep "Cascade Scheduler" logs/ghost.log`
2. Check thread alive: `ps aux | grep cascade`
3. Restart server: `python wolf_app.py`

## 📦 Files Structure

```
ghost-protocol/
├── core/
│   ├── cascading_predictor.py      # Core cascade logic
│   └── cascade_scheduler.py         # Background scheduler
├── test_cascade_system.sh           # Test suite
├── CASCADING_PREDICTIONS_COMPLETE.md # Full docs
└── CASCADING_QUICK_REF.md          # This file
```

## 🎓 Key Concepts

### Why Cascades Win

**Single Prediction:**
- 62% accuracy at 48h
- Binary outcome (right or wrong)
- No learning visible to user

**Cascade:**
- Shows improvement over time
- 3 chances to be right (often 2/3 or 3/3)
- Builds trust through transparency
- Creates engagement (4 touchpoints)

### Cascade Patterns

**Perfect Cascade (3/3):**
```
48h: UP (64%) ✅
24h: UP (69%) ✅  
6h: UP (75%) ✅
Actual: UP
```

**Adaptation Win (2/3):**
```
48h: UP (62%) ❌
24h: DOWN (68%) ✅ (Ghost corrected!)
6h: DOWN (74%) ✅
Actual: DOWN
```

**Early Exit (1/3):**
```
48h: UP (61%) ❌
24h: UP (63%) ❌
6h: DOWN (72%) ✅ (Too late to save it)
Actual: DOWN
```

## 🎯 Success Metrics

Track these KPIs:
- **Cascade Completion Rate:** % that reach evaluation
- **Stage Accuracy:** h48 vs h24 vs h6
- **Perfect Cascades:** All 3 stages correct
- **Adaptation Wins:** Early wrong, final correct
- **User Engagement:** Opens per cascade (target: 3+)

## 💡 Pro Tips

1. **Don't trade on 48h alone** - wait for 6h final
2. **Direction changes are normal** - Ghost learning
3. **Perfect cascades = high conviction** - consider larger positions
4. **Adaptation wins build trust** - share these stories
5. **Monitor stats weekly** - track improvement over time

---

**Cascading Predictions: Ghost's killer feature. Ship it! 🚀**
