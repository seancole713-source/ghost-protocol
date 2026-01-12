# 🎯 GHOST PROTOCOL V2 - QUICK START GUIDE

**TL;DR**: V2 = Quality over Quantity. Verify performance, filter assets, send only high-conviction picks.

---

## 🚀 IMMEDIATE ACTIONS (Do These First!)

### 1. Check Current Win Rate (5 min)
```bash
# Get verified performance from last 14 days
curl "https://ghost-protocol-production.up.railway.app/api/v2/performance/dashboard?days=14" | python3 -m json.tool

# You'll see:
# - Total predictions
# - Actual win rate (not guessed, from database!)
# - Top 10 performing symbols
# - Bottom 10 performing symbols
```

**What to look for**:
- If win rate < 50%: URGENT - need to blacklist poor performers
- Top performers: These should be whitelisted
- Bottom performers: These should be blacklisted

---

### 2. Update Quality Filters (2 min)
```bash
# This reads last 30 days of data and sets whitelist/blacklist
curl -X POST "https://ghost-protocol-production.up.railway.app/api/v2/quality/update?days=30"

# Returns:
# - Whitelist count (WR >= 55%)
# - Blacklist count (WR < 45%)
# - Changes made
```

**This should be run**:
- Daily (automated via cron)
- After major model changes
- When win rate drops

---

### 3. Check Quality Status (1 min)
```bash
# See current whitelist/blacklist
curl "https://ghost-protocol-production.up.railway.app/api/v2/quality/status" | python3 -m json.tool

# Shows:
# - Which assets are approved (whitelist)
# - Which assets are blocked (blacklist)
# - Filter configuration
```

---

## 📊 WHAT V2 DOES

### Before V2:
```
1. Model predicts 400 assets
2. Sort by confidence
3. Send TOP 20 (10 stocks + 10 crypto)
4. No quality checks
5. ~50-60% win rate (with false alerts)
```

### After V2:
```
1. Model predicts 400 assets
2. V2 Quality Filter runs 5 gates:
   ✅ Gate 1: Check whitelist/blacklist
   ✅ Gate 2: Require 3+ signals align
   ✅ Gate 3: Check market conditions
   ✅ Gate 4: Verify confidence requirements
   ✅ Gate 5: Calculate conviction score
3. Only 3-5 picks pass all gates
4. If <2 picks pass → Send NO predictions today
5. TARGET: 70%+ win rate
```

---

## 🎯 THE 5 QUALITY GATES

### Gate 1: Symbol Quality
```python
# Checks asset status
if symbol in blacklist:
    REJECT  # WR < 45%, historically bad

if symbol in whitelist:
    PASS    # WR >= 55%, proven performer

if symbol in watchlist:
    PASS if confidence >= 80%  # WR 45-55%, need high confidence
    REJECT otherwise

if symbol unknown:
    PASS if confidence >= 75%  # New asset, cautious
    REJECT otherwise
```

### Gate 2: Signal Alignment
```python
# Count how many signals agree with direction
signals = ["MACD_bullish", "RSI_oversold", "Volume_spike", "Trend_up"]
direction = "UP"

aligned = 4  # All 4 signals agree

if aligned < 3:
    REJECT  # Insufficient confirmation
```

### Gate 3: Market Condition
```python
if market_condition == "choppy":
    REJECT  # Poor environment for predictions
```

### Gate 4: Confidence by Status
```python
# Different requirements based on asset status
if whitelist:
    min_confidence = 0.60  # Proven, can be lower
elif watchlist:
    min_confidence = 0.80  # Unproven, need high
elif unknown:
    min_confidence = 0.75  # New, cautious
```

### Gate 5: Conviction Score
```python
# Blend multiple factors
conviction = (
    confidence * 0.6 +                    # Model confidence
    (signal_alignment / 5) * 0.2 +        # Signal strength
    (historical_wr - 0.5) * 0.4           # Historical accuracy
)

if conviction < 0.70:
    REJECT  # Not confident enough
```

---

## 🧪 TESTING THE SYSTEM

### Test Pick Filter Locally:
```python
from core.v2_pick_filter import get_pick_filter

filter = get_pick_filter()

# Test a prediction
test_prediction = {
    "symbol": "BTC",
    "direction": "UP",
    "confidence": 0.85,
    "signals": [
        {"name": "MACD", "direction": "UP"},
        {"name": "RSI", "direction": "UP"},
        {"name": "Volume", "direction": "UP"},
        {"name": "Trend", "direction": "UP"}
    ],
    "market_condition": "trending",
    "current": 45000
}

# Run through gates
pick = filter.evaluate_pick(test_prediction)

if pick:
    print(f"✅ PASSED: {pick.symbol}, conviction {pick.conviction_score:.2f}")
else:
    print("❌ REJECTED")
```

### Test Daily Selection:
```python
# Simulate daily pick selection
candidates = [...]  # 100+ predictions

# Apply V2 filter
picks = filter.select_daily_picks(candidates, max_picks=5)

# Check if we should send
should_send, reason = filter.should_send_predictions_today(picks)

if should_send:
    print(f"✅ Send {len(picks)} picks today")
else:
    print(f"🔇 Skip today: {reason}")
```

---

## 📈 MONITORING PERFORMANCE

### Daily:
```bash
# Check yesterday's results
curl "https://ghost-protocol-production.up.railway.app/api/v2/performance/dashboard?days=1"
```

### Weekly:
```bash
# Get 7-day performance
curl "https://ghost-protocol-production.up.railway.app/api/v2/performance/dashboard?days=7"

# See which assets are improving/declining
# Update whitelist/blacklist if needed
```

### Monthly:
```bash
# Full 30-day analysis
curl "https://ghost-protocol-production.up.railway.app/api/v2/performance/dashboard?days=30"

# Get recommendations
curl "https://ghost-protocol-production.up.railway.app/api/v2/recommendations?days=30"

# Apply recommendations
curl -X POST "https://ghost-protocol-production.up.railway.app/api/v2/quality/update?days=30"
```

---

## 🔧 INTEGRATION GUIDE

### Integrate V2 into Daily TOP 10:

**File**: `core/ghost_notifications.py` or `core/top10_aggregator.py`

**Before**:
```python
def send_top10():
    # Get all predictions
    predictions = get_high_confidence_predictions()
    
    # Sort by confidence
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Send top 20
    top = predictions[:20]
    send_telegram(format_message(top))
```

**After**:
```python
def send_top10():
    from core.v2_pick_filter import get_pick_filter
    
    # Get all predictions
    predictions = get_high_confidence_predictions()
    
    # Apply V2 quality filter
    filter = get_pick_filter()
    picks = filter.select_daily_picks(predictions, max_picks=5)
    
    # Check if we should send
    should_send, reason = filter.should_send_predictions_today(picks)
    
    if not should_send:
        send_telegram(f"🔇 Ghost sitting out today. {reason}")
        return False
    
    # Send only high-quality picks (3-5)
    send_telegram(format_v2_message(picks))
    return True
```

---

## 💬 NEW MESSAGE FORMAT

**Current Format** (20 picks):
```
🎯 TOP 10 STOCKS + TOP 10 CRYPTO
[Long list of 20 predictions]
```

**V2 Format** (3-5 picks):
```
🎯 GHOST TOP PICKS — Jan 12, 2026

Only 3 high-conviction setups today:

1. 🟢 NVDA — BUY
   Entry: $185.20 → Target: $190.75 (+3%)
   Confidence: 76% | Conviction: 87%
   Signals: Momentum ✓ Sector ✓ Volume ✓ RSI ✓
   Status: Whitelist (61% historical WR)

2. 🟢 META — BUY
   Entry: $650.41 → Target: $669.92 (+3%)
   Confidence: 74% | Conviction: 82%
   Signals: MACD ✓ Trend ✓ Volume ✓
   Status: Whitelist (58% historical WR)

3. 🔴 ETH — SELL
   Entry: $2,280 → Target: $2,212 (-3%)
   Confidence: 75% | Conviction: 80%
   Signals: RSI overbought ✓ Funding ✓ BTC weak ✓
   Status: Whitelist (56% historical WR)

━━━━━━━━━━━━━━━━━━━━━
📊 Last 7 Days: 12/19 wins (63.2%)
🎯 Only 75%+ conviction picks
Ghost watching 25 assets more carefully.
```

---

## 🚨 WHAT TO WATCH FOR

### Good Signs ✅:
- Win rate trending up week-over-week
- Fewer daily picks but higher quality
- Whitelist growing (more proven performers)
- Conviction scores correlate with win rate

### Bad Signs ⚠️:
- Win rate still < 55% after 2 weeks
- Blacklist keeps growing (losing assets)
- No "no prediction" days (filter too loose)
- Conviction scores don't correlate with outcomes

### Emergency Actions 🔴:
If win rate < 50% after 1 week of V2:
1. Check if quality filter is actually running
2. Verify whitelist/blacklist are up to date
3. Increase MIN_SIGNAL_ALIGNMENT to 4
4. Increase MIN_CONVICTION_SCORE to 0.80
5. Reduce MAX_DAILY_PICKS to 3

---

## 📋 MAINTENANCE CHECKLIST

### Daily (Automated):
- [ ] Update quality filters from last 30 days
- [ ] Run V2 pick filter on new predictions
- [ ] Send 3-5 picks OR skip day if quality low

### Weekly (Manual):
- [ ] Review performance dashboard
- [ ] Check top/bottom performers
- [ ] Verify whitelist/blacklist accuracy
- [ ] Adjust MIN_SIGNAL_ALIGNMENT if needed

### Monthly (Manual):
- [ ] Full 30-day performance analysis
- [ ] Review blacklist (remove if improving)
- [ ] A/B test filter parameter changes
- [ ] Update model weights based on results

---

## 🎯 SUCCESS CRITERIA

**Week 1**:
- [ ] V2 filter is running
- [ ] Sending 3-5 picks daily (or skipping low-quality days)
- [ ] Win rate >= 50%

**Week 2**:
- [ ] Win rate >= 55%
- [ ] At least 2 "no prediction" days
- [ ] Whitelist has 15-25 assets

**Month 1**:
- [ ] Win rate >= 60%
- [ ] Blacklist stable (not growing)
- [ ] Signal alignment correlates with wins

**Month 3**:
- [ ] 🎯 **WIN RATE >= 70%**
- [ ] Consistent 3-5 high-quality daily picks
- [ ] Ghost Protocol V2 = Proven system

---

## 💡 PRO TIPS

1. **Trust the Filter**: If it says "skip today", skip. Low-quality days hurt more than they help.

2. **Watch Conviction Scores**: If picks consistently have conviction > 0.85, we're on track.

3. **Whitelist = Gold**: Protect these assets. They're proven performers.

4. **Blacklist = Poison**: Never override. If WR < 45%, we have NO edge.

5. **Signal Alignment Matters**: 5 aligned signals >> 1 strong signal.

6. **Update Filters Often**: Performance changes. Keep whitelist/blacklist current.

7. **No Shame in Skipping**: "Ghost sitting out today" is better than bad predictions.

---

## 🆘 TROUBLESHOOTING

**"No picks passing quality gates"**:
- Check if whitelist is empty (run POST /api/v2/quality/update)
- Verify MIN_SIGNAL_ALIGNMENT isn't too high (should be 3)
- Check if all symbols are blacklisted (query /api/v2/quality/status)

**"Win rate not improving"**:
- Verify quality filter is actually being used (check logs)
- Ensure whitelist/blacklist are up to date
- Increase MIN_CONVICTION_SCORE from 0.70 to 0.75

**"Too many 'skip today' messages"**:
- Whitelist might be too small (lower WR threshold to 52%)
- MIN_CONVICTION_SCORE might be too high (lower to 0.65)
- Signal alignment too strict (lower to 2 temporarily)

---

**The Bottom Line**: Ghost Protocol V2 is about predicting BETTER, not MORE. Quality over quantity. 70%+ win rate is achievable when we only predict what we're good at.

Let's get it. 🚀
