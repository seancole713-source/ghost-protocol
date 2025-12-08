# 🧪 GHOST PREDICTION + SCORING FIX – TEST PLAN

**Date**: November 20, 2025
**Status**: READY FOR DEPLOYMENT
**Risk Level**: LOW (no execution/trading changes)

---

## ✅ PRE-DEPLOYMENT SAFETY CHECKS

### 1. Verify No Execution Changes

```bash
cd /Users/studio713/ghost-protocol

# Confirm no AUTO_TRADE or SIM_MODE logic modified

git diff wolf_app.py | grep -E "(AUTO_TRADE|SIM_MODE|execute_order|_broker_execute)"

# Expected: No matches (or only in comments/logging)

```text

**Status**: ✅ SAFE - No execution logic modified

---

### 2. Verify No Trading Logic Changes

```bash

# Check that broker, order, and trade functions untouched

git diff wolf_app.py | grep -E "(def.*order|def.*trade|def.*execute)"

# Expected: No new trading functions

```text

**Status**: ✅ SAFE - No trading functions added

---

### 3. Review All Changes

```bash

# View complete diff

git diff wolf_app.py core/telegram_alerts.py core/prediction_tracker.py

# Count lines changed

git diff --stat

```text

**Expected Changes**:

- `wolf_app.py`: ~60 lines added (prediction logging, telegram commands, morning report)
- `core/telegram_alerts.py`: ~6 lines added (0% confidence filter)
- `core/prediction_tracker.py`: ~1 line modified (confidence filter in SQL)
- `GHOST_SCORING_AUTOPSY.md`: New file (documentation)
- `GHOST_SCORING_TEST_PLAN.md`: New file (this file)


---

## 🧪 POST-DEPLOYMENT TEST PLAN

### Test 1: 0% Confidence Filter

**Purpose**: Verify low-confidence predictions are not sent as alerts

**Steps**:

1. Deploy to Railway
2. Wait for next scheduled prediction run (8:00 AM, 12:00 PM, or 4:00 PM ET)
3. Check Telegram for alerts
4. Check logs for "Skipping 0% confidence" messages


**Expected Results**:

- Telegram receives only predictions with confidence ≥ 10%
- Logs show: `"Hunter prediction skipped (0% confidence): WOLF"`
- Morning Report excludes 0% predictions from accuracy stats


**How to Verify**:

```bash

# Check Railway logs

railway logs --tail 100 | grep "confidence"

# Should see lines like

# "Hunter prediction generated: AAPL (confidence: 65%)"

# "Hunter prediction skipped (0% confidence): WOLF"

```text

---

### Test 2: Telegram /predict Command

**Purpose**: Verify manual prediction trigger works

**Steps**:

1. Open Telegram bot chat
2. Send: `/predict`
3. Observe response


**Expected Results**:

- Bot responds: "🔮 Generating prediction now..."
- Predictions generated for all HUNTER symbols
- Alert sent if any have confidence ≥ 10%
- No error messages


**How to Verify**:

```bash

# Check Railway logs for force_multi_prediction

railway logs --tail 50 | grep "multi-prediction"

# Should see

# "[MULTI-PREDICTION] 🔮 Generating predictions at..."

# "[MULTI-PREDICTION] ✅ Predictions generated successfully"

```text

---

### Test 3: Telegram /check Command

**Purpose**: Verify accuracy check works

**Steps**:

1. Open Telegram bot chat
2. Send: `/predict` (to generate a prediction first)
3. Wait 1 minute
4. Send: `/check`
5. Observe comparison


**Expected Results**:

- Bot responds with formatted comparison:


  ```text

  ⚠️ PREDICTION CHECK

  PREDICTED:
    Direction: UP
    Price: $17.51
    Confidence: 65%

  ACTUAL:
    Direction: UP
    Price: $18.42 (+5.19%)

  RESULT: ✅ CORRECT

  ```text

- Uses same provider for consistency
- Shows actual market movement


**How to Verify**:
Take screenshot and compare with user's expected format

---

### Test 4: Morning Report Accuracy

**Purpose**: Verify Morning Report shows correct prediction count and accuracy

**Steps**:

1. Generate several predictions with `/predict` command
2. Wait for next Morning Report (7:00 AM next business day)
3. Check Telegram for Morning Report message


**Expected Results**:

- "Ghost Accuracy" shows actual percentage (not 0.0%)
- "(N predictions)" shows actual count (not 0)
- "Top Opportunities" section shows high-confidence predictions (≥70%)
- "No high-quality opportunities" only appears if truly none exist


**How to Verify**:

```bash

# Check ghost_predictions table

railway run sqlite3 data/wolf.db "SELECT COUNT(*) FROM ghost_predictions WHERE confidence >= 0.10;"

# Expected: Non-zero count

# Check _LATEST_PREDICTIONS in memory

curl <<<<<https://ghost-protocol-production.up.railway.app/api/debug/predictions>>>>> | jq '.count'

# Expected: Non-zero count

```text

---

### Test 5: High-Confidence Predictions in Morning Report

**Purpose**: Verify 70%+ predictions appear in Morning Report

**Steps**:

1. Generate predictions for multiple symbols
2. Check which have confidence ≥ 70%
3. Wait for Morning Report
4. Verify those symbols appear in "Top Opportunities"


**Expected Results**:

- Morning Report lists symbols with 70%+ confidence
- Shows action (BUY/SELL/HOLD)
- Shows predicted % move
- Sorted by confidence (highest first)


**How to Verify**:

```bash

# Check current high-confidence predictions

curl <<<<<https://ghost-protocol-production.up.railway.app/api/hunter/snapshot>>>>> | jq '.predictions.stocks[] | select(.confidence >= 70)'

# Expected: List of stocks with confidence ≥ 70

```text

---

### Test 6: Provider Consistency

**Purpose**: Verify evaluation uses same provider as prediction

**Steps**:

1. Generate prediction (note provider in logs)
2. Wait 1 hour
3. Run `/check` command
4. Verify evaluation used same provider


**Expected Results**:

- Provider stored with prediction: `"provider": "yahoo"`
- Evaluation attempts to use same provider
- If provider unavailable, marked as "unscorable" (not incorrect)


**How to Verify**:

```bash

# Check _LATEST_PREDICTIONS

curl <<<<<https://ghost-protocol-production.up.railway.app/api/debug/predictions>>>>> | jq '.store.WOLF.provider'

# Expected: "yahoo" or "polygon" (provider name)

```text

---

### Test 7: Accuracy Exclusion of 0% Confidence

**Purpose**: Verify 0% predictions don't affect accuracy stats

**Steps**:

1. Check current accuracy
2. Generate mix of 0% and >10% predictions
3. Wait for evaluation
4. Check accuracy again


**Expected Results**:

- Only predictions with confidence ≥ 10% counted
- 0% predictions ignored in accuracy calculation
- Morning Report reflects accurate stats


**How to Verify**:

```bash

# Query ghost_predictions table

railway run sqlite3 data/wolf.db "SELECT confidence, correct, checked FROM ghost_predictions ORDER BY predicted_at DESC
LIMIT 10;"

# Expected: No rows with confidence < 0.10

```text

---

### Test 8: Ghost Predictions Table Population

**Purpose**: Verify predictions are logged to ghost_predictions table

**Steps**:

1. Check current row count
2. Generate prediction with `/predict`
3. Check row count again


**Expected Results**:

- New row added to ghost_predictions table
- Only if confidence ≥ 10%
- Contains symbol, direction, confidence, timeframe


**How to Verify**:

```bash

# Before

railway run sqlite3 data/wolf.db "SELECT COUNT(*) FROM ghost_predictions;"

# Generate prediction via Telegram: /predict

# After

railway run sqlite3 data/wolf.db "SELECT COUNT(*) FROM ghost_predictions;"

# Expected: Count increased by number of symbols with confidence ≥ 10%

```text

---

## 🔍 SMOKE TESTS (Quick Validation)

```bash

# 1. Health check

curl <<<<<https://ghost-protocol-production.up.railway.app/api/health>>>>>

# 2. Cockpit snapshot (should show predictions)

curl <<<<<https://ghost-protocol-production.up.railway.app/api/cockpit/snapshot>>>>> | jq '.predictions'

# 3. Hunter snapshot (should show filtered predictions)

curl <<<<<https://ghost-protocol-production.up.railway.app/api/hunter/snapshot>>>>> | jq '.predictions'

# 4. Debug predictions endpoint

curl <<<<<https://ghost-protocol-production.up.railway.app/api/debug/predictions>>>>> | jq

# All should return 200 OK with valid JSON

```text

---

## 📊 EXPECTED OUTCOMES

### Success Criteria

✅ **0% Confidence Policy Enforced**- No alerts sent for confidence < 10%

- No 0% predictions in Morning Report accuracy
- Logs show "skipped" messages for low-confidence


✅**Telegram Commands Working**- `/predict` triggers predictions successfully

- `/check` shows accurate comparison
- No "function doesn't exist" errors


✅**Morning Report Accurate**- Shows actual prediction count (not 0)

- Shows actual accuracy percentage (not 0.0%)
- Lists high-confidence opportunities (≥70%)


✅**Provider Consistency**- Predictions store provider info

- Evaluation uses same provider when possible


✅**Prediction Logging Wired**- ghost_predictions table populates automatically

- Only for predictions with confidence ≥ 10%


---

## 🚨 FAILURE SCENARIOS

### Scenario 1: Telegram Commands Fail**Symptoms**: `/predict` or `/check` returns error

**Diagnosis**:

```bash

railway logs --tail 100 | grep "Error"

```text

**Fix**: Check for import errors or function signature mismatches

---

### Scenario 2: Morning Report Still Shows 0.0%

**Symptoms**: Morning Report says "0.0% (0 predictions)"

**Diagnosis**:

```bash

# Check ghost_predictions table

railway run sqlite3 data/wolf.db "SELECT * FROM ghost_predictions LIMIT 5;"

# Check if log_prediction is actually called

railway logs --tail 500 | grep "log_prediction"

```text

**Fix**: Verify log_prediction is being called in api_predict_run

---

### Scenario 3: All Predictions Filtered Out

**Symptoms**: No alerts sent, Morning Report says "no opportunities"

**Diagnosis**:

```bash

# Check prediction confidence levels

curl <<<<<https://ghost-protocol-production.up.railway.app/api/debug/predictions>>>>> | jq '.store[].confidence'

# Are all predictions < 0.10

```text

**Fix**: May indicate data quality issue - check price providers

---

## 📝 ROLLBACK PLAN

If critical issues found:

```bash

# 1. Revert changes

cd /Users/studio713/ghost-protocol
git log --oneline -5
git revert <commit-hash>
git push origin main

# 2. Railway auto-deploys reverted code

# 3. Verify rollback

railway logs --tail 20

# 4. Test basic functionality

curl <<<<<https://ghost-protocol-production.up.railway.app/api/health>>>>>

```text

---

## ✅ DEPLOYMENT CHECKLIST

- [x] All code changes reviewed
- [x] No execution/trading logic modified
- [x] Documentation created (GHOST_SCORING_AUTOPSY.md)
- [x] Test plan created (this file)
- [x] Safety checks defined
- [ ] Code committed to git
- [ ] Pushed to Railway (auto-deploy)
- [ ] Smoke tests passed
- [ ] Telegram commands tested
- [ ] Morning Report verified (next day)
- [ ] Accuracy tracking validated (after 24h)


---

## 📅 TEST SCHEDULE

**Day 1 (Deployment Day)**:

- [ ] Deploy code
- [ ] Run smoke tests
- [ ] Test `/predict` command
- [ ] Test `/check` command
- [ ] Verify alert filtering (next scheduled run)


**Day 2 (Next Business Day)**:

- [ ] Check Morning Report (7:00 AM)
- [ ] Verify accuracy percentage not 0.0%
- [ ] Verify opportunity list populated
- [ ] Check ghost_predictions table row count


**Day 3-7 (Ongoing Validation)**:

- [ ] Monitor accuracy trends
- [ ] Verify no false positives (0% in alerts)
- [ ] Check provider consistency in evaluations
- [ ] Collect user feedback on messaging clarity


---

**Status**: READY FOR DEPLOYMENT
**Next Action**: Commit changes and push to Railway

