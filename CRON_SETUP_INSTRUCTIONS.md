# 🕒 Cron Job Setup for Ghost Protocol

**Purpose**: Trigger daily prediction evaluations at 2 AM UTC
**Method**: External cron service (cron-job.org)
**Endpoint**: `POST <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/evaluate`>>>>>

---

## Option 1: cron-job.org (Recommended - Free & Easy)

### Step 1: Create Account

1. Go to: <<<<<https://cron-job.org/en/signup/>>>>>
2. Sign up with email (free account, no credit card required)
3. Verify email

### Step 2: Create Cron Job

1. Login and click **"Create cronjob"**
2. Configure:

   ```text
   Title: Ghost Protocol - Daily Prediction Evaluation
   URL: <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/evaluate>>>>>
   Method: POST
   Schedule:

     - Minute: 0
     - Hour: 2
     - Day: *- Month:*
     - Weekday: *


   Timezone: UTC
   Enabled: ✓

   ```text

3. Click **"Create"**### Step 3: Test Immediately

4. Click the job name
5. Click**"Execute now"**3. Wait ~10 seconds
6. Check "Execution history" for status**Expected Response**:

```json

{
  "ok": true,
  "evaluated": 12,
  "correct": 9,
  "accuracy": 0.75,
  "execution_time_s": 5.2
}

```text

---

## Option 2: GitHub Actions (Alternative)

### Step 1: Create Workflow File

In your repo, create `.github/workflows/daily-evaluation.yml`:

```yaml

name: Daily Prediction Evaluation

on:
  schedule:

    - cron: '0 2 ***'  # 2 AM UTC daily


  workflow_dispatch:  # Allow manual trigger

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:

      - name: Trigger Evaluation


        run: |
          curl -X POST \
            <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/evaluate>>>>> \
            -H "Content-Type: application/json" \
            -w "\nHTTP Status: %{http_code}\n"

```text

### Step 2: Enable in GitHub

1. Commit and push the workflow file
2. Go to repo → Actions tab
3. Enable workflows if needed
4. Test: Click "Daily Prediction Evaluation" → "Run workflow"


---

## Option 3: EasyCron (Another Free Service)

1. Go to: <<<<<https://www.easycron.com/>>>>>
2. Sign up (free tier: 1 job every 60 min)
3. Create cron job:


   ```text

   URL: <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/evaluate>>>>>
   Method: POST
   Schedule: 0 2 ***(daily at 2 AM UTC)

   ```text

---

## Testing Your Cron Job

### Manual Test via curl

```bash

curl -X POST <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/evaluate>>>>>

```text**Expected Response**:

```json

{
  "ok": true,
  "evaluated": 15,
  "correct": 11,
  "accuracy": 0.73,
  "execution_time_s": 4.8,
  "output": "...recent evaluator output..."
}

```text

### Check Railway Logs

```bash

railway logs --tail 50 | grep -i "evaluating\|evaluated"

```text

**Expected in logs**:

```text

🔍 Evaluating 15 expired predictions...
✅ Evaluated: 12/15
✅ Correct: 9/12 (75.0%)

```text

---

## Verification Checklist

After setting up cron job, verify it's working:

- [ ] Cron job created and enabled
- [ ] Test execution shows HTTP 200 response
- [ ] Response JSON contains `"ok": true`
- [ ] Railway logs show evaluation output
- [ ] Outcomes table growing (check next day)
- [ ] `/api/v3/accuracy/summary` returns real data


---

## Troubleshooting

### Issue: 500 Error Response

**Cause**: Evaluator script failing
**Check**: Railway logs for error details
**Fix**: Verify `scripts/evaluate_predictions.py` deployed correctly

### Issue: Timeout

**Cause**: Evaluation taking >60s
**Check**: Database size (too many predictions?)
**Fix**: Increase timeout or optimize evaluator

### Issue: No Predictions to Evaluate

**Cause**: All predictions too recent (need 48h to expire)
**Check**: Last prediction timestamp in DB
**Wait**: 48 hours after first predictions

### Issue: Cron Not Triggering

**Cause**: Service suspended or schedule wrong
**Check**: Cron service dashboard
**Fix**: Verify timezone is UTC and schedule format correct

---

## Success Metrics

Once cron is working correctly:

✅ **Daily**: Evaluator runs at 2 AM UTC
✅ **Logs**: Railway shows evaluation output
✅ **Database**: Outcomes table growing (~10-50 records/day)
✅ **API**: `/api/v3/accuracy/summary` returns real accuracy
✅ **Status**: **Ghost Protocol 100% OPERATIONAL**🎯

---**Estimated Setup Time**: 5 minutes
**Recommended Service**: cron-job.org (easiest)
**Cost**: Free forever

---

**Next Step**: Choose Option 1, 2, or 3 and set it up now! 🚀
