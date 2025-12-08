# Ghost Protocol - PredictionStore Deployment Checklist

**Current Status:**✅ Phase 1 Complete - SQLite Active, Postgres Ready**Next Phase:**Dual-Write Validation

---

## Pre-Flight Verification (Complete ✅)

- [x] PredictionStore abstraction deployed and live
- [x] SQLite primary backend operational (11ms writes)
- [x] PostgreSQL backend fully implemented (9 methods, connection pooling)
- [x] Migration script ready (`scripts/migrate_predictions_to_postgres.py`)
- [x] BTC/ETH/XRP predictions flowing through abstraction
- [x] Production logs show `[SQLiteBackend]` messages (correct default)


---

## Phase 2A: Enable Dual-Write (SQLite Primary + Postgres Secondary)**Goal:**Validate PostgreSQL writes without switching primary backend

### Step 1: Test Migration Script Locally (Optional but Recommended)

```bash
cd ~/ghost-protocol
python3 scripts/migrate_predictions_to_postgres.py --dry-run

```text**Expected Output:**```text

SQLite records found:

  - Predictions: N
  - Points: M
  - Outcomes: K


[DRY RUN] Migration plan:
  ...
[DRY RUN] No changes will be made

```text**If it fails:**Check `DATABASE_URL` is set and `psycopg2` is installed.

---

### Step 2: Enable Dual-Write in Railway**In Railway Dashboard → ghost-protocol → Variables:**

**Add or Modify:**```text

PREDICTION_DUAL_WRITE=1

```text**Keep Existing:**```text

PREDICTION_STORE_ENGINE=sqlite (or leave unset - defaults to sqlite)
DATABASE_URL=postgresql://... (Railway Postgres URL)
GHOST_PREDICT_DB=/app/data/ghost_predictions.db

```text**Save**→ Railway will auto-redeploy (30-60 seconds)

---

### Step 3: Monitor Logs (2-4 Hours)**In Railway Dashboard → Deploy Logs:**

**✅ Success Indicators:**```json

{"message":"✅ Dual-write enabled: SQLite (primary) + PostgreSQL (secondary)"}
{"message":"[SQLiteBackend] Saved prediction 5 for BTC (25 points, 11ms)"}
{"message":"[DUAL-WRITE] [PostgresBackend] Saved prediction 5 for BTC (14ms)"}

```text**⚠️ Warning Signs (But Not Fatal):**```json

{"message":"[DUAL-WRITE] [PostgresBackend] Failed for ETH: timeout"}

```text

→ SQLite keeps working, Postgres issue logged**❌ Fatal Errors:**```json

{"message":"Failed to create prediction for BTC"}

```text

→**ROLLBACK:**Set `PREDICTION_DUAL_WRITE=0`, redeploy

---

### Step 4: Verify Postgres Data**Option A: Railway Dashboard → Database → Query:**

```sql

SELECT COUNT(*), MAX(run_at) FROM ghost_predictions;

```text

**Expected:**Count increasing, recent timestamps**Option B: psql from terminal:**```bash

# Get DATABASE_URL from Railway Variables

psql $DATABASE_URL

# Query

SELECT symbol, run_at, confidence, direction
FROM ghost_predictions
ORDER BY run_at DESC
LIMIT 5;

```text**Expected:**Recent BTC/ETH/XRP predictions (last 1-2 hours)

---

### Step 5: Validate API Still Works

```bash

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=BTC>>>>>

```text**Expected:**`200 OK` with prediction data

---

### ✅ Phase 2A Complete When

- [ ] Dual-write logs present for 2+ hours
- [ ] PostgreSQL data visible in database queries
- [ ] No increase in prediction errors
- [ ] API response times unchanged (<50ms)


---

## Phase 2B: Migrate Historical Data**Goal:**Move existing SQLite predictions to PostgreSQL

### Step 1: Dry-Run Migration

```bash

cd ~/ghost-protocol
python3 scripts/migrate_predictions_to_postgres.py --dry-run

```text**Review output:**Should show exact counts and migration plan.

---

### Step 2: Execute Migration**⚠️ IMPORTANT:**Temporarily set `PREDICTION_STORE_ENGINE=postgres` for migration script (don't deploy this to Railway yet)

```bash

export PREDICTION_STORE_ENGINE=postgres
export DATABASE_URL="postgresql://..." # From Railway
export GHOST_PREDICT_DB="./data/ghost_predictions.db"

python3 scripts/migrate_predictions_to_postgres.py --batch-size 100 --verify

```text**Expected Duration:**1-3 minutes for ~150 predictions**Expected Output:**```text

✅ Connected to backend: PostgresBackend
Migrating batch 1/2 (predictions 1-100)...
Migrating batch 2/2 (predictions 101-150)...

Migration Summary:

  - Predictions migrated: 150
  - Points migrated: 3750
  - Outcomes migrated: 12
  - Failed: 0


✅ Verification passed

```text

---

### Step 3: Verify Migration**PostgreSQL Query:**

```sql

SELECT
  symbol,
  COUNT(*) as prediction_count,
  MIN(run_at) as first,
  MAX(run_at) as last
FROM ghost_predictions
GROUP BY symbol
ORDER BY prediction_count DESC;

```text

**Expected:**All symbols (BTC, ETH, XRP, etc.) with historical counts

---

### Step 4: Clean Up Environment

```bash

unset PREDICTION_STORE_ENGINE  # Back to default (sqlite) for production

```text**⚠️ Do NOT set `PREDICTION_STORE_ENGINE=postgres` in Railway yet!**---

### ✅ Phase 2B Complete When

- [ ] Migration script reports 0 failures
- [ ] PostgreSQL counts match SQLite counts
- [ ] Historical predictions queryable in Postgres


---

## Phase 3: Switch to PostgreSQL Primary (with SQLite Safety Net)**Goal:**Make PostgreSQL the primary backend, keep SQLite as fallback

### Step 1: Update Railway Variables**In Railway Dashboard → Variables:**

**Change:**```text

PREDICTION_STORE_ENGINE=postgres

```text**Keep:**```text

PREDICTION_DUAL_WRITE=1 (keep SQLite as secondary for safety)
DATABASE_URL=postgresql://...
GHOST_PREDICT_DB=/app/data/ghost_predictions.db

```text**Save**→ Auto-redeploy

---

### Step 2: Monitor Logs (24-48 Hours)**✅ Success Indicators:**```json

{"message":"🎯 Using PostgreSQL backend for predictions"}
{"message":"✅ Dual-write enabled: PostgreSQL (primary) + SQLite (secondary)"}
{"message":"[PostgresBackend] Saved prediction 200 for BTC (25 points, 14ms)"}
{"message":"[DUAL-WRITE] [SQLiteBackend] Saved prediction 200 for BTC (11ms)"}

```text**Performance Check:**- PostgreSQL writes: 10-20ms (faster than SQLite due to pooling)

- SQLite dual-write: +10ms overhead (acceptable)
- API responses: <50ms (unchanged)


---

### Step 3: Validate API Endpoints

```bash

# Latest prediction

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=BTC>>>>>

# All predictions

curl <<<<<https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?limit=100>>>>>

```text**Expected:**Same responses as before, data from PostgreSQL

---

### Step 4: Verify Both Backends Writing**PostgreSQL:**```sql

SELECT symbol, run_at FROM ghost_predictions ORDER BY run_at DESC LIMIT 3;

```text**SQLite (Railway shell):**```bash

sqlite3 /app/data/ghost_predictions.db \
  "SELECT symbol, run_at FROM predictions ORDER BY run_at DESC LIMIT 3;"

```text**Expected:**Same predictions in both databases (dual-write working)

---

### ✅ Phase 3 Complete When

- [ ] PostgreSQL primary stable for 48+ hours
- [ ] No connection errors or timeouts
- [ ] Both databases receiving writes (dual-write active)
- [ ] API performance acceptable


---

## Phase 4: Disable Dual-Write (PostgreSQL Only)**Goal:**Run PostgreSQL-only for maximum performance

### Step 1: Disable Dual-Write**In Railway Dashboard → Variables:**

**Change:**```text

PREDICTION_DUAL_WRITE=0

```text**Keep:**```text

PREDICTION_STORE_ENGINE=postgres
DATABASE_URL=postgresql://...

```text**Save**→ Auto-redeploy

---

### Step 2: Monitor Performance**Expected Improvement:**- PostgreSQL writes: 10-15ms (dual-write overhead removed)

- Simplified logs (no dual-write messages)**Logs Should Show:**```json


{"message":"🎯 Using PostgreSQL backend for predictions"}
{"message":"[PostgresBackend] Saved prediction 250 for BTC (25 points, 12ms)"}

```text**No more `[DUAL-WRITE]` messages**✅

---

### Step 3: Archive SQLite Backup (Optional)

```bash

# In Railway shell or download locally

cp /app/data/ghost_predictions.db /app/backups/ghost_predictions_$(date +%Y%m%d).db

```text

---

### ✅ Phase 4 Complete When

- [ ] PostgreSQL-only mode stable for 48+ hours
- [ ] Performance improved (dual-write overhead gone)
- [ ] SQLite backup archived


---

## Rollback Procedures

### From Dual-Write (Any Phase)**In Railway Dashboard:**```text

PREDICTION_DUAL_WRITE=0

```text**Save**→ Redeploy → Dual-write disabled (SQLite or Postgres single-write only)

---

### From PostgreSQL Primary to SQLite Primary**In Railway Dashboard:**```text

PREDICTION_STORE_ENGINE=sqlite
PREDICTION_DUAL_WRITE=0 (or 1 to keep Postgres as secondary)

```text**Save**→ Redeploy →**Instant rollback**(0 data loss if dual-write was enabled)

---

### Emergency Rollback (Production Issue)**Fastest Path (30 seconds):**1. Railway Dashboard → Variables

1. Set `PREDICTION_STORE_ENGINE=sqlite`
2. Set `PREDICTION_DUAL_WRITE=0`
3. Click**Save**(auto-redeploy)
4. Wait 30-60 seconds for restart**Result:**Ghost back to SQLite primary, all data intact


---

## Troubleshooting

### Issue: Dual-write logs show Postgres failures**Symptoms:**```json

{"message":"[DUAL-WRITE] [PostgresBackend] Failed for BTC: connection refused"}

```text**Diagnosis:**- Secondary (Postgres) writes failing

- Primary (SQLite) still working ✅**Action:**- Check `DATABASE_URL` is correct in Railway Variables
- Check Railway Postgres database is running
- If persistent: Disable dual-write (`PREDICTION_DUAL_WRITE=0`)


---

### Issue: Prediction API returns 500 errors**Symptoms:**- `/api/v3/predictions/latest?symbol=BTC` returns 500

- Logs show `Failed to create prediction`**Diagnosis:**- Primary backend failure (SQLite or Postgres)**Action:**1.**Immediate rollback:**Set `PREDICTION_STORE_ENGINE=sqlite`, `PREDICTION_DUAL_WRITE=0`
1. Check logs for root cause (connection, schema, permissions)
2. Fix issue before retrying


---

### Issue: Migration script fails**Symptoms:**```text

Failed to initialize PostgreSQL backend: connection timeout

```text**Diagnosis:**- `DATABASE_URL` not set or incorrect

- PostgreSQL not reachable
- Missing `psycopg2` dependency**Action:**1. Verify `DATABASE_URL` is set: `echo $DATABASE_URL`
1. Test connection: `psql $DATABASE_URL`
2. Install dependencies: `pip install psycopg2-binary`
3. Re-run with `--dry-run` first


---

## Summary Timeline

|**Phase**|**Duration**|**Risk**|**Rollback Time**|
|-----------|-------------|---------|------------------|
|**2A: Dual-Write**| 2-4 hours | Low | Instant (30s) |
|**2B: Migration**| 5-10 minutes | Very Low | N/A (read-only) |
|**3: Postgres Primary**| 24-48 hours | Low | Instant (30s) |
|**4: Postgres Only** | Ongoing | Very Low | Instant (30s)*|*After 48+ hours in Postgres-only, SQLite data is stale.
Rollback requires re-migration (not instant).

---

## Key Contacts & Resources

**Documentation:**- Full verification report: `PREDICTION_STORE_VERIFICATION_REPORT.md`

- Phase 1 implementation: `PREDICTION_STORE_PHASE1_COMPLETE.md`**Code:**- Abstraction: `core/prediction_store.py`
- Service layer: `services/predictor.py`
- Migration script: `scripts/migrate_predictions_to_postgres.py`**Railway:**- Production URL: `https://ghost-protocol-production.up.railway.app`
- Database: `tender-benevolence` (PostgreSQL)
- Environment: `ghost-protocol` → Variables**Support:**- Ghost Protocol Storage & Reliability Engineer (this agent)


---**END OF CHECKLIST**

🚀 Ready to proceed with Phase 2A dual-write enablement when you are!
