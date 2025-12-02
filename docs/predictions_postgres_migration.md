# Ghost Protocol Prediction Storage: PostgreSQL Migration

## Migration Status: ✅ COMPLETE

**Date Completed**: December 1, 2025  
**Migration Script**: `scripts/migrate_predictions_to_postgres.py`  
**Records Migrated**: 39 predictions, 1,221 forecast points

---

## Overview

Ghost Protocol v3 now uses **PostgreSQL** for prediction storage in production. Historical SQLite predictions have been successfully migrated to the PostgreSQL backend on Railway.

### What Was Migrated

✅ **Predictions** (39 records)
- Prediction metadata (symbol, confidence, direction, run_at, method)
- Feature JSON and parameter JSON
- Horizon and tags

✅ **Prediction Points** (1,221 records)
- Forecast points (timestamp, price pairs)
- Actual price points (for accuracy tracking)

⚠️ **Outcomes** (NOT migrated)
- Historical outcome records from SQLite remain in SQLite only
- This is **intentional** - outcomes are optional for live trading
- Future predictions will create outcomes in PostgreSQL when they close

---

## Architecture

### Backend Selection

The prediction storage backend is controlled by environment variables:

```bash
# Use PostgreSQL (production)
PREDICTION_STORE_ENGINE=postgres
DATABASE_URL=postgresql://user:pass@host:port/dbname

# Use SQLite (local development)
PREDICTION_STORE_ENGINE=sqlite
GHOST_PREDICT_DB=./data/ghost_predictions.db
```

### Database Schema

Both SQLite and PostgreSQL use the same logical schema:

**predictions** table:
- `id` (primary key)
- `symbol`, `run_at`, `horizon_h`
- `method`, `confidence`, `direction`
- `features_json`, `params_json`, `tag`

**prediction_points** table:
- `id` (primary key)
- `prediction_id` (foreign key)
- `ts`, `kind` (forecast/actual), `price`

**outcomes** table:
- `prediction_id` (primary key)
- `closed_at`, `mae`, `map`, `rmse`
- `hit_direction`, `hit_ratio_window`, `notes`

### Code Flow

1. **Prediction Creation** (`/api/predict/run`):
   - Calls `services.predictor.create_prediction()`
   - Uses `core.prediction_store.PredictionStore.save_prediction()`
   - Writes to PostgreSQL backend
   - Caches in `_LATEST_PREDICTIONS` in-memory dict

2. **Prediction Retrieval** (`/api/v3/predictions/latest`):
   - Reads from `_LATEST_PREDICTIONS` cache (fast)
   - Cache populated when predictions are created
   - No direct database queries for latest predictions

3. **Historical Queries**:
   - Use `PredictionStore.get_prediction_history(symbol, limit)`
   - Direct PostgreSQL queries for accuracy analysis

---

## Verification

### 1. Confirm Backend in Use

Check Railway logs for prediction creation:

```bash
railway logs --service ghost-protocol --lines 50 | grep "PostgresBackend"
```

Expected output:
```
[INFO] [PostgresBackend] Created prediction 50 for BTC: UP @ 0.46 confidence
[INFO] [PostgresBackend] Saved prediction 50 for BTC (25 points, 18ms)
```

### 2. Test Prediction API

```bash
# Create new prediction
curl -X POST "https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC"

# Get latest prediction
curl "https://ghost-protocol-production.up.railway.app/api/v3/predictions/latest?symbol=BTC"
```

Expected response:
```json
{
  "ok": true,
  "predictions": [
    {
      "symbol": "BTC",
      "direction": "UP",
      "confidence": 0.46,
      "expected_move": 2.3,
      "horizon_h": 48,
      "run_at": 1764598699.358
    }
  ],
  "count": 1
}
```

### 3. Verify PostgreSQL Data

Connect to Railway Postgres and query:

```bash
# Get Railway database shell
railway connect ghost-protocol --service <postgres-service-name>
```

```sql
-- Count predictions
SELECT COUNT(*) FROM predictions;
-- Should return: 39 (migrated) + new predictions

-- Get latest predictions by symbol
SELECT id, symbol, run_at, confidence, direction 
FROM predictions 
ORDER BY run_at DESC 
LIMIT 10;

-- Count forecast points
SELECT COUNT(*) FROM prediction_points WHERE kind='forecast';
-- Should return: 1,221 (migrated) + new points

-- Check for any duplicate predictions
SELECT symbol, COUNT(*) as count 
FROM predictions 
GROUP BY symbol 
HAVING COUNT(*) > 1;
```

---

## Migration Details

### Script Usage

**⚠️ DO NOT RE-RUN THE MIGRATION**

The migration has been completed. Running it again would create duplicate predictions.

If you need to verify the migration script:

```bash
# Dry-run mode (safe, no changes)
cd ~/ghost-protocol
export PREDICTION_STORE_ENGINE=postgres
export DATABASE_URL="postgresql://..."
export GHOST_PREDICT_DB="./data/ghost_predictions.db"
python3 scripts/migrate_predictions_to_postgres.py --dry-run
```

### Migration Logs

Final migration output (December 1, 2025):

```
2025-12-01 09:06:04,701 - INFO - SQLite → PostgreSQL Migration
2025-12-01 09:06:04,704 - INFO - SQLite records found:
2025-12-01 09:06:04,704 - INFO -   - Predictions: 39
2025-12-01 09:06:04,704 - INFO -   - Points: 1221
2025-12-01 09:06:04,704 - INFO -   - Outcomes: 38

[Migration execution logs...]

2025-12-01 09:06:XX,XXX - INFO - ✅ Migration complete!
2025-12-01 09:06:XX,XXX - INFO -   - Migrated: 39
2025-12-01 09:06:XX,XXX - INFO -   - Failed: 0
2025-12-01 09:06:XX,XXX - INFO -   - Total: 39
```

---

## Production Configuration

Current Railway environment variables:

```bash
# PostgreSQL backend enabled
PREDICTION_STORE_ENGINE=postgres

# PostgreSQL connection (Railway internal + external)
DATABASE_URL=postgresql://postgres:***@metro.proxy.rlwy.net:28328/railway

# SQLite path (used for local dev only)
GHOST_PREDICT_DB=/app/data/ghost_predictions.db

# Dual-write mode (optional, currently disabled)
PREDICTION_DUAL_WRITE=0
```

---

## Troubleshooting

### Issue: Predictions not saving to PostgreSQL

**Check**:
1. `PREDICTION_STORE_ENGINE=postgres` in Railway variables
2. `DATABASE_URL` is set and accessible
3. Railway logs show `[PostgresBackend]` messages

**Fix**:
```bash
# Verify environment
railway run --service ghost-protocol env | grep PREDICTION

# Check logs
railway logs --service ghost-protocol --lines 100 | grep -i "postgres\|sqlite"
```

### Issue: API returns empty predictions

**Check**:
1. In-memory cache `_LATEST_PREDICTIONS` populated
2. At least one prediction created since deployment

**Fix**:
```bash
# Create a test prediction
curl -X POST "https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC"

# Verify cache
railway logs --service ghost-protocol --lines 50 | grep "prediction_id"
```

### Issue: Historical predictions missing

**Check**:
1. Migration completed successfully
2. PostgreSQL contains expected record count

**Fix**:
```sql
-- Connect to Railway Postgres
SELECT COUNT(*) FROM predictions;  -- Should be >= 39
SELECT COUNT(*) FROM prediction_points;  -- Should be >= 1221
```

---

## Next Steps (Optional)

### Phase 2: Dual-Write Mode

If you need to test PostgreSQL before fully switching:

```bash
# Railway variables
PREDICTION_STORE_ENGINE=sqlite  # Primary backend
PREDICTION_DUAL_WRITE=1         # Write to both backends

# This will:
# - Use SQLite as primary (fast reads)
# - Write to both SQLite and PostgreSQL (data sync)
# - Log any dual-write failures (non-blocking)
```

### Phase 3: Full PostgreSQL Cutover

Already complete! Production is using PostgreSQL as primary backend.

---

## References

- **Migration Script**: `scripts/migrate_predictions_to_postgres.py`
- **Prediction Store**: `core/prediction_store.py`
- **Predictor Service**: `services/predictor.py`
- **API Endpoints**: `wolf_app.py` (lines 5822, 6619)

---

## Support

For issues or questions:
1. Check Railway logs: `railway logs --service ghost-protocol`
2. Review this document's troubleshooting section
3. Verify PostgreSQL connectivity from Railway environment

**Last Updated**: December 1, 2025
