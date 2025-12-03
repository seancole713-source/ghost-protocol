# 🔧 Railway Deployment Troubleshooting Plan

## If Deployment Still Fails After commit 63e88e5

### Phase 1: Immediate Diagnostics (30 seconds)

```bash
bash scripts/emergency_fallback.sh
```

This will test:
- ✅ Is server responding at all?
- ✅ Does `/health` return valid JSON?
- ✅ Are watchlist endpoints working?
- ✅ Is `/api/recent_alerts` still blocked?

### Phase 2: Check Railway Logs (2 minutes)

**Railway Dashboard → Deployments → Latest Failed → Deploy Logs**

Look for these critical errors:

#### A. PostgreSQL Connection Issues
```
Error: connection to postgres.railway.internal:5432 failed
```
**Fix:** Check PostgreSQL service is running, verify DATABASE_URL env var

#### B. Import Errors
```
ModuleNotFoundError: No module named 'X'
```
**Fix:** Check requirements.txt has all dependencies, rebuild may be needed

#### C. Still SQL Syntax Errors
```
ERROR: syntax error at or near "..."
```
**Fix:** More SQL bugs in personal_watchlist.py - run full audit

#### D. Timeout During Startup
```
[GHOST STARTUP] Beginning initialization...
[No further logs, then timeout]
```
**Fix:** Something blocking during startup (Redis, PostgreSQL, external API call)

### Phase 3: Emergency Workarounds

#### Option A: Disable Watchlist Scheduler (SAFEST)

```bash
bash scripts/disable_watchlist_scheduler.sh
git add wolf_app.py
git commit -m "Emergency: disable watchlist scheduler to unblock deployment"
git push origin main
```

**Effect:**
- ✅ Server will start normally
- ✅ All other endpoints work
- ❌ No automatic watchlist predictions
- ✅ Manual predictions still work

#### Option B: Rollback to Last Working Commit

```bash
# Find last successful deployment
git log --oneline -20

# Rollback (example: to commit abc1234)
git revert HEAD --no-edit
git push origin main
```

#### Option C: Skip Watchlist Tables (Nuclear Option)

Edit `core/migration_runner.py` to skip watchlist migration:

```python
# Skip personal watchlist migration temporarily
if "001_personal_watchlist" in migration_file:
    LOGGER.warning("⚠️ Skipping watchlist migration (emergency mode)")
    continue
```

Then commit and push.

### Phase 4: Deep Dive SQL Audit

If SQL errors persist, run comprehensive check:

```bash
# Check all SQL queries in personal_watchlist.py
grep -n "execute" core/personal_watchlist.py | grep -i "interval\|%s"

# Look for other parameter binding issues
grep -n "cursor.execute" core/personal_watchlist.py -A 5 | grep "%s"
```

Common PostgreSQL gotchas:
- ❌ `INTERVAL '%s minutes'` - Cannot parameterize inside string
- ❌ `... VALUES (%s, %s, %s)` with wrong tuple length
- ❌ Using `?` placeholders (SQLite syntax) instead of `%s`
- ❌ Missing `::` type casts for JSON/JSONB columns

### Phase 5: Fallback Architecture

If watchlist keeps failing, implement lazy initialization:

```python
# In wolf_app.py startup
try:
    from core.personal_watchlist import get_personal_watchlist_manager
    WATCHLIST_ENABLED = True
except Exception as e:
    LOGGER.warning(f"⚠️ Watchlist disabled: {e}")
    WATCHLIST_ENABLED = False

# Only start scheduler if enabled
if WATCHLIST_ENABLED:
    # Start watchlist scheduler
    pass
```

### Phase 6: Alternative Fixes

#### Fix 1: Increase Healthcheck Timeout
Edit `railway.toml`:
```toml
[deploy]
healthcheckTimeout = 300  # Increase from 100 to 300 seconds
```

#### Fix 2: Change Healthcheck Path
Use simpler endpoint that doesn't depend on database:
```toml
[deploy]
healthcheckPath = "/api/status"  # Simpler than /health
```

#### Fix 3: Disable Healthcheck Temporarily
```toml
[deploy]
# Comment out healthcheck to allow startup
# healthcheckPath = "/health"
# healthcheckTimeout = 100
```

### Phase 7: Debug Locally with Railway Env

Simulate Railway environment locally:

```bash
# Get Railway env vars
railway variables

# Run locally with Railway Postgres
export DATABASE_URL="postgresql://user:pass@localhost:5432/ghost"
export RAILWAY_ENVIRONMENT="local-test"
export PORT=8080

python3 wolf_app.py
```

Test if SQL errors reproduce locally.

### Success Criteria

All three endpoints working:

```bash
# Should all return {"ok": true, ...}
curl "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/enriched"
curl "https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user"
curl "https://ghost-protocol-production.up.railway.app/api/recent_alerts?limit=5"
```

---

## Quick Reference

**Check deployment status:**
```bash
bash scripts/monitor_deployment.sh
```

**Emergency disable watchlist:**
```bash
bash scripts/disable_watchlist_scheduler.sh
```

**Full diagnostics:**
```bash
bash scripts/emergency_fallback.sh
```

**Railway logs in terminal:**
```bash
railway logs
```

---

**Last Updated:** Dec 3, 2025 01:20 AM  
**Current Commit:** 63e88e5 (PostgreSQL INTERVAL fix)
