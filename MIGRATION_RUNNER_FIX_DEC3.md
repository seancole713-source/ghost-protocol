# Migration Runner Fix - December 3, 2025

## 🔴 CRITICAL BUG FIXED: KeyError: 0 in Migration Runner

### **Root Cause**
The migration runner was crashing with `KeyError: 0` because it was trying to access PostgreSQL query results using **integer indices** (`result[0]`) instead of **column names** (`result['exists']`).

When `wolf_app.py` uses `psycopg2` with `RealDictCursor`, query results are **dictionaries**, not tuples. The code was attempting tuple-style access on a dictionary, causing the KeyError.

### **Error Sequence in Production Logs**
```
✅ PostgreSQL pool initialized (2-20 connections)
❌ Database error: 0                                      # KeyError: 0
❌ [MIGRATION] ❌ Migration runner failed: 0              # Re-raised as string "0"
❌ relation "ghost_watchlist_items" does not exist       # Tables never created
```

---

## 🔧 The Fix

### **File: `core/migration_runner.py`**

**Before (Lines 63-72):**
```python
cursor.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_name = 'ghost_watchlist_items'
    )
""")
result = cursor.fetchone()
table_exists = result[0] if result else False  # ❌ KeyError: 0
```

**After (Lines 63-72):**
```python
cursor.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND table_name = 'ghost_watchlist_items'
    ) as exists
""")
result = cursor.fetchone()
table_exists = result['exists'] if result else False  # ✅ Correct dictionary access
```

### **Changes Made**
1. ✅ Added `as exists` alias to SELECT EXISTS query (line 69)
2. ✅ Changed `result[0]` → `result['exists']` (line 72)
3. ✅ Applied same fix to `ensure_personal_watchlist_table()` function (line 147)
4. ✅ Improved exception handling to show error type for debugging (line 115)

---

## 🧪 Verification Tests

### **Test 1: Python Syntax**
```bash
python3 -m py_compile core/migration_runner.py
# ✅ PASS - No syntax errors
```

### **Test 2: RealDictCursor Behavior**
```python
# Simulate psycopg2 RealDictCursor
result = {'exists': False}  # Returns dict, not tuple

# OLD CODE WOULD CRASH:
# table_exists = result[0]  # KeyError: 0

# NEW CODE WORKS:
table_exists = result['exists']  # ✅ False
```

---

## 📋 Deployment Checklist

### **1. Deploy to Railway** 🚀
```bash
cd ~/ghost-protocol
git add core/migration_runner.py MIGRATION_RUNNER_FIX_DEC3.md
git commit -m "fix: Migration runner KeyError with RealDictCursor"
git push origin main
```

### **2. Monitor Railway Logs** 👀
Watch for these success indicators:
```
✅ PostgreSQL pool initialized (2-20 connections)
✅ [MIGRATION] ✅ 001_personal_watchlist.sql - applied successfully
✅ [GHOST STARTUP] ✅ Database migrations complete
```

**Expected:** No more `Database error: 0` or `KeyError: 0`

### **3. Verify Tables Created** ✅
```bash
# Connect to Railway Postgres via dashboard Query tab
SELECT tablename FROM pg_tables 
WHERE tablename LIKE '%watchlist%' 
ORDER BY tablename;
```

**Expected Result:**
```
ghost_watchlist_items
watchlist_prediction_tracking
watchlist_price_snapshots
watchlist_telegram_history
```

### **4. Test Personal Watchlist Endpoint** 🧪
```bash
curl https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user
```

**Expected Response:**
```json
{
  "items": [],
  "count": 0,
  "timestamp": 1733191234.567
}
```
- ✅ Should return 200 OK (not 404 or 500)
- ✅ Empty list is correct (no symbols added yet)

---

## 🏗️ Architecture Context

### **PostgreSQL Cursor Types**

| Cursor Type | Result Format | Access Method | Used By |
|-------------|---------------|---------------|---------|
| **Standard** | `tuple` | `result[0]` | Most DB libraries |
| **RealDictCursor** | `dict` | `result['column_name']` | Ghost Protocol |

### **Why RealDictCursor?**
Ghost Protocol uses `RealDictCursor` (configured in `core/db_engine.py` line 60) because:
1. **Explicit column names** → Safer, more readable code
2. **JSON-friendly** → Easy to serialize to REST API responses
3. **Schema changes** → Less brittle when columns are reordered

### **The Bug's Impact**
- ❌ Migrations **never ran** in production
- ❌ Personal watchlist tables **never created**
- ❌ Endpoint returned **404 Not Found**
- ❌ Healthcheck **failed** (app marked unhealthy)

---

## 🔍 Technical Deep Dive

### **Why "KeyError: 0" and Not "KeyError: 'column_name'"?**
When you access `dict[0]`, Python looks for the **key** `0` (integer), not index 0. Since RealDictCursor returns `{'exists': True}`, key `0` doesn't exist.

### **Why Did This Pass Local Testing?**
If local testing used:
- SQLite (no migrations needed)
- Standard cursor without RealDictCursor
- Mock database with tuple-style results

Then the bug wouldn't manifest until production PostgreSQL with RealDictCursor.

### **Alternative Fix Approaches Considered**
1. ❌ **Switch to standard cursor** → Would break other code expecting dicts
2. ❌ **Convert result to tuple** → Extra overhead on every query
3. ✅ **Use column names** → Correct approach for RealDictCursor

---

## 📊 Before/After Comparison

### **Before Fix (Production Logs Dec 2)**
```
✅ PostgreSQL pool initialized
❌ Database error: 0
❌ [MIGRATION] ❌ Migration runner failed: 0
⚠️  [GHOST STARTUP] ⚠️  Some migrations failed
❌ relation "ghost_watchlist_items" does not exist
❌ Healthcheck failed!
```

### **After Fix (Expected Dec 3)**
```
✅ PostgreSQL pool initialized
✅ [MIGRATION] ✅ 001_personal_watchlist.sql - applied successfully
✅ [GHOST STARTUP] ✅ Database migrations complete
✅ Personal Watchlist endpoints registered
✅ Cockpit V3 LIVE endpoints registered
✅ Healthcheck passed
```

---

## 🎯 Success Criteria

| Check | Status | Verification |
|-------|--------|--------------|
| No KeyError in logs | ⏳ | Deploy and monitor logs |
| Migrations run successfully | ⏳ | Check for "applied successfully" |
| Tables exist in DB | ⏳ | Run `SELECT tablename FROM pg_tables` |
| `/api/v3/watchlist/user` returns 200 | ⏳ | `curl` test |
| Healthcheck passes | ⏳ | Railway deployment succeeds |

---

## 🔗 Related Files

- `core/migration_runner.py` - **FIXED** (RealDictCursor access)
- `core/db_engine.py` - Configures RealDictCursor (line 60)
- `migrations/001_personal_watchlist.sql` - SQL to create tables
- `api/personal_watchlist_endpoints.py` - Graceful error handling (already fixed)
- `wolf_app.py` - Calls `run_migrations()` on startup (line 3470)

---

## 💡 Lessons Learned

1. **Always check cursor type** when accessing query results
2. **Test with production database config** (not just SQLite)
3. **Use explicit column aliases** in SELECT statements
4. **Log exception types** to distinguish KeyError from other errors
5. **Add integration tests** for database migrations

---

## 📞 Support

If migrations still fail after this fix:
1. Check Railway logs for new error messages
2. Verify `DATABASE_URL` is set correctly
3. Manually run migration SQL via Railway dashboard
4. Check if tables already exist: `\dt ghost_watchlist_*`

---

**Fix Applied:** December 3, 2025  
**Deployed By:** GitHub Copilot + Operator  
**Status:** ✅ Ready for Production Deployment
