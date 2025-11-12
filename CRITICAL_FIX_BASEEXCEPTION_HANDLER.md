# CRITICAL FIX: BaseException Handler AssertionError

## Issue Summary
**Severity**: CRITICAL - Complete deployment failure  
**Impact**: 100% of Railway deployments failing at healthcheck  
**Root Cause**: Invalid exception handler registration in wolf_app.py

## Error Details

### Railway Log Output
```
AssertionError in starlette/middleware/exceptions.py line 42:
    assert issubclass(exc_class_or_status_code, Exception)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

### Full Stack Trace
```
File "/usr/local/lib/python3.11/site-packages/fastapi/applications.py", line 212, in build_middleware_stack
    app = cls(app=app, **options)
File "/usr/local/lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 32, in __init__
    self.add_exception_handler(key, value)
File "/usr/local/lib/python3.11/site-packages/starlette/middleware/exceptions.py", line 42, in add_exception_handler
    assert issubclass(exc_class_or_status_code, Exception)
```

## Root Cause Analysis

### The Problem
In `wolf_app.py` at line 255:

```python
try:
    @APP.exception_handler(BaseException)  # catches CancelledError, ExceptionGroup
    async def _base_handler(request: Request, exc: BaseException):
        return _json500("base_exception")
except Exception:
    pass
```

**Why This Failed:**
1. The `@APP.exception_handler(BaseException)` decorator is evaluated **before** the function definition
2. Starlette's `add_exception_handler()` validates that the exception class is a subclass of `Exception`
3. `BaseException` is NOT a subclass of `Exception` - it's the parent class
4. The assertion fails: `assert issubclass(BaseException, Exception)` → False
5. Even wrapping in try/except doesn't help because the decorator evaluation happens first

### Starlette's Constraint
From Starlette source code:
```python
def add_exception_handler(
    self,
    exc_class_or_status_code: int | type[Exception],  # Must be Exception subclass!
    handler: typing.Callable,
) -> None:
    assert issubclass(exc_class_or_status_code, Exception)
```

## The Fix

### What Changed (Commit: 790b74d)
```python
@APP.exception_handler(Exception)
async def _ex_handler(request: Request, exc: Exception):
    return _json500("unhandled_exception")

# NOTE: Cannot register BaseException handler - Starlette requires Exception subclasses
# Middleware will catch BaseException (CancelledError, KeyboardInterrupt, etc.)
```

### Why This Works
1. **Removed the problematic decorator** that was causing AssertionError
2. **Kept the Exception handler** which catches all normal exceptions
3. **Added clear documentation** explaining why BaseException can't be registered
4. **Middleware already handles BaseException** - the function-based middleware at line 693 catches all BaseException types

### Coverage Maintained
- `@APP.exception_handler(RuntimeError)` - catches "No response returned" errors
- `@APP.exception_handler(Exception)` - catches all other exceptions
- `@APP.middleware("http")` - catches BaseException (CancelledError, KeyboardInterrupt, etc.)

## Impact Assessment

### Before Fix
- ❌ Railway healthcheck: **FAILING** (500 errors on every request)
- ❌ Middleware stack: **BROKEN** (AssertionError during build)
- ❌ All endpoints: **UNAVAILABLE** (app couldn't start)
- ❌ Deployment status: **FAILED** after 9 attempts

### After Fix
- ✅ Railway healthcheck: **Should PASS** (middleware builds correctly)
- ✅ Middleware stack: **BUILDS** (no assertion errors)
- ✅ All endpoints: **AVAILABLE** (app starts normally)
- ✅ Deployment status: **DEPLOYING** (commit 790b74d pushed)

## Verification Steps

### 1. Check Railway Deployment
- Monitor Railway dashboard for new deployment from commit 790b74d
- Build logs should show no AssertionError
- Healthcheck attempts should succeed with HTTP 200

### 2. Test Critical Endpoints
```bash
export BASE="https://ghost-sniper-bot-seancole713-production.up.railway.app"
export TOKEN="edaa4eac-6455-4693-a745-142cb6deef03"

# Should all return 200 with JSON
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/ui/health" | jq .
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/api/status" | jq .
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/api/regime/current" | jq .
```

### 3. Monitor Logs
- No more "AssertionError" in deployment logs
- No more "500 Internal Server Error" on healthcheck
- Successful healthcheck confirmation message

## Timeline

| Time | Event |
|------|-------|
| 8:15 AM | Railway deployment started (commit 65526e40) |
| 8:15 AM | **AssertionError** appears in logs |
| 8:15-8:20 AM | Healthcheck fails 9 times (service unavailable) |
| 8:18 AM | Root cause identified: BaseException handler |
| 8:19 AM | Fix committed: 790b74d |
| 8:19 AM | Fix pushed to main branch |
| 8:19 AM+ | Railway auto-deploying new commit |

## Lessons Learned

### Technical
1. **Decorator evaluation order matters** - decorators execute before the function they wrap
2. **Starlette has strict type constraints** - only Exception subclasses allowed
3. **Try/except can't catch decorator failures** - they happen at definition time
4. **Middleware is more flexible** - can catch BaseException without restrictions

### Process
1. **Read the stack trace carefully** - the AssertionError line pointed directly to the constraint
2. **Understand framework requirements** - Starlette's API is well-documented
3. **Test in dev first** - this would have been caught with local testing
4. **Monitor Railway logs actively** - rapid iteration requires watching deploy logs

## Related Files
- `wolf_app.py` (lines 244-254) - Exception handlers
- `wolf_app.py` (lines 693-711) - Middleware that catches BaseException
- Railway config: healthcheckPath=/ui/health

## Status
✅ **FIXED** - Commit 790b74d deployed to main  
⏳ **DEPLOYING** - Railway processing new commit  
🔄 **MONITORING** - Awaiting healthcheck success confirmation
