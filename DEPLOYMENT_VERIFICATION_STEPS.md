# Deployment Verification Steps

## Status: Code Ready for Deployment ✅

**Commit**: `2d772ed` - "chore(api): simplify request middleware to always return JSON 500 on error"
**Changes**: Simplified `_log_requests` middleware from 70+ lines to 14 lines

---

## Step 1: Push to Railway (MANUAL)

Since we're in a dev container without Railway remote configured, you need to push from your host machine:

```bash

# From your host machine (where Railway remote is configured)

cd /path/to/ghost-cockpit
git pull  # Get the latest commit (2d772ed)
git push railway main  # Or just 'git push' if Railway is default remote

```text

**Alternative**: Use Railway CLI

```bash

railway up

```text

---

## Step 2: Redeploy on Railway (MANUAL)

1. Open <<<<<https://railway.app/project/ghost-sniper-bot-seancole713>>>>>
2. Click on your web service
3. Go to **Deployments**tab
4. Click**Redeploy**on the latest commit (2d772ed)
5. Wait for**"Healthcheck succeeded"**message (2-3 minutes)


---

## Step 3: Verify OpenAPI Schema

Once deployed, check that routes are loaded:

```bash

curl -s <<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/openapi.json>>>>> | python -m json.tool | grep -E "(regime|tick|purge|movers)"

```text**Expected**: Should contain:

- `/api/regime/current`
- `/api/tick`
- `/api/cache/purge`
- `/api/scan/movers`
- `/api/scan/health`


---

## Step 4: Export Environment Variables

```bash

export GHOST_BASE_URL="<<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app">>>>>
export GHOST_API_TOKEN="edaa4eac-6455-4693-a745-142cb6deef03"

```text

---

## Step 5: Smoke Test Core Endpoints

### Test 1: Status Endpoint

```bash

curl -s "$GHOST_BASE_URL/api/status" | python -m json.tool

```text

**Expected**: `{"active": true, ...}` with HTTP 200

### Test 2: Tick Endpoint (Counter Increment)

```bash

echo "First call:"
curl -s "$GHOST_BASE_URL/api/tick" | python -m json.tool
echo -e "\nWaiting 3 seconds..."
sleep 3
echo "Second call:"
curl -s "$GHOST_BASE_URL/api/tick" | python -m json.tool

```text

**Expected**: Second call should have `tick` value greater than first call

### Test 3: Regime Endpoint

```bash

curl -s "$GHOST_BASE_URL/api/regime/current" | python -m json.tool

```text

**Expected**: `{"regime": "neutral", ...}` with HTTP 200 (not 404)

### Test 4: Diagnostics Endpoint (AAPL Routing Fix)

```bash

curl -s "$GHOST_BASE_URL/api/price/diagnostics?symbol=AAPL" | python -m json.tool

```text

**Expected**:

- HTTP 200 (not 404)
- `symbol: "AAPL"` in response
- `price` should be AAPL price (NOT WOLF price - this was the bug)
- No 10+ second timeout


### Test 5: Cache Purge Endpoint

```bash

curl -s -H "Authorization: Bearer $GHOST_API_TOKEN" \
     -H "Content-Type: application/json" \
     -X POST \
     -d '{"patterns":["price:AAPL","diagnostics:*"]}' \
     "$GHOST_BASE_URL/api/cache/purge" | python -m json.tool

```text

**Expected**: `{"ok": true, "purged_count": N}` with HTTP 200 (not 404)

---

## Step 6: Verify Server-Sent Events (SSE)

Open in browser:

```text

<<<<<https://ghost-sniper-bot-seancole713-production.up.railway.app/api/cockpit/stream>>>>>

```text

**Expected Events**(should appear continuously):

- `event: status` - Initial connection status
- `event: ping` - Periodic keepalive (every 30s)
- `event: snapshot` - Portfolio/position updates**Failure Signs**:

- Connection drops immediately
- No events appear
- 404 or 499 error


---

## Step 7: Monitor HTTP Logs (5 Minutes)

1. Open Railway Dashboard → Your Service → **Logs**tab
2. Filter to show HTTP requests
3. Watch for:
   - ❌**0 occurrences**of HTTP 499
   - ❌**0 occurrences**of HTTP 502
   - ✅ Any 500s should be JSON responses: `{"error": "internal_error"}`
   - ✅ All tested endpoints return 200**How to Check**:


```bash

# From Railway CLI (if available)

railway logs --follow | grep -E '(499|502|500)'

```text

**Success Criteria**:

- No 499 Client Closed Request errors
- No 502 Bad Gateway errors
- If 500 errors occur, they should be proper JSON responses (not empty)
- Average response time < 3000ms (ideally < 1000ms)


---

## Step 8: HTTP Client Timeout Enforcement (If Needed)

**Only do this if you still see 10+ second hangs in Step 5.**### Check Current Timeout Configuration

```bash

grep -n "AsyncClient\|httpx\.Client\|requests\.get\|requests\.post" wolf_app.py | head -20

```text

### Add Timeout to HTTP Clients

If HTTP clients don't have timeouts, add:

```python

# For httpx

SHARED_CLIENT = httpx.AsyncClient(
    timeout=float(os.getenv("REQUESTS_DEFAULT_TIMEOUT_S", "1.0"))
)

# For requests

TIMEOUT = float(os.getenv("REQUESTS_DEFAULT_TIMEOUT_S", "1.0"))
response = requests.get(url, timeout=TIMEOUT)

```text

Then redeploy.

---

## Expected Results Summary

| Test | Expected Result | Status |
|------|----------------|--------|
| OpenAPI schema | Contains all new routes | ⏳ Pending |
| /api/status | 200 OK | ⏳ Pending |
| /api/tick | Counter increments | ⏳ Pending |
| /api/regime/current | 200 OK (not 404) | ⏳ Pending |
| /api/price/diagnostics?symbol=AAPL | AAPL price (not WOLF) | ⏳ Pending |
| /api/cache/purge | {"ok": true} | ⏳ Pending |
| /api/cockpit/stream | SSE events flowing | ⏳ Pending |
| HTTP logs (5 min) | 0 × 499, 0 × 502 | ⏳ Pending |

---

## Troubleshooting

### If 404 errors persist

- Check OpenAPI schema - routes may not be loading
- Verify middleware isn't preventing route registration
- Check for duplicate route definitions


### If 499 errors persist

- Middleware may not be catching all exceptions
- Check for timeouts in background tasks
- Review upstream service timeouts


### If 10s+ timeouts persist

- Add HTTP client timeouts (Step 8)
- Check PRICE_PROVIDER_TIMEOUT_S env var
- Review provider health in diagnostics


### If SSE stream fails

- Check CORS headers
- Verify SSE endpoint isn't timing out
- Review browser console for connection errors


---

## Rollback Plan

If deployment causes issues:

```bash

# Railway Dashboard

Deployments → Select previous stable deployment → Redeploy

# Or via CLI

railway rollback

```text**Previous Stable**: Version before commit 2d772ed

---

## Next Steps After Verification

1. **If all tests pass**: Document success in GHOST_STATUS_REPORT.json
2. **If any tests fail**: Report which endpoint failed and exact error message
3. **If timeouts persist**: Implement Step 8 (HTTP client timeouts)
4. **If 499s persist**: Review exception handling in background tasks


---

**Status**: Ready for deployment. Commit 2d772ed is prepared locally.
**Action Required**: Push from host machine + redeploy on Railway
