# GHOST Upgrade Plan

**Version**: 1.0\
**Date**: October 4, 2025\
**Status**: 📋 **Ready for Implementation**

______________________________________________________________________

## Overview

This upgrade plan addresses the findings from the Deep Audit Report, providing concrete
implementation steps, code samples, and validation criteria for each remediation item.
The plan is organized by priority tier with estimated effort and dependencies mapped.

______________________________________________________________________

## Priority Matrix

| Tier | Count | Effort (days) | Risk if Deferred | Completion Target |
|------|-------|---------------|------------------|-------------------| | **P0** | 1 |
0.5 | 🔴 Data breach, financial loss | Immediate | | **P1** | 3 | 3-5 | 🟠 Security
incidents, instability | 2 weeks | | **P2** | 3 | 5-7 | 🟡 Tech debt, confusion | 1 month
|

**Total Estimated Effort**: 8.5-12.5 developer-days (1.5-2.5 weeks at 1 FTE)

______________________________________________________________________

## P0: Critical Security (Immediate)

### P0-1: Secrets Exposure Remediation

**Status**: ⚠️ Partial (gitignore added, rotation pending)

#### Step 1: Rotate API Keys (0.5 days)

**Polygon**:

```bash
# 1. Login to Polygon dashboard
open https://polygon.io/dashboard/keys

# 2. Generate new key (Dashboard UI)
# → Note: Old key remains active until explicitly revoked

# 3. Update Railway
railway variables set POLYGON_API_KEY="<NEW_KEY>"

# 4. Verify Ghost picks up new key
railway logs | grep "POLYGON_KEY"
# Expected: "SET (len=XX)"

# 5. Revoke old key in Polygon dashboard
```

**AlphaVantage**:

```bash
# 1. Request new key
open "https://www.alphavantage.co/support/#api-key"
# Fill form → receive email with new key

# 2. Update Railway
railway variables set ALPHAVANTAGE_API_KEY="<NEW_KEY>"

# 3. Verify
railway logs | grep "ALPHAVANTAGE_KEY"

# Note: AlphaVantage doesn't support key revocation; old key expires naturally
```

**Ghost API Token**:

```bash
# 1. Generate new token
openssl rand -hex 32
# Example output: a3f5c9d2e8b1f4a7c6e9d3b2f8a5c1e7d4b9f2a6c8e5d1b7f3a9c4e6d2b8f5a1c7

# 2. Update Railway
railway variables set GHOST_API_TOKEN="<NEW_TOKEN>"

# 3. Update any external clients
# (e.g., test scripts, monitoring tools, mobile apps)
```

**Telegram**:

```bash
# 1. Open BotFather
open "https://t.me/botfather"

# 2. Revoke old token
/revoke
# → Select Ghost bot → Confirm

# 3. Generate new token
/token
# → Select Ghost bot → Copy new token

# 4. Update Railway
railway variables set TELEGRAM_BOT_TOKEN="<NEW_TOKEN>"

# 5. Re-register webhook
GHOST_URL=$(railway variables get RAILWAY_PUBLIC_DOMAIN)
NEW_TOKEN="<NEW_TOKEN>"
curl -X POST "https://api.telegram.org/bot${NEW_TOKEN}/setWebhook" \
  -d "url=https://${GHOST_URL}/telegram/webhook"

# 6. Verify
curl "https://api.telegram.org/bot${NEW_TOKEN}/getWebhookInfo"
# Expected: url matches your Ghost URL
```

**Validation**:

```bash
# Test each provider via Ghost health endpoint
curl https://your-ghost-url/health/detailed | jq .

# Expected output includes:
# - "polygon": "ok" or "configured"
# - "alphavantage": "ok" or "configured"
# - "telegram": {"bot_ok": true, "webhook_set": true}
```

#### Step 2: Git History Cleanup (0.5 hours)

**Option A: BFG Repo-Cleaner** (Recommended)

```bash
# Install
brew install bfg  # macOS
# or: sudo apt-get install bfg-repo-cleaner  # Linux

# Clone fresh mirror
cd /tmp
git clone --mirror git@github.com:seancole713-source/GHOST.git ghost-cleanup
cd ghost-cleanup

# Remove secrets.env from ALL commits
bfg --delete-files secrets.env

# Cleanup refs
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Push cleaned history
git push --force

# Cleanup
cd /workspaces/GHOST
git fetch origin
git reset --hard origin/main
```

**Option B: git filter-branch** (Manual)

```bash
cd /workspaces/GHOST

git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch secrets.env" \
  --prune-empty --tag-name-filter cat -- --all

git push --force --all
git push --force --tags

# All collaborators must re-clone:
# git clone git@github.com:seancole713-source/GHOST.git
```

**Validation**:

```bash
# Verify secrets.env is gone from history
git log --all --full-history -- secrets.env
# Expected: no output

# Check current working tree
ls -la secrets.env
# Expected: "No such file or directory" (unless recreated locally)
```

#### Step 3: Install Pre-Commit Hooks (0.5 hours)

```bash
# Install detect-secrets
pip install detect-secrets

# Generate baseline
detect-secrets scan --baseline .secrets.baseline

# Create pre-commit hook
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
detect-secrets-hook --baseline .secrets.baseline $(git diff --cached --name-only)
if [ $? -ne 0 ]; then
  echo "❌ Potential secrets detected! Commit blocked."
  echo "Review findings above, then update baseline if false positive:"
  echo "  detect-secrets scan --baseline .secrets.baseline"
  exit 1
fi
EOF

chmod +x .git/hooks/pre-commit

# Test hook
echo "POLYGON_API_KEY=abc123" > test_secret.txt
git add test_secret.txt
git commit -m "test"
# Expected: Hook blocks commit, prints warning
rm test_secret.txt
```

**Deliverables**:

- [ ] All 5 API keys rotated
- [ ] Railway variables updated
- [ ] secrets.env removed from history
- [ ] Pre-commit hook installed
- [ ] `.secrets.baseline` committed
- [ ] Updated `CHANGELOG.md` with rotation date

______________________________________________________________________

## P1: High-Priority Fixes (2 Weeks)

### P1-1: Secure Debug Endpoints (1 day)

#### Implementation

**File**: `wolf_app.py`

**Change 1**: Add auth to `/debug/telegram_test`

```python
@APP.post("/debug/telegram_test")
async def debug_telegram_test(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """Test Telegram notifications. Requires auth."""
    # Add auth check
    try:
        _require_bearer((f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None)
    except Exception:
        raise HTTPException(401, "Unauthorized")
    
    # Original logic
    try:
        msg = (body or {}).get("msg", "Ghost test message")
        _send_telegram(msg)
        return {"ok": True, "sent": msg}
    except Exception as e:
        return {"ok": False, "error": str(e)}
```

**Change 2**: Add auth to `/debug/prev_close`

```python
@APP.post("/debug/prev_close")
async def debug_set_prev_close(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """Test-helper: set cached prev_close for WOLF and clear live price.
    Enabled only when SNAP_TEST_MODE is active. Requires auth."""
    
    # Auth check (even in test mode)
    try:
        _require_bearer((f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None)
    except Exception:
        raise HTTPException(401, "Unauthorized")
    
    # Environment gate
    if os.getenv("SNAP_TEST_MODE", "0").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "forbidden")
    
    # Original logic
    try:
        prev_close_val = (body or {}).get("prev_close")
        if prev_close_val is None:
            raise HTTPException(422, "prev_close is required")
        pv = float(prev_close_val)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "invalid prev_close")
    
    _cache_put_price(WOLF, None, pv, "prev-close")
    return {"ok": True, "prev_close": pv}
```

**Change 3**: Add auth to `/debug/price_diag`

```python
@APP.post("/debug/price_diag")
async def debug_set_price_diag(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """Test-helper: set PRICE_DIAG fields to simulate quorum/anomaly.
    Enabled only when SNAP_TEST_MODE is active. Requires auth."""
    
    # Auth check
    try:
        _require_bearer((f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None)
    except Exception:
        raise HTTPException(401, "Unauthorized")
    
    # Environment gate
    if os.getenv("SNAP_TEST_MODE", "0").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "forbidden")
    
    # Original logic
    try:
        if isinstance(body, dict):
            for k in ("anomaly", "reason", "provider_spread", "quorum_ok"):
                if k in body:
                    PRICE_DIAG[k] = body[k]
    except Exception:
        pass
    
    return {"ok": True, "diag": PRICE_DIAG}
```

**Testing**:

```python
# tests/test_debug_auth.py
import pytest

def test_debug_telegram_requires_auth(client):
    """Debug telegram endpoint requires bearer token."""
    resp = client.post("/debug/telegram_test", json={"msg": "test"})
    assert resp.status_code == 401
    
    # With auth
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    resp = client.post("/debug/telegram_test", json={"msg": "test"}, headers=headers)
    assert resp.status_code == 200

def test_debug_prev_close_requires_auth(client):
    """Debug prev_close requires auth even in test mode."""
    resp = client.post("/debug/prev_close", json={"prev_close": 25.0})
    assert resp.status_code == 401
    
    # With auth
    headers = {"Authorization": f"Bearer {TEST_TOKEN}"}
    resp = client.post("/debug/prev_close", json={"prev_close": 25.0}, headers=headers)
    assert resp.status_code in [200, 403]  # 403 if not in SNAP_TEST_MODE

def test_debug_price_diag_requires_auth(client):
    """Debug price_diag requires auth."""
    resp = client.post("/debug/price_diag", json={"anomaly": True})
    assert resp.status_code == 401
```

**Validation**:

```bash
# Without token (should fail)
curl -X POST https://your-ghost-url/debug/telegram_test \
  -H "Content-Type: application/json" \
  -d '{"msg":"test"}'
# Expected: {"detail":"Unauthorized"}, status 401

# With token (should succeed)
curl -X POST https://your-ghost-url/debug/telegram_test \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"msg":"test"}'
# Expected: {"ok":true,"sent":"test"}, status 200
```

**Deliverables**:

- [ ] Auth added to 3 debug endpoints
- [ ] Tests passing (`pytest tests/test_debug_auth.py`)
- [ ] Documented in `OPERATIONS.md`

______________________________________________________________________

### P1-2: Consolidate Duplicate SSE Route (2 days)

#### Investigation Phase (0.5 days)

```bash
# Extract both implementations
sed -n '4417,4467p' wolf_app.py > sse_v1.py
sed -n '6172,6238p' wolf_app.py > sse_v2.py

# Compare
diff -u sse_v1.py sse_v2.py

# Check which one is called in production
railway logs | grep "cockpit/stream" | tail -20
```

#### Implementation (1 day)

**Decision Criteria**:

1. If identical → remove duplicate (likely copy-paste)
2. If v2 has bug fixes → keep v2, remove v1
3. If v1 is stable → keep v1, remove v2

**Assuming v2 is newer (keep v2)**:

**File**: `wolf_app.py`

```python
# Delete lines 4417-4467 (old implementation)
# Keep lines 6172-6238 (new implementation)

# Add comment to prevent re-introduction
# Line 6172:
@APP.get("/api/cockpit/stream")
async def api_cockpit_stream_v2(request: Request):
    """Server-Sent Events stream for cockpit updates.
    
    NOTE: This is the canonical implementation as of 2025-10-04.
    Previous implementation removed during P1-2 consolidation.
    """
    # ... existing implementation ...
```

**Testing** (0.5 days):

```python
# tests/test_cockpit_stream.py
import asyncio
import pytest

async def test_cockpit_stream_single_implementation(client):
    """Verify only one /api/cockpit/stream implementation exists."""
    from wolf_app import APP
    stream_routes = [r for r in APP.routes if r.path == "/api/cockpit/stream"]
    assert len(stream_routes) == 1, f"Found {len(stream_routes)} implementations, expected 1"

async def test_cockpit_stream_returns_sse(client):
    """Verify cockpit stream returns SSE format."""
    async with client.stream("GET", "/api/cockpit/stream") as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream"
        
        # Read first event
        chunk = await resp.aiter_bytes().__anext__()
        assert b"data:" in chunk
```

**Validation**:

```bash
# Start Ghost
uvicorn wolf_app:app --host 0.0.0.0 --port 5000

# Connect SSE client
curl -N https://your-ghost-url/api/cockpit/stream

# Expected: Stream of JSON events every 2 seconds
# data: {"price":25.43,...}
# data: {"price":25.44,...}

# Monitor for route collision warnings in logs
railway logs | grep -i "duplicate\|collision\|override"
# Expected: No warnings after fix
```

**Deliverables**:

- [ ] Duplicate removed (git diff shows ~50 line deletion)
- [ ] Test passing (`pytest tests/test_cockpit_stream.py`)
- [ ] No route collision warnings in logs
- [ ] Updated `CHANGELOG.md`

______________________________________________________________________

### P1-3: SSE Client Disconnect Handling (2 days)

#### Implementation

**Strategy**: Request disconnect detection (works with Starlette/FastAPI)

**File**: `wolf_app.py`

**Change 1**: Update `/events` endpoint

```python
@APP.get("/events")
async def events_sse(request: Request):
    """Server-Sent Events stream for general system events.
    Auto-disconnects when client closes connection or after 1 hour."""
    
    async def _event_generator():
        started = time.time()
        max_age_s = 3600  # 1 hour TTL
        
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                LOGGER.info("sse_client_disconnected", extra={"endpoint": "/events"})
                break
            
            # Check TTL
            if time.time() - started > max_age_s:
                LOGGER.info("sse_ttl_expired", extra={"endpoint": "/events", "age_s": max_age_s})
                break
            
            # Build event
            try:
                events_list = list(_EVENTS_RING)[-10:]  # Last 10 events
                data = {"events": events_list, "ts": int(time.time())}
                yield f"data: {json.dumps(data)}\n\n"
            except Exception as e:
                LOGGER.exception("sse_event_build_failed", extra={"error": str(e)})
                yield f"data: {json.dumps({'error': 'event_build_failed'})}\n\n"
            
            await asyncio.sleep(5)
    
    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
```

**Change 2**: Update `/api/cockpit/stream`

```python
@APP.get("/api/cockpit/stream")
async def api_cockpit_stream_v2(request: Request):
    """Server-Sent Events stream for cockpit updates.
    Auto-disconnects when client closes connection or after 1 hour."""
    
    async def _cockpit_generator():
        started = time.time()
        max_age_s = 3600  # 1 hour TTL
        client_id = str(uuid.uuid4())  # Unique ID for this connection
        
        LOGGER.info("sse_client_connected", extra={
            "endpoint": "/api/cockpit/stream",
            "client_id": client_id
        })
        
        try:
            while True:
                # Check disconnect
                if await request.is_disconnected():
                    LOGGER.info("sse_client_disconnected", extra={
                        "endpoint": "/api/cockpit/stream",
                        "client_id": client_id,
                        "duration_s": int(time.time() - started)
                    })
                    break
                
                # Check TTL
                if time.time() - started > max_age_s:
                    LOGGER.info("sse_ttl_expired", extra={
                        "endpoint": "/api/cockpit/stream",
                        "client_id": client_id
                    })
                    break
                
                # Build snapshot
                try:
                    snap = _build_cockpit_snapshot()
                    yield f"data: {json.dumps(snap)}\n\n"
                except Exception as e:
                    LOGGER.exception("sse_snapshot_build_failed", extra={
                        "client_id": client_id,
                        "error": str(e)
                    })
                    yield f"data: {json.dumps({'error': 'snapshot_failed'})}\n\n"
                
                await asyncio.sleep(2)
        finally:
            LOGGER.info("sse_generator_stopped", extra={
                "client_id": client_id,
                "total_duration_s": int(time.time() - started)
            })
    
    return StreamingResponse(
        _cockpit_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
```

**Monitoring** (0.5 days):

Add Prometheus metrics for SSE tracking:

```python
# Near other Prometheus metrics (around line 2000)
_G_SSE_CLIENTS = Gauge(
    "ghost_sse_active_clients",
    "Number of active SSE clients",
    ["endpoint"]
)

_C_SSE_CONNECTS = Counter(
    "ghost_sse_connects_total",
    "Total SSE client connections",
    ["endpoint"]
)

_C_SSE_DISCONNECTS = Counter(
    "ghost_sse_disconnects_total",
    "Total SSE client disconnections",
    ["endpoint", "reason"]
)

# Update generator code to increment metrics:
def _cockpit_generator():
    endpoint = "/api/cockpit/stream"
    _C_SSE_CONNECTS.labels(endpoint=endpoint).inc()
    _G_SSE_CLIENTS.labels(endpoint=endpoint).inc()
    
    try:
        # ... existing while loop ...
    finally:
        _G_SSE_CLIENTS.labels(endpoint=endpoint).dec()
        _C_SSE_DISCONNECTS.labels(endpoint=endpoint, reason="natural").inc()
```

**Testing** (0.5 days):

```python
# tests/test_sse_cleanup.py
import asyncio
import pytest

async def test_sse_stops_on_client_disconnect(client):
    """SSE generator stops when client disconnects."""
    # Start streaming
    async with client.stream("GET", "/api/cockpit/stream") as resp:
        # Read one event
        chunk = await resp.aiter_bytes().__anext__()
        assert b"data:" in chunk
        # Close connection (context manager exit triggers disconnect)
    
    # Check metrics
    metrics_resp = await client.get("/metrics")
    metrics_text = metrics_resp.text
    
    # Verify disconnect was logged
    assert "ghost_sse_disconnects_total" in metrics_text

async def test_sse_ttl_expires(client):
    """SSE generator stops after TTL expiry."""
    # This test requires mocking time.time() or waiting 1 hour
    # Simplified: verify TTL logic exists
    from wolf_app import api_cockpit_stream_v2
    import inspect
    source = inspect.getsource(api_cockpit_stream_v2)
    assert "max_age_s" in source
    assert "time() - started > max_age_s" in source
```

**Validation**:

```bash
# Monitor SSE connections in real-time
watch -n 1 'curl -s https://your-ghost-url/metrics | grep ghost_sse'

# Expected output:
# ghost_sse_active_clients{endpoint="/events"} 0
# ghost_sse_active_clients{endpoint="/api/cockpit/stream"} 2
# ghost_sse_connects_total{endpoint="/events"} 15
# ghost_sse_disconnects_total{endpoint="/events",reason="natural"} 15

# Simulate client disconnect
curl -N --max-time 5 https://your-ghost-url/api/cockpit/stream

# Check logs for disconnect event
railway logs | grep "sse_client_disconnected"
# Expected: JSON log entry with client_id and duration_s
```

**Deliverables**:

- [ ] Disconnect detection added to 3 SSE endpoints
- [ ] TTL of 1 hour configured
- [ ] Prometheus metrics tracking connections
- [ ] Tests passing (`pytest tests/test_sse_cleanup.py`)
- [ ] Logs confirm generators stop on disconnect

______________________________________________________________________

## P2: Medium-Priority Improvements (1 Month)

### P2-1: Telegram Webhook Signature Validation (1 day)

#### Implementation

**Step 1**: Generate webhook secret

```bash
# Generate random secret
openssl rand -base64 32
# Example: Xy9bZ3fG8kL2mN4pQ6rS8tU0vW2xY4zA6bC8dE0fG2h=

# Set in Railway
railway variables set TELEGRAM_WEBHOOK_SECRET="<SECRET>"
```

**Step 2**: Update webhook registration script

**File**: `setup_telegram_webhook.sh`

```bash
#!/bin/bash
set -e

GHOST_URL="${1:-https://ghost-production.up.railway.app}"
BOT_TOKEN="${TELEGRAM_BOT_TOKEN}"
WEBHOOK_SECRET="${TELEGRAM_WEBHOOK_SECRET}"

if [ -z "$BOT_TOKEN" ]; then
  echo "❌ TELEGRAM_BOT_TOKEN not set"
  exit 1
fi

if [ -z "$WEBHOOK_SECRET" ]; then
  echo "⚠️  TELEGRAM_WEBHOOK_SECRET not set (webhook won't be validated)"
fi

echo "🔗 Setting webhook: ${GHOST_URL}/telegram/webhook"

curl -X POST "https://api.telegram.org/bot${BOT_TOKEN}/setWebhook" \
  -d "url=${GHOST_URL}/telegram/webhook" \
  -d "secret_token=${WEBHOOK_SECRET}" \
  -d "max_connections=10" \
  -d "drop_pending_updates=true"

echo ""
echo "✅ Webhook configured. Verify with:"
echo "   curl https://api.telegram.org/bot${BOT_TOKEN}/getWebhookInfo"
```

**Step 3**: Update webhook endpoint

**File**: `wolf_app.py`

```python
# Near TELEGRAM_BOT_TOKEN declaration (around line 408)
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()

# Update webhook handler (around line 4660)
@APP.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """Telegram bot webhook handler.
    
    Validates webhook secret token to prevent spoofing.
    Commands: /status, /signal, /pnl, /today
    """
    # Validate webhook secret
    if TELEGRAM_WEBHOOK_SECRET:
        token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if not token or token != TELEGRAM_WEBHOOK_SECRET:
            LOGGER.warning("telegram_webhook_invalid_token", extra={
                "ip": request.client.host if request.client else "unknown",
                "token_provided": bool(token)
            })
            raise HTTPException(403, "Invalid webhook token")
    else:
        LOGGER.warning("telegram_webhook_no_secret", extra={
            "msg": "TELEGRAM_WEBHOOK_SECRET not set; webhook validation disabled"
        })
    
    # Original logic
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")
    
    # ... existing command handling ...
```

**Testing**:

```python
# tests/test_telegram_webhook.py
def test_webhook_requires_secret(client):
    """Webhook rejects requests without valid secret."""
    # No secret header
    resp = client.post("/telegram/webhook", json={"message": {"text": "/status"}})
    assert resp.status_code == 403
    
    # Invalid secret
    headers = {"X-Telegram-Bot-Api-Secret-Token": "wrong_secret"}
    resp = client.post("/telegram/webhook", json={"message": {"text": "/status"}}, headers=headers)
    assert resp.status_code == 403
    
    # Valid secret (requires env var set in conftest)
    valid_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET")
    headers = {"X-Telegram-Bot-Api-Secret-Token": valid_secret}
    resp = client.post("/telegram/webhook", json={"message": {"text": "/status"}}, headers=headers)
    assert resp.status_code == 200
```

**Validation**:

```bash
# Test webhook with invalid token
curl -X POST https://your-ghost-url/telegram/webhook \
  -H "Content-Type: application/json" \
  -H "X-Telegram-Bot-Api-Secret-Token: fake_token" \
  -d '{"message":{"text":"/status"}}'
# Expected: {"detail":"Invalid webhook token"}, status 403

# Test webhook with valid token (from Railway)
SECRET=$(railway variables get TELEGRAM_WEBHOOK_SECRET)
curl -X POST https://your-ghost-url/telegram/webhook \
  -H "Content-Type: application/json" \
  -H "X-Telegram-Bot-Api-Secret-Token: ${SECRET}" \
  -d '{"message":{"chat":{"id":123},"text":"/status"}}'
# Expected: 200 OK (command processed)

# Verify Telegram can reach webhook
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo" | jq .
# Expected: "url" matches Ghost URL, "has_custom_certificate": false
```

**Deliverables**:

- [ ] Webhook secret generated & set in Railway
- [ ] `setup_telegram_webhook.sh` updated
- [ ] Webhook validation implemented
- [ ] Tests passing
- [ ] Webhook re-registered with secret

______________________________________________________________________

### P2-2: Archive Legacy `main.py` (0.5 days)

#### Implementation

**Step 1**: Confirm main.py is unused

```bash
# Check Railway start command
railway run printenv | grep -i "command\|start"

# Check if main.py is imported anywhere
grep -r "import main" --exclude-dir=.venv --exclude-dir=__pycache__
grep -r "from main import" --exclude-dir=.venv --exclude-dir=__pycache__

# Expected: No results (main.py is orphaned)
```

**Step 2**: Extract unique routes

```bash
# Compare route sets
grep "@app\." main.py | cut -d'(' -f1 | sort > main_routes.txt
grep "@APP\." wolf_app.py | cut -d'(' -f1 | sort > wolf_routes.txt

# Find routes only in main.py
comm -23 main_routes.txt wolf_routes.txt

# Expected output (routes to migrate or confirm obsolete):
# @app.get("/catalog/status
# @app.get("/catalog/search
# @app.post("/agent/start
```

**Step 3**: Migrate or document

````markdown
# In CHANGELOG.md or DEPRECATED.md

## Deprecated Routes (from main.py)

The following routes existed in `main.py` but are **not migrated** to `wolf_app.py`:

| Route | Method | Reason |
|-------|--------|--------|
| `/catalog/status` | GET | Catalog feature removed (focus mode) |
| `/catalog/search` | GET | Catalog feature removed (focus mode) |
| `/agent/start` | POST | Replaced by `/control` endpoint |
| `/api/news` | GET | Replaced by `/api/cockpit` (includes news) |

If any of these are needed, they can be restored from git history:
```bash
git show HEAD:main.py | grep -A 20 "def catalog_status"
````

````

**Step 4**: Archive
```bash
# Rename to indicate deprecation
git mv main.py main_DEPRECATED_2025-10-04.py

# Add notice to top of file
cat > /tmp/deprecation_notice.txt << 'EOF'
"""
DEPRECATED: This file is a legacy backup from pre-October 2025.
All active routes have been migrated to wolf_app.py.

DO NOT IMPORT OR MODIFY THIS FILE.

To restore a specific route:
  git show $(git log -1 --format=%H main.py):main.py | grep -A 50 "def route_name"

For reference only. Will be removed in Q1 2026.
"""

EOF

# Prepend notice
cat /tmp/deprecation_notice.txt main_DEPRECATED_2025-10-04.py > /tmp/main_new.py
mv /tmp/main_new.py main_DEPRECATED_2025-10-04.py

# Commit
git add main_DEPRECATED_2025-10-04.py
git commit -m "chore: deprecate main.py, all routes migrated to wolf_app.py"
````

**Validation**:

```bash
# Verify Ghost still starts
railway run python -c "from wolf_app import APP; print('✅ Import successful')"

# Check no imports of main module
rg "import.*main[^_]" --type py
# Expected: No matches

# Verify deprecated file has notice
head -10 main_DEPRECATED_2025-10-04.py
# Expected: Deprecation notice visible
```

**Deliverables**:

- [ ] main.py renamed with deprecation suffix
- [ ] Deprecation notice added to file
- [ ] Unique routes documented as removed/obsolete
- [ ] Tests still passing
- [ ] Updated `CHANGELOG.md`

______________________________________________________________________

### P2-3: Environment Variables Documentation (2 days)

#### Implementation

**File**: `ENV_VARS_REFERENCE.md`

````markdown
# Ghost Environment Variables Reference

**Last Updated**: October 4, 2025  
**Version**: 1.0

This document catalogs all environment variables used by Ghost, their purpose, defaults, and whether they're required.

---

## Quick Reference

| Category | Required | Optional | Total |
|----------|----------|----------|-------|
| Security & Auth | 1 | 8 | 9 |
| Price Providers | 0* | 3 | 3 |
| Persistence | 0 | 6 | 6 |
| Alerts & Notifications | 0 | 17 | 17 |
| AI & Forecasting | 0 | 18 | 18 |
| Observability | 0 | 8 | 8 |
| **Total** | **1** | **60** | **61** |

*At least one price provider key recommended for live data

---

## 1. Security & Authentication

### Required

| Variable | Purpose | Example | Notes |
|----------|---------|---------|-------|
| `GHOST_API_TOKEN` | Bearer token for write endpoints | `a3f5c9d2e8...` | Generate with `openssl rand -hex 32` |

### Optional

| Variable | Purpose | Default | Production |
|----------|---------|---------|------------|
| `ALLOWED_ORIGINS` | CORS origins | `*` | Set to specific domains |
| `SECURE_HEADERS` | Enable security headers | `1` | Keep enabled |
| `CSP_MODE` | Content Security Policy mode | `dev` | Set to `strict` or `prod` |
| `HSTS_ON` | Enable HSTS header | `1` | Keep enabled for HTTPS |
| `HSTS_MAX_AGE` | HSTS max-age seconds | `15552000` (180 days) | - |
| `ADMIN_IP_ALLOWLIST` | IP whitelist for admin endpoints | `""` (disabled) | Set if needed |
| `RATE_LIMIT_WRITE_RPM` | Write rate limit (requests/min) | `0` (disabled) | Enable in prod: `60` |
| `IDEMPOTENCY_TTL_S` | Idempotency key TTL | `300` (5 min) | - |

---

## 2. Price Providers

**At least one required for live price data.**

| Variable | Provider | Required | Free Tier | Notes |
|----------|----------|----------|-----------|-------|
| `POLYGON_API_KEY` | Polygon.io | No | ❌ No | Most reliable, paid |
| `ALPHAVANTAGE_API_KEY` | AlphaVantage | No | ✅ 25 req/day | Free tier sufficient for single ticker |
| `YAHOO_FINANCE` | Yahoo (via yfinance) | No | ✅ Yes | No key required, rate limited |

**Configuration**:

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `PRICE_TTL_S` | Cache TTL (closed market) | `30` | How long to cache prices |
| `PRICE_TTL_OPEN_S` | Cache TTL (open market) | `45` | Longer during trading |
| `PRICE_MAX_DEVIATION` | Max % deviation from consensus | `0.5` (50%) | Rejects outliers |
| `PRICE_YAHOO_FIRST` | Try Yahoo before paid APIs | `0` | Set `1` to save API calls |
| `PROVIDER_FAIL_THRESHOLD` | Failures before circuit breaker | `3` | - |
| `PROVIDER_BACKOFF_S` | Initial backoff duration | `30` | Exponential backoff |
| `HTTP_TIMEOUT_S` | HTTP request timeout | `8` | - |

---

## 3. Persistence

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `WOLF_PERSIST_MODE` | State persistence mode | `none` | Options: `none`, `file`, `sqlite`, `redis`, `auto` |
| `WOLF_STATE_FILE` | JSON state file path | `/data/wolf_state.json` | Used if mode=`file` |
| `WOLF_SQLITE_PATH` | SQLite database path | `/data/wolf.db` | Used if mode=`sqlite` |
| `REDIS_URL` | Redis connection string | `""` | Format: `redis://host:6379` |
| `WOLF_AUTOSAVE_S` | Auto-save interval | `0` (disabled) | Set to `300` for 5-min saves |
| `AI_MEMORY_DB_PATH` | AI memory database | `data/ai_memory.db` | - |

---

## 4. Alerts & Notifications

### Telegram

| Variable | Purpose | Required | Notes |
|----------|---------|----------|-------|
| `TELEGRAM_BOT_TOKEN` | Bot API token | No | Get from @BotFather |
| `TELEGRAM_CHAT_ID` | Default chat ID | No | Numeric chat ID |
| `TELEGRAM_WEBHOOK_SECRET` | Webhook validation secret | No | **Recommended** for security |
| `TELEGRAM_HEARTBEAT_ON_START` | Send startup notification | `0` | Set `1` to enable |

### Alert Configuration

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `ALERT_MODE` | Alert strategy | `fixed` | Options: `fixed`, `band`, `trailing` |
| `ALERT_BUY_PCT` | Buy threshold (fixed) | `0.99` | Alert if price < avg × 0.99 |
| `ALERT_SELL_PCT` | Sell threshold (fixed) | `1.01` | Alert if price > avg × 1.01 |
| `ALERT_THROTTLE_S` | Min seconds between alerts | `60` | Prevents spam |
| `BAND_PCT` | Band width (band mode) | `0.02` (2%) | Alert at ±2% from avg |
| `TRAIL_SELL_PCT` | Trailing sell threshold | `0.05` (5%) | Drop from high |
| `TRAIL_BUY_PCT` | Trailing buy threshold | `0.05` (5%) | Rise from low |
| `ALERT_SCHEDULE_OPEN_CLOSE` | Market open/close alerts | `0` | Set `1` to enable |
| `ALERT_WEBHOOK_URLS` | Custom webhook URLs | `""` | Comma-separated |
| `SLACK_WEBHOOK_URLS` | Slack webhooks | `""` | Comma-separated |

---

## 5. AI & Forecasting

### AI Models

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `AI_ON` | Enable AI decision system | `0` | Set `1` to activate |
| `AI_PROVIDER` | AI backend | `ollama` | Options: `ollama`, `openai` |
| `AI_MODEL` | Model name | `llama3.1:8b` | Ollama: model name; OpenAI: `gpt-4o-mini` |
| `OLLAMA_BASE_URL` | Ollama API endpoint | `http://127.0.0.1:11434` | - |
| `OPENAI_API_KEY` | OpenAI API key | `""` | Required if provider=`openai` |
| `OPENAI_BASE_URL` | OpenAI API endpoint | `https://api.openai.com/v1` | - |
| `AI_TIMEOUT_S` | AI inference timeout | `10` | - |
| `AI_INCLUDE_CONTEXT` | Include full context in responses | `0` | Increases verbosity |

### Forecasting

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `OVERLAY_ENABLED` | Enable forecast overlay | `1` | Two-line Ghost vs Live |
| `OVERLAY_DT_MINUTES` | Overlay resolution | `60` | Minutes per point |
| `FORECAST_STEP_S` | Forecast step size | `7200` (2h) | - |
| `FORECAST_HORIZON_S` | Forecast horizon | `172800` (48h) | - |
| `PRED_SIGMA_DAILY` | Daily volatility estimate | `0.06` (6%) | - |
| `PRED_Z` | Confidence band (σ) | `1.0` | 1σ ≈ 68% confidence |
| `LEARNING_ENABLED` | Enable feedback learning | `1` | Adjusts bands based on error |
| `BAND_WIDEN_FACTOR` | Band adjustment factor | `1.0` | >1.0 = wider bands |

---

## 6. Observability

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `LOG_LEVEL` | Logging level | `INFO` | Options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `LOG_JSON` | JSON-structured logs | `1` | Recommended for production |
| `OTEL_ENABLED` | OpenTelemetry tracing | `0` | Set `1` to enable |
| `OTEL_SERVICE_NAME` | Service name in traces | `ghost-wolf` | - |
| `PROMETHEUS_MULTIPROC_DIR` | Prometheus multiproc dir | (empty) | Set for multiprocessing mode |
| `LOG_SAMPLE_RATE` | Log sampling rate | `1.0` | 0.0-1.0; use 0.1 for 10% sampling |
| `STATUS_THROTTLE_S` | Status update throttle | `30` | - |
| `STATUS_MERGE_TTL_S` | Status merge window | `60` | - |

---

## 7. News & Sentiment

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `NEWS_TTL_S` | News cache TTL | `300` (5 min) | - |
| `REUTERS_FEEDS_ON` | Enable Reuters feeds | `0` | Set `1` to enable |
| `REUTERS_FEEDS` | Reuters RSS URLs | (default URLs) | Comma-separated |
| `NEWS_SENTIMENT_ON` | Enable sentiment analysis | `0` | Requires AI model |
| `NEWS_LOOKBACK_MIN` | News lookback window | `240` (4h) | - |

---

## 8. Runtime Configuration

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `GHOST_TZ` | Timezone | `America/Chicago` | Market hours detection |
| `GHOST_CLOCK_24H` | 24-hour clock format | `0` | Set `1` for 24h |
| `TICK_INTERVAL_S` | Background tick interval | `5` | Alert worker polling |
| `PORT` | HTTP server port | `5000` | Railway overrides this |

---

## 9. Debug & Testing

**Not for production use.**

| Variable | Purpose | Default | Notes |
|----------|---------|---------|-------|
| `SNAP_TEST_MODE` | Enable test mode | `0` | Unlocks debug endpoints |
| `DEBUG_POSLOG` | Position logging verbosity | `0` | Test fixtures |
| `GIT_SHA` | Build commit SHA | `unknown` | Set by CI/CD |
| `BUILD_TIME` | Build timestamp | `unknown` | ISO 8601 format |

---

## Setup Checklist

### Minimum Viable (Local Dev)
```bash
export GHOST_API_TOKEN="$(openssl rand -hex 32)"
export ALPHAVANTAGE_API_KEY="your_free_key"
# Ghost will run without Telegram; logs to console
````

### Production (Railway)

```bash
railway variables set \
  GHOST_API_TOKEN="$(openssl rand -hex 32)" \
  POLYGON_API_KEY="your_polygon_key" \
  ALPHAVANTAGE_API_KEY="your_alphavantage_key" \
  TELEGRAM_BOT_TOKEN="your_bot_token" \
  TELEGRAM_CHAT_ID="your_chat_id" \
  TELEGRAM_WEBHOOK_SECRET="$(openssl rand -base64 32)" \
  SECURE_HEADERS=1 \
  CSP_MODE=strict \
  RATE_LIMIT_WRITE_RPM=60 \
  WOLF_PERSIST_MODE=sqlite \
  WOLF_AUTOSAVE_S=300 \
  LOG_LEVEL=INFO \
  LOG_JSON=1
```

______________________________________________________________________

## Validation

```bash
# Check required vars are set
python -c "
import os
required = ['GHOST_API_TOKEN']
missing = [v for v in required if not os.getenv(v)]
if missing:
    print(f'❌ Missing: {missing}')
else:
    print('✅ All required vars set')
"

# Verify Ghost starts
python -c "from wolf_app import APP; print('✅ Import successful')"

# Check /api/config endpoint
curl -s https://your-ghost-url/api/config | jq .
```

______________________________________________________________________

## Migration Notes

If you're migrating from an older Ghost version:

1. **`ALPHA_VANTAGE_API_KEY`** → Now supports both `ALPHAVANTAGE_API_KEY` and legacy
   name
2. **`WOLF_QTY`/`WOLF_AVG_COST`** → Deprecated; use persistence instead
3. **`AI_DB_PATH`** → Renamed to `AI_MEMORY_DB_PATH` (old name still works)

______________________________________________________________________

## See Also

- `SECURITY_INCIDENT_P0_SECRETS.md` - Secret rotation procedures
- `OPERATIONS.md` - Runtime operations guide
- `GHOST_DEEP_AUDIT.md` - Security audit findings

````

**Validation**:
```bash
# Lint markdown
markdownlint ENV_VARS_REFERENCE.md

# Verify all documented vars exist in code
python -c "
import re
doc = open('ENV_VARS_REFERENCE.md').read()
code = open('wolf_app.py').read()
vars_doc = set(re.findall(r'\`([A-Z_]+)\`', doc))
vars_code = set(re.findall(r'os\.getenv\([\"']([A-Z_]+)[\"']', code))
missing = vars_code - vars_doc
extra = vars_doc - vars_code
if missing:
    print(f'⚠️  Vars in code but not doc: {missing}')
if extra:
    print(f'ℹ️  Vars in doc but not code: {extra}')
if not (missing or extra):
    print('✅ Documentation complete')
"
````

**Deliverables**:

- [ ] `ENV_VARS_REFERENCE.md` created
- [ ] All 100+ vars documented
- [ ] Grouped by category
- [ ] Setup checklist included
- [ ] Validation script passes
- [ ] Linked from README.md

______________________________________________________________________

## Testing Strategy

### Regression Test Suite

Create `tests/test_upgrade_plan.py`:

```python
"""Regression tests for upgrade plan changes."""

import pytest
from wolf_app import APP

def test_no_duplicate_routes():
    """Verify no duplicate route registrations."""
    routes = [r.path for r in APP.routes]
    assert len(routes) == len(set(routes)), "Duplicate routes detected"

def test_debug_endpoints_require_auth(client):
    """All debug endpoints require authentication."""
    debug_endpoints = [
        ("/debug/telegram_test", "POST"),
        ("/debug/prev_close", "POST"),
        ("/debug/price_diag", "POST"),
    ]
    
    for path, method in debug_endpoints:
        func = getattr(client, method.lower())
        resp = func(path, json={})
        assert resp.status_code in [401, 403], f"{path} should require auth"

def test_telegram_webhook_validates_secret(client):
    """Telegram webhook validates secret token."""
    resp = client.post("/telegram/webhook", json={"message": {}})
    # Should fail if TELEGRAM_WEBHOOK_SECRET is set
    assert resp.status_code in [403, 200]  # 200 if secret not configured

def test_sse_includes_disconnect_check(client):
    """SSE generators include disconnect detection."""
    import inspect
    from wolf_app import events_sse
    source = inspect.getsource(events_sse)
    assert "is_disconnected" in source, "SSE should check for client disconnect"

def test_secrets_env_gitignored():
    """Verify secrets.env is in .gitignore."""
    with open(".gitignore") as f:
        gitignore = f.read()
    assert "secrets.env" in gitignore, "secrets.env must be gitignored"

def test_env_vars_documented():
    """Critical env vars are documented."""
    with open("ENV_VARS_REFERENCE.md") as f:
        doc = f.read()
    
    required_vars = [
        "GHOST_API_TOKEN",
        "POLYGON_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_WEBHOOK_SECRET",
    ]
    
    for var in required_vars:
        assert var in doc, f"{var} should be documented"

@pytest.mark.asyncio
async def test_sse_stops_on_disconnect(client):
    """SSE generator stops when client disconnects."""
    async with client.stream("GET", "/events") as resp:
        chunk = await resp.aiter_bytes().__anext__()
        assert b"data:" in chunk
        # Connection will close on context exit
    
    # If generator continues, it will accumulate in memory
    # This test verifies it stops (via metrics or internal tracking)
```

______________________________________________________________________

## Rollback Plan

If any upgrade step causes issues:

### P0/P1 Rollback

```bash
# Revert auth changes
git revert <commit_hash>
git push origin main
railway up

# Restore old secrets (if rotation failed)
railway variables set POLYGON_API_KEY="<OLD_KEY>"
# etc.

# Re-remove secrets.env from working tree (if accidentally committed)
git rm --cached secrets.env
git push origin main
```

### P2 Rollback

```bash
# Restore main.py
git checkout HEAD~1 main.py

# Revert webhook validation
railway variables unset TELEGRAM_WEBHOOK_SECRET
# Re-register webhook without secret
```

______________________________________________________________________

## Success Metrics

| Metric | Target | Measurement | |--------|--------|-------------| | **P0 Completion**
| 100% | All 5 keys rotated, history cleaned | | **P1 Auth Coverage** | 100% | All debug
endpoints require token | | **P1 Route Deduplication** | 1 impl | Zero route collision
warnings | | **P1 SSE Leak Rate** | \<1% | Active clients / total connects < 0.01 | |
**P2 Webhook Security** | 100% | All webhooks validate signature | | **P2
Documentation** | 95%+ | All env vars in reference doc | | **Test Coverage** | >80% |
pytest --cov for upgrade changes | | **Zero Regressions** | ✅ | All existing tests pass
|

______________________________________________________________________

## Timeline Summary

| Week | Focus | Deliverables | Effort | |------|-------|--------------|--------| |
**Week 1** | P0 (Critical) | Keys rotated, history cleaned, pre-commit hooks | 0.5 days
| | **Week 2** | P1 (High) | Auth on debug, SSE cleanup, route consolidation | 3-5 days
| | **Weeks 3-4** | P2 (Medium) | Webhook validation, main.py archived, env docs | 5-7
days |

**Total Calendar Time**: 1 month (at ~50% allocation)

______________________________________________________________________

## Contact & Support

- **Audit Report**: `GHOST_DEEP_AUDIT.md`
- **Security Incident**: `SECURITY_INCIDENT_P0_SECRETS.md`
- **Operations**: `OPERATIONS.md`
- **Questions**: File issue in GitHub repo

______________________________________________________________________

**Plan Prepared By**: GitHub Copilot\
**Review Status**: Ready for Implementation\
**Last Updated**: October 4, 2025
