# Ghost Cockpit Live Implementation Status

## ✅ COMPLETED MODULES

### 1. core/telegram_alerts.py
**Status:** ✅ Created and linted
**Features:**
- `render_alert()`: Single source of truth for alert formatting
- Standardized message template with timezone support (America/Chicago default)
- Deduplication via Redis SET with 24h TTL
- No 0% confidence contradictions (forces HOLD when confidence < 0.10)
- After-hours detection and labeling
- Horizon buckets: SHORT (2h-30d), LONG (30d-6m)
- `send_alert()`: Sends with dedup check
- `get_recent_alerts()`: Returns last 20 alert envelopes

### 2. core/cache_tools.py
**Status:** ✅ Created and linted
**Features:**
- `purge_prices(symbols)`: Namespace-safe price cache purge (DEL price:{symbol}:*)
- `purge_alert_dedup(older_than_days)`: Clean old alert dedup keys
- `get_cache_stats()`: Count price/alert/other keys
- **NO FLUSHDB** - only targeted deletions

### 3. core/beast_scheduler.py
**Status:** ✅ Created and linted
**Features:**
- Stock schedule (CT): 07:55, 09:35, 12:00, 15:10
- Crypto schedule: Every 2 hours (:00 on even hours)
- Horizon support: SHORT + LONG
- `start_beast_scheduler()`: Background thread
- `stop_beast_scheduler()`: Graceful shutdown
- `trigger_manual_prediction()`: Manual testing
- Watch lists: AAPL, WOLF, NVDA + 10 crypto symbols

### 4. tests/test_live_pipeline.py
**Status:** ✅ Created and linted
**Tests:**
- Price diagnostics (AAPL, WOLF, BTC)
- Prediction endpoints (AAPL, BTC)
- SSE stream (status, ping, snapshot events)
- Telegram alert format (dry run)
- Summary with pass/fail counts

## ⚠️  PENDING CHANGES TO wolf_app.py

Due to wolf_app.py's size (20,319 lines), here are the required changes:

### Change 1: ENV Validation at Startup
**Location:** After line 3740 (where SIM_MODE is read)
**Action:** Add config enforcement block

```python
# ENFORCE RUNTIME CONFIG (Ghost Cockpit Live Mode)
REQUIRED_ENV_VARS = {
    "SIM_MODE": "0",
    "DELISTED_MODE": "0",
    "PRICE_STRICT_LIVE": "1",
    "STOCKS_ENABLED": "1",
    "PREDICT_STOCKS_ENABLED": "1",
    "CRYPTO_ENABLED": "1",
    "STOCK_PRICE_SOURCE": "polygon",
    "GHOST_TZ": "America/Chicago",
}

# Validate and log config
_config_errors = []
for key, expected in REQUIRED_ENV_VARS.items():
    actual = os.getenv(key, "").strip()
    if actual != expected:
        _config_errors.append(f"{key}={actual} (expected {expected})")

if _config_errors:
    print("[GHOST CONFIG] ⚠️  Configuration mismatches:")
    for err in _config_errors:
        print(f"  - {err}")
    # Don't fail startup, but log clearly
else:
    print("[GHOST CONFIG] ✅ All required ENV vars validated")

# Ensure these are actually enforced
if os.getenv("SIM_MODE", "0") != "0":
    print("[GHOST CONFIG] ❌ FATAL: SIM_MODE must be 0 for live mode")
    exit(1)

if os.getenv("PRICE_STRICT_LIVE", "0") != "1":
    print("[GHOST CONFIG] ⚠️  WARNING: PRICE_STRICT_LIVE not set, using cached prices")
```

### Change 2: Price Provider Order Enforcement
**Location:** Around line 1269 (where _DEFAULT_PROVIDER_ORDER is defined)
**Action:** Ensure order is: polygon → alphavantage → yfinance → yahoo

```python
# Ensure provider order is correctly set
_DEFAULT_PROVIDER_ORDER = ("polygon", "alphavantage", "yfinance", "yahoo")

# Allow ENV override
_PROVIDER_ORDER_ENV = os.getenv("STOCK_PRICE_SOURCE", "").strip()
if _PROVIDER_ORDER_ENV:
    # If single provider specified, try it first then fall back to defaults
    _custom_order = [_PROVIDER_ORDER_ENV.lower()]
    for p in _DEFAULT_PROVIDER_ORDER:
        if p not in _custom_order:
            _custom_order.append(p)
    _DEFAULT_PROVIDER_ORDER = tuple(_custom_order)
    print(f"[PRICE] Provider order (ENV override): {_DEFAULT_PROVIDER_ORDER}")
else:
    print(f"[PRICE] Provider order (default): {_DEFAULT_PROVIDER_ORDER}")
```

### Change 3: After-Hours Detection in Price Functions
**Location:** In get_wolf_price() and similar price fetching functions
**Action:** Add after_hours flag when price == prev_close and market closed

```python
# Example modification (find actual function and adapt):
def get_wolf_price():
    # ... existing code ...
    
    # Detect if we're using prev_close as fallback
    after_hours = False
    if price == prev_close and provider in ["fallback", "cache"]:
        # Check if market is closed
        now = datetime.now(ZoneInfo("America/New_York"))
        if now.weekday() >= 5 or now.hour < 9 or now.hour >= 16:
            after_hours = True
    
    return price, prev_close, provider, after_hours
```

### Change 4: SSE /api/cockpit/stream Enhancement
**Location:** Find existing /api/cockpit/stream endpoint (search for "cockpit/stream")
**Action:** Add status event on connect, ping heartbeat, auth check

```python
@APP.get("/api/cockpit/stream")
async def api_cockpit_stream(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(bearer_auth),
):
    """Enhanced SSE stream with status, ping, and snapshot events"""
    
    # Validate auth
    if not _validate_bearer_token(credentials.credentials):
        # Send error event then close
        async def auth_error_stream():
            yield 'event: status\n'
            yield 'data: {"status":"auth_error","message":"Invalid token"}\n\n'
        return StreamingResponse(auth_error_stream(), media_type="text/event-stream")
    
    async def event_stream():
        # Send initial status
        yield 'event: status\n'
        yield f'data: {{"mode":"live","ts":{int(time.time())},"sim_mode":{int(SIM_MODE)}}}\n\n'
        
        last_ping = time.time()
        
        while True:
            # Heartbeat ping every 10 seconds
            now = time.time()
            if now - last_ping >= 10:
                yield 'event: ping\n'
                yield f'data: {{"ts":{int(now)}}}\n\n'
                last_ping = now
            
            # Send snapshot data when available
            # ... existing snapshot logic ...
            yield 'event: snapshot\n'
            yield f'data: {json.dumps(snapshot_data)}\n\n'
            
            await asyncio.sleep(1)
    
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

### Change 5: /api/health/alerts Endpoint
**Location:** Add near other /api/health endpoints
**Action:** Return last 20 alert envelopes

```python
@APP.get("/api/health/alerts")
async def api_health_alerts():
    """Get recent alert envelopes for verification"""
    from core import telegram_alerts
    
    alerts = telegram_alerts.get_recent_alerts(limit=20)
    
    return {
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": int(time.time()),
    }
```

### Change 6: /api/regime/current Default Response
**Location:** Find existing /api/regime/current endpoint
**Action:** Ensure it returns 200 with default neutral when macro brain disabled

```python
@APP.get("/api/regime/current")
async def api_regime_current():
    """Get current market regime (always returns 200)"""
    try:
        # ... existing logic ...
        if regime_data:
            return regime_data
    except Exception as e:
        # Log but don't fail
        logger.error(f"Regime detection error: {e}")
    
    # Default neutral response
    return {
        "regime": "neutral",
        "confidence": 0.5,
        "factors": [],
        "source": "default",
        "timestamp": int(time.time()),
    }
```

### Change 7: Module Initialization (at bottom of file, before if __name__)
**Location:** Around line 20300 (before `if __name__ == "__main__"`)
**Action:** Initialize new modules

```python
# Initialize Telegram Alerts Module
try:
    from core import telegram_alerts
    telegram_alerts.REDIS_CLIENT = _REDIS
    telegram_alerts.TELEGRAM_SEND_FUNC = _tg_send_chat_message
    telegram_alerts.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    telegram_alerts.LOGGER = logger
    print("[INIT] ✅ Telegram alerts module initialized")
except Exception as e:
    print(f"[INIT] ⚠️  Telegram alerts init failed: {e}")

# Initialize Cache Tools
try:
    from core import cache_tools
    cache_tools.REDIS_CLIENT = _REDIS
    cache_tools.LOGGER = logger
    print("[INIT] ✅ Cache tools module initialized")
except Exception as e:
    print(f"[INIT] ⚠️  Cache tools init failed: {e}")

# Initialize Beast Scheduler
try:
    from core import beast_scheduler
    beast_scheduler.REDIS_CLIENT = _REDIS
    beast_scheduler.LOGGER = logger
    beast_scheduler.GET_PRICE_FUNC = _get_price_for_beast
    beast_scheduler.RUN_PREDICTION_FUNC = _run_prediction_for_beast
    beast_scheduler.TELEGRAM_ALERTS_MODULE = telegram_alerts
    
    # Start scheduler if enabled
    if os.getenv("BEAST_SCHEDULER_ENABLED", "1") == "1":
        beast_scheduler.start_beast_scheduler()
    
    print("[INIT] ✅ Beast scheduler initialized")
except Exception as e:
    print(f"[INIT] ⚠️  Beast scheduler init failed: {e}")
```

### Helper Functions Needed
**Location:** Add near other helper functions

```python
def _get_price_for_beast(symbol: str, market: str):
    """
    Get price for beast scheduler
    Returns: (price, prev_close, provider, after_hours)
    """
    if market == "stock":
        # Use existing stock price function
        price_data = get_stock_price(symbol)  # Find actual function name
        # ... adapt to return tuple ...
    else:
        # Use crypto price function
        price_data = get_crypto_price(symbol)  # Find actual function name
        # ... adapt to return tuple ...
    
    return price_data

def _run_prediction_for_beast(symbol: str, market: str, horizon: str):
    """
    Run prediction for beast scheduler
    Returns: prediction dict with action, confidence, direction, factors
    """
    # Use existing prediction logic
    # ... adapt to return standardized dict ...
    pass
```

## 🚀 DEPLOYMENT CHECKLIST

### 1. Configuration Enforcement
- [ ] Add ENV validation block after SIM_MODE definition
- [ ] Add fatal check for SIM_MODE != 0
- [ ] Log all config values at startup

### 2. Price Provider Fixes
- [ ] Verify _DEFAULT_PROVIDER_ORDER = (polygon, alphavantage, yfinance, yahoo)
- [ ] Add after_hours detection to price functions
- [ ] Update all price returns to include after_hours flag

### 3. SSE Endpoint Enhancement
- [ ] Find existing /api/cockpit/stream
- [ ] Add status event on connect
- [ ] Add ping heartbeat every 10s
- [ ] Add snapshot event for data changes
- [ ] Add auth error handling

### 4. New Endpoints
- [ ] Add /api/health/alerts endpoint
- [ ] Fix /api/regime/current to always return 200

### 5. Module Initialization
- [ ] Import and initialize telegram_alerts module
- [ ] Import and initialize cache_tools module
- [ ] Import and initialize beast_scheduler module
- [ ] Create helper functions for beast scheduler

### 6. Testing
- [ ] Run tests/test_live_pipeline.py
- [ ] Verify 0×499, 0×502 errors
- [ ] Check SSE stream outputs
- [ ] Verify Telegram alerts (dry run)

## 📊 VERIFICATION RUNBOOK

```bash
# 1. Config enforce + safe cache purge
python << 'PYEOF'
import os
from core import cache_tools

# Verify ENV
required = {
    "SIM_MODE": "0",
    "DELISTED_MODE": "0",
    "PRICE_STRICT_LIVE": "1",
}

for k, v in required.items():
    actual = os.getenv(k, "")
    print(f"{k}: {actual} (expect {v})")

# Purge specific symbol caches
symbols = ["AAPL", "WOLF", "BTC", "ETH", "SOL"]
deleted = cache_tools.purge_prices(symbols)
print(f"Purged {deleted} price cache keys")

# Purge old dedup keys
deleted_dedup = cache_tools.purge_alert_dedup(older_than_days=7)
print(f"Purged {deleted_dedup} old dedup keys")
PYEOF

# 2. Provider self-tests
curl -s "$GHOST_BASE_URL/api/price/diagnostics?symbol=AAPL" | jq '.price, .provider'
curl -s "$GHOST_BASE_URL/api/price/diagnostics?symbol=WOLF" | jq '.price, .provider'
curl -s "$GHOST_BASE_URL/api/crypto/price/BTC" | jq '.price'

# 3. Predict and alerts dry run
TOKEN=$GHOST_API_TOKEN
curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -X POST -d '{"symbol":"AAPL"}' "$GHOST_BASE_URL/api/predict/run" | jq '.prediction_id, .price'

curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -X POST -d '{"symbol":"BTC"}' "$GHOST_BASE_URL/api/predict/run" | jq '.prediction_id, .price'

# 4. SSE check
curl -Ns -H "Authorization: Bearer $TOKEN" "$GHOST_BASE_URL/api/cockpit/stream" | head -c 4096

# 5. HTTP logs acceptance (5 min window)
# Monitor logs for 499/502 errors - should be 0
```

## 📝 HEALTH_REPORT.json Template

```json
{
  "timestamp": "2025-01-10T12:00:00Z",
  "config": {
    "SIM_MODE": "0",
    "DELISTED_MODE": "0",
    "PRICE_STRICT_LIVE": "1",
    "STOCKS_ENABLED": "1",
    "CRYPTO_ENABLED": "1"
  },
  "sse": {
    "status": true,
    "ping_interval_s": 10,
    "snapshot_working": true
  },
  "providers": {
    "AAPL": {
      "price": 150.25,
      "provider": "polygon",
      "after_hours": false
    },
    "BTC": {
      "price": 45000.00,
      "provider": "coingecko",
      "after_hours": false
    }
  },
  "alerts": {
    "last_20": [
      {"market": "stock", "symbol": "AAPL", "horizon": "SHORT", "date": "2025-01-10"},
      {"market": "crypto", "symbol": "BTC", "horizon": "LONG", "date": "2025-01-10"}
    ]
  },
  "http": {
    "499": 0,
    "502": 0,
    "window_minutes": 5
  },
  "tests": {
    "price_diagnostics": "passed",
    "predictions": "passed",
    "sse_stream": "passed",
    "telegram_format": "passed"
  }
}
```

## 🔄 ROLLBACK PROCEDURE

```bash
# This implementation is in separate modules, so rollback is clean:

# 1. Stop beast scheduler
python << 'PYEOF'
from core import beast_scheduler
beast_scheduler.stop_beast_scheduler()
PYEOF

# 2. Revert wolf_app.py changes
git revert <commit_hash>

# 3. Remove new modules (if needed)
rm -f core/telegram_alerts.py core/cache_tools.py core/beast_scheduler.py
rm -f tests/test_live_pipeline.py

# 4. Restart server
pkill -f wolf_app
python wolf_app.py
```

## ✅ COMPLETED ITEMS

1. ✅ Git installed and initialized
2. ✅ cSpell configuration fixed (renamed to cspell.json)
3. ✅ core/telegram_alerts.py created with full functionality
4. ✅ core/cache_tools.py created with namespace-safe operations
5. ✅ core/beast_scheduler.py created with scheduled jobs
6. ✅ tests/test_live_pipeline.py created with comprehensive tests
7. ✅ All new files linted and formatted with Ruff

## ⏳ PENDING ITEMS

1. ⏳ wolf_app.py modifications (detailed above)
2. ⏳ Helper function implementations (_get_price_for_beast, _run_prediction_for_beast)
3. ⏳ Module initialization code
4. ⏳ SSE endpoint enhancement
5. ⏳ ENV validation enforcement
6. ⏳ Test execution and validation
7. ⏳ HEALTH_REPORT.json generation
