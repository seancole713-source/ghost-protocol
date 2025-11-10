# Prometheus Metrics Endpoint - Debug Report

## Issue

Metrics endpoint `/metrics` returns 200 OK but with `content-length: 0` (empty body)
locally.

## Probable Root Causes

### 1. **Multiprocess Mode Misconfiguration**

The metrics endpoint (`wolf_app.py:3343-3362`) uses Prometheus multiprocess mode when
`PROMETHEUS_MULTIPROC_DIR` is set:

```python
mp_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR", "").strip()
if mp_dir:
    from prometheus_client import CollectorRegistry, multiprocess
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    blob = generate_latest(registry)
    return Response(blob, media_type=CONTENT_TYPE_LATEST)
```

**Issue:** If the directory exists but has no metric files (because no metrics have been
incremented yet), it returns empty output.

### 2. **Lazy Metric Registration**

Metrics are registered on first use, not at module load time. If no endpoints have been
called that increment metrics, the registry remains empty.

Example metrics defined but not necessarily incremented:

- `ghost_price_fetch_seconds` - only incremented when price providers are called
- `ghost_telegram_send_total` - only when Telegram sends occur
- `ghost_snapshot_duration_seconds` - only on `/api/cockpit` calls

### 3. **Default Registry Not Used**

If multiprocess mode is active, the default registry (where metrics are initially
registered) is not consulted. Only files in `PROMETHEUS_MULTIPROC_DIR` are read.

## Solutions

### Option A: Force Metric Initialization at Startup

Add to startup handler after metric definitions (`wolf_app.py:~1850`):

```python
# Force initial metric registration
try:
    # Increment all counters with 0 to register them
    _PRICE_FETCH_SECONDS.labels(provider="init", throttled="false").observe(0)
    _TELEGRAM_SEND_TOTAL.labels(result="init").inc(0)
    _SNAP_DURATION_SECONDS.observe(0)
    LOGGER.info("metrics_initialized", extra={"component": "startup"})
except Exception as e:
    LOGGER.warning("metrics_init_failed", extra={"component": "startup", "error": str(e)})
```

### Option B: Disable Multiprocess Mode for Development

In local dev, don't set `PROMETHEUS_MULTIPROC_DIR`:

```bash
# In secrets.env or environment, REMOVE or comment:
# export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom

# Restart server - will use default registry (single-process mode)
```

### Option C: Force Metric Export on Health Checks

Modify `/health` or `/ready` to increment a heartbeat metric:

```python
@APP.get("/health")
async def health():
    if _G_UP:
        _G_UP.set(1)  # Already done
    if _HEARTBEAT_COUNTER:  # NEW
        _HEARTBEAT_COUNTER.inc()
    return {"status": "ok"}
```

## Verification Steps

### 1. Check Multiprocess Directory

```bash
ls -lah /tmp/ghost_prom/
# Should show .db files if metrics are being written
```

### 2. Force Metric Generation

```bash
# Call endpoints that increment metrics
curl http://localhost:5000/api/cockpit
curl http://localhost:5000/api/telegram/test?send=false
curl http://localhost:5000/health

# Then check metrics
curl http://localhost:5000/metrics
```

### 3. Check Default Registry (Single Process)

```bash
# Temporarily disable multiprocess
unset PROMETHEUS_MULTIPROC_DIR
# Restart server
# Check metrics - should show Python process metrics at minimum
curl http://localhost:5000/metrics | head -20
```

## Expected Output (When Working)

```prometheus
# HELP ghost_price_fetch_seconds Time spent fetching price from providers
# TYPE ghost_price_fetch_seconds histogram
ghost_price_fetch_seconds_bucket{le="0.1",provider="yahoo",throttled="false"} 3
ghost_price_fetch_seconds_bucket{le="0.5",provider="yahoo",throttled="false"} 5
...

# HELP ghost_telegram_send_total Total Telegram messages sent
# TYPE ghost_telegram_send_total counter
ghost_telegram_send_total{result="success"} 12
...

# HELP ghost_up Server is up and running
# TYPE ghost_up gauge
ghost_up 1
```

## Recommended Fix (Immediate)

**Add eagerinitializer in `wolf_app.py` startup:**

```python
def _ensure_metrics_registered():
    """Force metric registration by observing/incrementing with zero values"""
    try:
        # Price fetch metrics
        for provider in ["yahoo", "alphavantage", "polygon", "yfinance"]:
            _PRICE_FETCH_SECONDS.labels(provider=provider, throttled="false").observe(0.001)
        
        # Telegram metrics
        _TELEGRAM_SEND_TOTAL.labels(result="success").inc(0)
        _TELEGRAM_TEST_TOTAL.labels(result="preview").inc(0)
        
        # Snapshot metrics
        _SNAP_DURATION_SECONDS.observe(0.001)
        
        # Alert metrics
        _ALERT_LATENCY_SECONDS.observe(0.001)
        
        return True
    except Exception as e:
        LOGGER.exception("metrics_registration_failed", extra={"error": str(e)})
        return False

# In @APP.on_event("startup") around line 1850:
try:
    _ensure_metrics_registered()
    LOGGER.info("metrics_registry_initialized", extra={"component": "startup"})
except Exception:
    LOGGER.exception("metrics_init_failed", extra={"component": "startup"})
```

## Production Deployment Note

In production (Railway), metrics may work correctly because:

1. Server has been running longer with actual traffic
2. Metrics naturally accumulate from real requests
3. Multiprocess mode may not be enabled (check `PROMETHEUS_MULTIPROC_DIR` in Railway env
   vars)

## Related Code

- Metrics endpoint: `wolf_app.py:3343-3362`
- Metric definitions: `wolf_app.py:1950-2100` (approx)
- Startup handler: `wolf_app.py:1629-1850`

## Status

⚠️ **DOCUMENTED** - Root cause identified. Requires code change for eager initialization
or environment adjustment.

______________________________________________________________________

**Updated:** 2025-10-07 16:00 UTC\
**Sprint:** Deep Scrub + Full Fix - Phase 4
