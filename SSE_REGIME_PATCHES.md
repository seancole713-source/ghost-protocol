# Ghost Cockpit SSE & Regime Endpoint Enhancements

# Mission: Add proper SSE event types and /api/regime/current endpoint

# Date: 2025-11-10

## PHASE 2: Patches Required for wolf_app.py

### Patch 1: Add /api/regime/current endpoint (around line 10910)

```python
@APP.get("/api/regime/current")
async def api_regime_current():
    """
    Get current market regime (neutral fallback if Stage 3 not enabled).
    Returns: {"regime": str, "ts": int, "confidence": float}
    """
    try:
        if STAGE3_ENABLED:
            regime_detector = get_regime_detector()
            return {
                "regime": regime_detector.current_regime.lower(),
                "ts": int(time.time()),
                "confidence": float(regime_detector.confidence),
                "source": "stage3_detector"
            }
        else:

            # Fallback: neutral regime when Stage 3 disabled

            return {
                "regime": "neutral",
                "ts": int(time.time()),
                "confidence": 0.5,
                "source": "fallback"
            }
    except Exception as e:
        LOGGER.error(f"regime_current_error: {e}")
        return {
            "regime": "neutral",
            "ts": int(time.time()),
            "confidence": 0.5,
            "source": "error_fallback",
            "error": str(e)
        }

```text

### Patch 2: Enhance SSE /api/cockpit/stream with event types (line 11653-11730)

**CRITICAL CHANGE:**Replace current SSE implementation with proper event types:

```python

@APP.get("/api/cockpit/stream")
async def sse_cockpit_stream(request: Request):
    """SSE stream with proper event types: status, ping, snapshot."""
    async def gen():
        last_sent_etag = None
        start_time = time.time()
        last_heartbeat = time.time()

        # Event 1: Send status event on connect

        try:
            status_data = {
                "status": "live",
                "ts": int(time.time()),
                "sim_mode": SIM_MODE,
                "focus_wolf_only": FOCUS_WOLF_ONLY
            }
            yield f"event: status\\ndata: {json.dumps(status_data)}\\n\\n"
        except Exception:
            pass

        # Event 2: Send initial snapshot immediately

        try:
            snap_resp = await api_cockpit()
            data = getattr(snap_resp, "body", None)
            if data is None:
                if isinstance(snap_resp, JSONResponse):
                    try:
                        content = snap_resp.body if hasattr(snap_resp, "body") else b"{}"
                        data = content if isinstance(content, bytes) else json.dumps(content).encode("utf-8")
                    except Exception:
                        data = b"{}"
                elif isinstance(snap_resp, dict):
                    data = json.dumps(snap_resp).encode("utf-8")
                else:
                    data = json.dumps(str(snap_resp)).encode("utf-8")

            yield f"event: snapshot\\ndata: {data.decode('utf-8')}\\n\\n"
        except Exception as e:
            LOGGER.error(f"sse_initial_snapshot_error: {e}")

        while True:

            # Check if client disconnected

            if await request.is_disconnected():
                LOGGER.info("SSE cockpit client disconnected")
                break

            # TTL: Close stream after 30 minutes

            if time.time() - start_time > 1800:
                LOGGER.info("SSE cockpit stream TTL expired (30min)")
                break

            # Event 3: Send ping every 10 seconds (reduced from 15s for better responsiveness)

            if time.time() - last_heartbeat > 10:
                ping_data = {"ts": int(time.time())}
                yield f"event: ping\\ndata: {json.dumps(ping_data)}\\n\\n"
                last_heartbeat = time.time()

            # Wait 5 seconds between snapshot checks

            await _async_sleep(5.0)

            # Event 4: Send snapshot if data changed

            try:
                snap_resp = await api_cockpit()
                raw = getattr(snap_resp, "body", None)
                if raw is None:
                    raw = json.dumps(snap_resp).encode("utf-8")  # type: ignore

                # Naive change detection by ETag

                etag = None
                try:
                    etag = getattr(snap_resp, "headers", {}).get("ETag")  # type: ignore
                except Exception:
                    etag = None

                if etag:
                    if etag == last_sent_etag:
                        continue  # No change, skip sending
                    last_sent_etag = etag

                yield f"event: snapshot\\ndata: {raw.decode('utf-8')}\\n\\n"
            except Exception as e:
                LOGGER.error(f"sse_snapshot_error: {e}")
                continue

    return StreamingResponse(gen(), media_type="text/event-stream")

```text

### Patch 3: Verify /api/price/<symbol> uses ensure_price_cached**Already correct**- Line 8948 shows

```python

@APP.get("/api/price/{symbol}")
async def api_price_symbol(symbol: str):
    price, prev, provider, fresh = await ensure_price_cached(symbol.upper())
    return {
        "symbol": symbol.upper(),
        "price": price,
        "prev_close": prev,
        "provider": provider,
        "fresh": fresh
    }

```text

This already returns instantly on cache hit. ✅

### Patch 4: Confirm /api/portfolio and /api/position use cached snapshots**Location check needed**- Verify these endpoints don't block on price providers

______________________________________________________________________

## Implementation Priority

1.**HIGH**: Add `/api/regime/current` endpoint (simple fallback, prevents 404s)

1. **HIGH**: Enhance SSE stream with event types (status/ping/snapshot)
2. **MEDIUM**: Verify /api/portfolio and /api/position are non-blocking
3. **LOW**: Add /api/admin/cache/flush endpoint if missing


______________________________________________________________________

## Testing After Patches

```bash

# Test regime endpoint

curl -s "$BASE_URL/api/regime/current" | python -m json.tool

# Test SSE with event types

curl -N "$BASE_URL/api/cockpit/stream" | grep -E "^event:"

# Expected output

# event: status

# event: snapshot

# event: ping

# event: snapshot

```text