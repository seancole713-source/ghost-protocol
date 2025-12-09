# Railway Health Check Status

_Last updated: 2025-11-23 13:54 UTC_

## Configuration Snapshot

- `railway.toml` keeps `healthcheckPath = "/health"`, a 100 second timeout, and launches

  `uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080}` so Railway and the Docker
  container use the same entry point.

- `Dockerfile` mirrors the uvicorn command in `CMD`, exposes port 8080, and leaves health

  probing to Railway (no overlapping container healthcheck directive).

- `wolf_app.py` (line 1120) defines `@APP.get("/health")` that returns

  `{"status":"ok","service":"ghost-protocol","uptime":<seconds>}` without touching Redis,
  price feeds, or any other optional providers, so the endpoint responds even while startup
  jobs are still warming up.

## Local Verification (macOS) - Re-tested Nov 23

1. Started the app with `PYTHONPATH=/Users/studio713/ghost-protocol python3 -m uvicorn wolf_app:APP --host 127.0.0.1 --port 8080`
2. `curl -s -D - <<<<<http://127.0.0.1:8080/health`>>>>> returned `HTTP/1.1 200 OK` in <1s:

   ```json
   HTTP/1.1 200 OK
   content-type: application/json
   {"status":"ok","service":"ghost-protocol","uptime":21}

   ```text

3. Application logs showed no provider errors blocking `/health` response.

4. Confirmed endpoint remains ultra-lightweight and startup-safe.

## Recent Fixes (Nov 23, 2025)

### Meme Ticker Provider Storm Resolution

**Commit**: `3e98a7c` - Deployed to Railway

**Problem**:

- Unsupported meme tickers (WEPE, LILPEPE, DORKL, SLOTH, APC) in `HUNTER_CRYPTO_SYMBOLS`
- Caused 6-8s of failed API calls during crypto movers scan:
  - CoinGecko: "Unknown symbol" warnings
  - Binance: "All endpoints exhausted" errors
  - Coinbase: 404 Not Found responses
- Added significant startup latency, contributing to Railway healthcheck timeouts

**Solution**:

```python

# Before

HUNTER_CRYPTO_SYMBOLS = ["WEPE", "LILPEPE", "DORKL", "SLOTH", "APC", "BTC"]

# After (only exchange-supported coins)

HUNTER_CRYPTO_SYMBOLS = ["BTC", "ETH", "SOL", "DOGE", "ADA", "XRP"]

```text

**Impact**:

- Eliminates provider 404/429 storms during startup
- Reduces startup latency by ~6-8 seconds
- Allows `/health` endpoint to respond within Railway's 100s window
- Removed VIP coin classification logic (no longer needed)


## Smoke Script Coverage

- Made `smoke-test.sh` executable and ran `./smoke-test.sh <<<<<http://127.0.0.1:8100`>>>>> while the


  server was running. The script's first step hits `/health`, and the run completed with the
  success banner, proving the smoke workflow now covers the healthcheck explicitly.

## Production Verification Command

Once Railway build completes (commit `3e98a7c`):

```bash

curl -s -D - <<<<<https://ghost-protocol-production.up.railway.app/health>>>>>

```text

**Expected Response**:

```text

HTTP/1.1 200 OK
...
{"status":"ok","service":"ghost-protocol","uptime":N}

```text

## Outstanding Items

- **Railway Deploy**: Awaiting build completion for commit `3e98a7c`
- **Production Smoke**: Run `scripts/ghost_smoke.sh railway` after deploy succeeds
- **HTTP Logs**: Monitor Railway logs to confirm no meme ticker warnings (WEPE, LILPEPE, etc.)
