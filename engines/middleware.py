"""HTTP middleware registration for Ghost Protocol Wolf.

Extracted from wolf_app.py (Step 12 cleanup). Call register_middleware(APP)
from wolf_app.py after the FastAPI app and CORS are configured.
"""
import logging
import os
import random
import threading
import time
import traceback
import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from wolf_helpers import _cv_trace_id, _cv_path, _cv_method, _compute_csp

# ── Rate-limiter state (module-level so `global` works inside _rate_limit_mw) ─
RATE_LIMIT_WRITE_RPM = int(os.getenv("RATE_LIMIT_WRITE_RPM", "0"))  # 0 disables
RATE_LIMIT_EXEMPT_AUTH = int(os.getenv("RATE_LIMIT_EXEMPT_AUTH", "1"))
_RATE_CAPACITY = max(0, RATE_LIMIT_WRITE_RPM)
_RATE_TOKENS = float(_RATE_CAPACITY)
_RATE_LAST_REFILL = time.monotonic()

LOGGER = logging.getLogger("ghost")


def register_middleware(APP):
    """Register all HTTP middleware on the FastAPI application.

    Call this once from wolf_app.py after APP creation and CORS setup.
    Middleware is applied in LIFO order: last registered = outermost.
    """
    import engines.app_config as _ac  # late import avoids circular deps

    # ── Capture app_config globals so closures below can access them ──────
    IP_ALLOWLIST = getattr(_ac, "IP_ALLOWLIST", set())
    IP_ALLOWLIST_ENABLED = getattr(_ac, "IP_ALLOWLIST_ENABLED", False)
    _OTEL_TRACER = getattr(_ac, "_OTEL_TRACER", None)
    ADMIN_IP_ALLOWLIST = getattr(_ac, "ADMIN_IP_ALLOWLIST", [])
    _G_RATE_LIMIT_TOKENS = getattr(_ac, "_G_RATE_LIMIT_TOKENS", None)
    _C_RATE_LIMIT_DROPS = getattr(_ac, "_C_RATE_LIMIT_DROPS", None)

    # Insert after APP definition (search earlier in file for FastAPI instantiation)
    try:
        _APP_INSTRUMENTED  # type: ignore
    except NameError:
        try:
            import time
            import traceback
            import uuid

            from fastapi import Request

            @APP.middleware("http")
            async def _exception_diagnostics_mw(request: Request, call_next):  # type: ignore
                rid = request.headers.get("x-trace-id") or str(uuid.uuid4())
                start = time.time()
                try:
                    response = await call_next(request)
                    # Tag slow requests
                    dur = (time.time() - start) * 1000.0
                    if dur > 1200:
                        try:
                            LOGGER.warning(
                                "slow_request",
                                extra={
                                    "path": request.url.path,
                                    "ms": round(dur, 2),
                                    "rid": rid,
                                },
                            )
                        except Exception:
                            pass  # logging meta-failure - nothing to fall back to
                    return response
                except Exception as e:  # noqa: BLE001
                    tb = traceback.format_exc(limit=6)
                    try:
                        LOGGER.error(
                            "unhandled_exception",
                            extra={
                                "path": request.url.path,
                                "error": str(e),
                                "rid": rid,
                                "trace": tb,
                            },
                        )
                    except Exception:
                        pass  # logging meta-failure - nothing to fall back to
                    from starlette.responses import JSONResponse

                    return JSONResponse(
                        {
                            "ok": False,
                            "error": str(e),
                            "rid": rid,
                            "trace_excerpt": tb.splitlines()[-5:],
                        },
                        status_code=500,
                    )

            _APP_INSTRUMENTED = True  # type: ignore
        except Exception as e:
            import logging as _inst_logging
            _inst_logging.getLogger("ghost").warning(f"app_instrumentation_failed: {e}")


    # ═══════════════════════════════════════════════════════════════════
    # PER-IP API THROTTLE — Protects DB from chatty browsers / duplicate tabs
    # Max 30 API requests per 10-second window per IP (3 req/s average).
    # Only throttles /api/ paths — static files, cockpit page, health unaffected.
    # ═══════════════════════════════════════════════════════════════════
    _THROTTLE_WINDOW = 10       # seconds
    _THROTTLE_MAX = 30          # max requests per window
    _throttle_buckets: dict = {}  # ip -> [count, window_start]

    @APP.middleware("http")
    async def api_throttle_middleware(request: Request, call_next):
        """Per-IP rate limiter for /api/ endpoints to prevent DB exhaustion."""
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        bucket = _throttle_buckets.get(client_ip)
        if bucket is None or (now - bucket[1]) > _THROTTLE_WINDOW:
            # New window
            _throttle_buckets[client_ip] = [1, now]
        else:
            bucket[0] += 1
            if bucket[0] > _THROTTLE_MAX:
                LOGGER.warning(f"[THROTTLE] {client_ip} exceeded {_THROTTLE_MAX} reqs/{_THROTTLE_WINDOW}s — throttled")
                return JSONResponse(
                    status_code=429,
                    content={"error": "too_many_requests", "retry_after": _THROTTLE_WINDOW}
                )

        # Evict stale IPs every ~100 requests to prevent memory leak
        if len(_throttle_buckets) > 100:
            cutoff = now - _THROTTLE_WINDOW * 3
            stale = [ip for ip, b in _throttle_buckets.items() if b[1] < cutoff]
            for ip in stale:
                del _throttle_buckets[ip]

        return await call_next(request)

    # Fast-fail auth middleware: return 401 JSON immediately if Bearer token missing
    @APP.middleware("http")
    async def auth_fast_fail_middleware(request: Request, call_next):
        """
        Return 401 JSON immediately on missing auth for protected endpoints.
    
        #57: Consolidated from 35+ individual if-statements into ONE set lookup.
        All read-only API endpoints are public. Write endpoints require Bearer token.
        """
        path = request.url.path
    
        # ── Exact-match public paths ──
        PUBLIC_EXACT = {
            "/", "/health", "/metrics", "/docs", "/redoc", "/openapi.json",
            "/api/status", "/api/health", "/api/openapi.json",
            "/api/predictions/multi/run", "/api/predictions/run",
            "/api/predictions/symbols", "/api/health/predictions",
            "/api/cockpit", "/api/recent_alerts", "/retrain-trigger",
        }
    
        if path in PUBLIC_EXACT:
            return await call_next(request)
    
        # ── Prefix-match public paths (read-only data feeds) ──
        # Consolidated from 35+ individual if-statements
        PUBLIC_PREFIXES = (
            "/api/system/", "/api/system_status",
            "/api/walk_forward_analysis/", "/api/monte_carlo/",
            "/api/momentum_shift/", "/api/research/", "/api/hedging/",
            "/api/predict/", "/api/price/",
            "/api/intel/", "/api/agentkit/",
            "/api/stage1/", "/api/stage2/", "/api/stage3/",
            "/api/stage4/", "/api/stage5/",
            "/api/runtime/", "/api/watcher/", "/api/crypto/",
            "/api/scan", "/api/opportunit", "/api/goals/",
            "/api/cockpit/", "/api/v3/", "/api/v4/", "/api/v2/",
            "/api/xrp/", "/api/presale/", "/api/config",
            "/api/corporate_actions", "/api/portfolio/",
            "/api/forecast/", "/api/movers/", "/api/gates/",
            "/api/money-game/",
            "/api/xray",
            "/api/doctor",
            "/alerts/",
            "/api/debug/crypto-check/",
            "/static/",
        )
    
        if path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)
    
        # ── Everything else under /api/ requires Bearer token ──
        if path.startswith("/api/"):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401,
                    content={"error": "unauthorized", "message": "Bearer token required"}
                )

        return await call_next(request)

    # IP Allowlisting Middleware
    if IP_ALLOWLIST_ENABLED:

        @APP.middleware("http")
        async def ip_allowlist_middleware(request: Request, call_next):
            """Restrict API access by IP address."""
            # Skip IP allowlist if not configured (local dev)
            if not IP_ALLOWLIST:
                return await call_next(request)

            client_ip = request.client.host if request.client else None

            # Allow health checks
            if request.url.path in ["/health", "/metrics"]:
                return await call_next(request)

            # Check IP allowlist
            if client_ip and client_ip not in IP_ALLOWLIST:
                return JSONResponse(
                    status_code=403, content={"error": "IP not allowed", "ip": client_ip}
                )

            return await call_next(request)


    # Optional security headers
    SECURE_HEADERS = os.getenv("SECURE_HEADERS", "1").lower() not in ("0", "false", "no")
    # CSP mode: dev (permissive) vs strict (production)
    CSP_MODE = os.getenv("CSP_MODE", "dev").strip().lower()
    APP_ENV = os.getenv("APP_ENV", os.getenv("ENV", "")).strip().lower()




    REFERRER_POLICY = os.getenv("REFERRER_POLICY", "no-referrer")
    HSTS_ON = os.getenv("HSTS_ON", "1").lower() not in ("0", "false", "no")
    HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "15552000"))  # 180 days


    @APP.middleware("http")
    async def _security_headers_mw(request: Request, call_next):  # type: ignore[override]
        response = await call_next(request)
        if not SECURE_HEADERS:
            return response
        try:
            response.headers.setdefault("X-Content-Type-Options", "nosnif")
            response.headers.setdefault("X-Frame-Options", "DENY")
            response.headers.setdefault("Referrer-Policy", REFERRER_POLICY)
            csp = _compute_csp()
            # Loosen CSP for UI pages only to allow inline <script> and inline handlers.
            # This preserves strict CSP for API responses while keeping the prebuilt UI functional.
            try:
                path = request.url.path or ""
                if (
                    path == "/"
                    or path.startswith("/ui")
                    or path.startswith("/assets")
                    or path.startswith("/static")
                    or path == "/index.html"
                    or path == "/cockpit"
                    or path == "/cockpit.html"
                ):
                    # Check if script-src specifically lacks 'unsafe-inline', not if it's anywhere in CSP
                    if "script-src" in csp and "script-src 'self' 'unsafe-inline'" not in csp and "script-src 'unsafe-inline'" not in csp:
                        csp = csp.replace("script-src ", "script-src 'unsafe-inline' ")
            except Exception as e:
                logging.getLogger("ghost").warning(f"Failed to adjust CSP for UI path: {e}")
            response.headers["Content-Security-Policy"] = csp
            if HSTS_ON and (
                request.url.scheme == "https" or os.getenv("FORCE_HSTS", "0") in ("1", "true")
            ):
                response.headers.setdefault(
                    "Strict-Transport-Security",
                    f"max-age={HSTS_MAX_AGE}; includeSubDomains; preload",
                )
        except Exception as e:
            logging.getLogger("ghost").warning(f"Failed to set security headers: {e}", exc_info=True)
        return response


    @APP.middleware("http")
    async def _trace_mw(request: Request, call_next):  # type: ignore[override]
        # Correlate requests with a lightweight trace id; if OTEL enabled, start span
        rid = (
            request.headers.get("X-Request-Id")
            or request.headers.get("X-Correlation-Id")
            or str(uuid.uuid4())
        )
        token_trace = _cv_trace_id.set(rid)
        token_path = _cv_path.set(request.url.path)
        token_method = _cv_method.set(request.method)
        if _OTEL_TRACER is not None:
            try:
                with _OTEL_TRACER.start_as_current_span(
                    f"HTTP {request.method} {request.url.path}"
                ) as span:  # type: ignore[attr-defined]
                    span.set_attribute("http.method", request.method)
                    span.set_attribute("http.target", request.url.path)
                    span.set_attribute("http.scheme", request.url.scheme)
                    response = await call_next(request)
                    span.set_attribute("http.status_code", response.status_code)
            finally:
                _cv_trace_id.reset(token_trace)
                _cv_path.reset(token_path)
                _cv_method.reset(token_method)
            response.headers.setdefault("X-Request-Id", rid)
            return response
        try:
            response = await call_next(request)
        finally:
            _cv_trace_id.reset(token_trace)
            _cv_path.reset(token_path)
            _cv_method.reset(token_method)
        response.headers.setdefault("X-Request-Id", rid)
        return response

    @APP.middleware("http")
    async def _rate_limit_mw(request: Request, call_next):
        # Disable limiter entirely in test mode
        if os.getenv("SNAP_TEST_MODE", "0").lower() in ("1", "true", "yes"):
            return await call_next(request)
        if RATE_LIMIT_WRITE_RPM <= 0:
            return await call_next(request)
        try:
            if request.method in ("POST", "PUT", "PATCH", "DELETE"):
                path = request.url.path or ""
                if path.startswith("/api") or path.startswith("/alerts"):
                    # Admin IP allowlist if configured
                    try:
                        if ADMIN_IP_ALLOWLIST:
                            client_ip = request.client.host if request.client else None
                            if client_ip and client_ip not in ADMIN_IP_ALLOWLIST:
                                return JSONResponse({"error": "forbidden"}, status_code=403)
                    except Exception:
                        pass
                    # Exempt valid bearer if configured
                    if RATE_LIMIT_EXEMPT_AUTH:
                        token = os.getenv("GHOST_API_TOKEN", "").strip()
                        if token:
                            auth = request.headers.get("authorization", "")
                            if (
                                auth.lower().startswith("bearer ")
                                and auth.split(" ", 1)[1].strip() == token
                            ):
                                return await call_next(request)
                    # Token bucket
                    global _RATE_TOKENS, _RATE_LAST_REFILL
                    now = time.monotonic()
                    rate_per_sec = _RATE_CAPACITY / 60.0 if _RATE_CAPACITY > 0 else 0.0
                    if _RATE_TOKENS < _RATE_CAPACITY and rate_per_sec > 0:
                        elapsed = max(0.0, now - _RATE_LAST_REFILL)
                        refill = elapsed * rate_per_sec
                        if refill >= 1.0:
                            _RATE_TOKENS = min(_RATE_CAPACITY, _RATE_TOKENS + int(refill))
                            _RATE_LAST_REFILL = now
                    if _RATE_TOKENS >= 1.0:
                        _RATE_TOKENS -= 1.0
                        try:
                            if _G_RATE_LIMIT_TOKENS is not None:
                                _G_RATE_LIMIT_TOKENS.set(_RATE_TOKENS)
                        except Exception:
                            pass
                        return await call_next(request)
                    else:
                        try:
                            if _C_RATE_LIMIT_DROPS is not None:
                                _C_RATE_LIMIT_DROPS.inc()
                        except Exception:
                            pass
                        # Estimate next token availability (~60/RPM seconds)
                        retry_after = max(1, int(round(60.0 / max(1, RATE_LIMIT_WRITE_RPM))))
                        resp = JSONResponse({"error": "rate-limited"}, status_code=429)
                        try:
                            resp.headers["Retry-After"] = str(retry_after)
                        except Exception:
                            pass
                        return resp
        except Exception:
            return await call_next(request)
        return await call_next(request)


    @APP.middleware("http")
    async def _log_requests(request, call_next):
        # Catch absolutely everything and always return a JSON response.
        # Add x-ghost-mw header to confirm middleware execution.
        from starlette.responses import JSONResponse
        try:
            response = await call_next(request)
            if response is None:
                LOGGER.error("call_next returned None for %s %s", request.method, request.url.path)
                resp = JSONResponse({"error": "internal_error", "detail": "no_response_returned"}, status_code=500)
                resp.headers["x-ghost-mw"] = "on"
                return resp
            # Add header to all responses
            try:
                response.headers["x-ghost-mw"] = "on"
            except Exception:
                pass
            return response
        except BaseException as e:  # includes Exception, CancelledError, etc.
            try:
                LOGGER.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=e)
            except Exception:
                pass  # logging should never crash the request path
            resp = JSONResponse({"error": "internal_error"}, status_code=500)
            resp.headers["x-ghost-mw"] = "on"
            return resp


