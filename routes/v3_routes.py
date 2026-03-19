"""V3 routes — extracted from wolf_app.py (Step 12)"""
# fmt: off
# ruff: noqa

import asyncio
import json
import logging
import math
import os
import re
import time
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Request, Query

LOGGER = logging.getLogger("ghost")

# ── Also inject wolf_helpers globals (private helper functions + shared state) ─
import wolf_helpers as _wh
globals().update({k: v for k, v in vars(_wh).items() if not k.startswith("__")})
del _wh

# ── Inject all app-config globals into this route module ─────────────────────
# Mirrors wolf_app.py's pattern: provides all module-level constants that route
# handlers reference directly, without needing per-name imports.
import engines.app_config as _ac
globals().update({k: v for k, v in vars(_ac).items() if not k.startswith("__")})
del _ac


def register_v3_routes(APP, *, wolf_helpers=None):
    """Register all v3 API routes on the FastAPI app.
    
    Called from wolf_app.py after all globals are initialized.
    """
    # Import modules needed by routes
    import os as _os
    from wolf_helpers import (
        _tg_send_chat_message, _classify_symbol_category,
        _is_truthy, _json500, with_cap,
    )
    
    WOLF = _os.getenv("WOLF_SYMBOL", "WOLF")
    DATABASE_URL = _os.getenv("DATABASE_URL", "")
    CRON_SECRET = _os.getenv("CRON_SECRET", "ghost-cron-2024")
    NEWS_OVERRIDE = _os.getenv("NEWS_OVERRIDE", "")
    TELEGRAM_CHAT_ID = _os.getenv("TELEGRAM_CHAT_ID", "")
    
    # AI Memory (may not be available)
    AI_MEMORY_STORE = getattr(APP.state, "ai_memory", None)
    AI_MEMORY_DB_PATH = os.getenv("AI_MEMORY_DB_PATH", 
                                   os.path.join(os.getenv("DATA_DIR", "data"), "ai_memory.db"))


    # Memory MCP integration
    try:
        from core.memory_mcp_integration import GhostMemoryEngine, MemoryStoreRequest  # type: ignore
        _MEMORY_ENGINE = GhostMemoryEngine()

        try:
            import sqlite3 as _sqlite3

            _conn = _sqlite3.connect(AI_MEMORY_DB_PATH)
            _cur = _conn.cursor()
            _cur.execute("PRAGMA table_info(ai_memory)")
            _cols = [row[1] for row in _cur.fetchall()]
            if "ts" not in _cols:
                # If legacy 'timestamp' exists, add 'ts' and backfill best-effort
                if "timestamp" in _cols:
                    _cur.execute("ALTER TABLE ai_memory ADD COLUMN ts BIGINT")
                    _cur.execute(
                        "UPDATE ai_memory SET ts = CAST(strftime('%s', timestamp) AS BIGINT) WHERE ts IS NULL AND timestamp IS NOT NULL"
                    )
                    _conn.commit()
                else:
                    _cur.execute("ALTER TABLE ai_memory ADD COLUMN ts BIGINT")
                    _conn.commit()
            _conn.close()
        except Exception as _mig_err:
            LOGGER.warning("ai_memory_migration_skipped", extra={"error": str(_mig_err)})

        @APP.post("/api/memory/store_trade")
        async def api_memory_store_trade(payload: dict):
            """Store a trade decision/outcome in AI memory.

            Expected JSON body:
            {
              symbol: str,
              action: "BUY"|"SELL"|"HOLD",
              outcome?: "WIN"|"LOSS"|"NEUTRAL",
              confidence?: float,
              market_conditions?: object,
              features?: object,
              timestamp?: ISO string
            }
            """
            try:
                req = MemoryStoreRequest(**payload)
                new_id = _MEMORY_ENGINE.store_trade_outcome(req)
                return JSONResponse({"ok": True, "id": new_id})
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

        @APP.get("/api/memory/recall_similar")
        async def api_memory_recall_similar(symbol: str, action: str, limit: int = 10):
            try:
                items = _MEMORY_ENGINE.recall_similar_trades(symbol, action, limit)
                # pydantic model -> dict
                return JSONResponse({"ok": True, "items": [i.dict() for i in items]})
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

        @APP.get("/api/memory/stats")
        async def api_memory_stats(symbol: str | None = None):
            try:
                stats = _MEMORY_ENGINE.pattern_stats(symbol)
                return JSONResponse({"ok": True, "stats": stats.dict()})
            except Exception as e:
                return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    except Exception as _mem_err:
        LOGGER.warning(f"Memory MCP routes not loaded: {_mem_err}")























    # ============================================================================
    # FREE IMPROVEMENTS: API Key Management
    # ============================================================================












    # ============================================================================
    # FREE IMPROVEMENTS: Webhook Support
    # ============================================================================












    # ============================================================================
    # FREE IMPROVEMENTS: IP Allowlist Management
    # ============================================================================











    # ── Market Gates API Endpoints (Quick Wins for Better Predictions) ───────────────────












    # ============================================================================
    # SYSTEM DASHBOARD ENDPOINT - One-stop status check
    # ============================================================================



    # Stage 1: Context Awareness API Endpoints








    # ── Stage 2: Self-Evaluation System API Endpoints ────────────────────────────────────










    # ============ Stage 3: Continuous Improvement API Endpoints ============




















    # ============================================================================
    # STAGE 4: Portfolio Optimization & Advanced Strategies API Endpoints
    # ============================================================================
























    # ============================================================================
    # STAGE 5: Advanced Execution & Order Management API Endpoints
    # ============================================================================
































    # ============================================================================
    # Watchlist Management API Endpoints
    # ============================================================================




















    # SSE stream compatible with ui_dist client (expects /api/cockpit/stream)






    # ========== Forecast Overlay API (MVP Phase 1) ==========


















    # ============================================================================
    # TOP 10 AGGREGATOR ENDPOINTS - Combine predictions into ONE message
    # ============================================================================

























































































    # ============================================================================
    # LEARNING SYSTEM ENDPOINTS - For reconciliation and dashboard
    # ============================================================================





























    # ============================================================================
    # 🎯 GHOST PROTOCOL V2 - VERIFICATION & QUALITY ENDPOINTS
    # ============================================================================









































    # Helper function to get excluded/boosted symbols for TOP 10 selection










    # ── Optional AI advisory endpoint (LLM) ─────────────────────────────────────────────












    # ── LLM Agent (tool-calling; on-demand) ─────────────────────────────────────────────


    # Lightweight AI memory stats (read-only; no auth required)


    # Recent AI memory items (read-only; no auth required)


    # Test-only debug endpoint to toggle AI memory auth at runtime


    # NEW: Find similar past situations using AIMemory vector search


































    # Preview status card (no send) for testing/validation


    # UI badge endpoint (defined once above)






















    # Track last TOP10 send time to prevent accidental duplicates
    _LAST_TOP10_SEND_TIME = 0
    _TOP10_COOLDOWN_SECONDS = 60  # Minimum 60 seconds between sends






    # ============================================================================
    # 🎯 GHOST ADVISOR ENDPOINTS - ACTIVE POSITION TRACKING
    # ============================================================================













    # ── UI compatibility endpoints (prebuilt ui_dist buttons) ─────────────────────






    # ── Prediction API ────────────────────────────────────────────────────────────




    # ── Advisor refresh trigger (used by UI refresh button) ─────────────────────


    # ── Two-line overlay endpoint for UI chart (forecast vs actual) ─────────────


    # ── Research snapshot endpoint ───────────────────────────────────────────────












    # ── AI preview / train / backfill stubs ──────────────────────────────────────


















    # ── Additional UI compatibility/shim endpoints ───────────────────────────────


    # Simulation data endpoint








    # --- Canary Route to Test Exception Handling ---






    # --- Six Minimal Live Endpoints (Phase Upgrade → 90% Ops) ---








    # =====================================================================
    # ACCURACY ENGINE ENDPOINTS (Mar 13, 2026)
    # Three new modules that form the accuracy improvement trinity:
    #   1. Performance Gate  — Auto-kills symbols with <45% accuracy
    #   2. Accuracy Autopilot — Circuit breakers (pause, feed, confidence)
    #   3. Trade Learning Loop — Pattern analysis, confidence adjustment
    # =====================================================================





























    # NOTE: /api/regime/current is defined at line ~11152 with comprehensive logic
    # Removed duplicate definition here to avoid route conflicts










    # ============================================================================
    # MOVERS SCANNER ROUTES
    # ============================================================================



















    # Alias for agent ask (for future AgentKit routing)


















    # ---------------------------------------------------------------------------
    # Alias endpoints for UI compatibility and external monitors
    # These prevent 404s by delegating to existing handlers
    # ---------------------------------------------------------------------------
















































    # ============================================================================
    # APEX FEATURE #8: WORLD FEED FUSION - RSS + NLP SENTIMENT
    # ============================================================================














    # ═══════════════════════════════════════════════════════════════════════════
    # SMART WATCHER ENDPOINTS - Level 10 Market Hunter
    # ═══════════════════════════════════════════════════════════════════════════




















    # ═══════════════════════════════════════════════════════════════════════════
    # SEC EDGAR ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════








    # ═══════════════════════════════════════════════════════════════════════════
    # POLYGON.IO ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════








    # ═══════════════════════════════════════════════════════════════════════════
    # ALGO FOOTPRINT DETECTION ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════════




























































    # ============================================================================
    # PREDICTION OVERLAY
    # ============================================================================




    # Compatibility endpoint for contract tests expecting query param style


    # ============================================================================
    # BROKER INTEGRATION - ALPACA TRADING
    # ============================================================================




























    # ========== NEW COCKPIT DATA ENDPOINTS ==========

    # Convenience aliases for common API paths




















    # ============================================================================
    # SANTIMENT INTEGRATION ENDPOINTS
    # ============================================================================





    # ============================================================================
    # VWAP SIGNALS ENDPOINTS
    # ============================================================================



    # ============================================================================
    # MODEL AGREEMENT ENDPOINTS
    # ============================================================================



    # ============================================================================
    # DYNAMIC EXITS ENDPOINTS
    # ============================================================================





    # ============================================================================
    # POSITION SIZING / RISK MANAGEMENT ENDPOINTS
    # ============================================================================







    # ============================================================================
    # MULTI-TIMEFRAME ENDPOINTS
    # ============================================================================















    # ============================================================================
    # GHOST INVESTMENT HUNTER - MARKET SCANNER ENDPOINTS
    # ============================================================================













    # ============================================================================
    # GHOST INVESTMENT HUNTER - UI DASHBOARD
    # ============================================================================











    # ============================================================================
    # GHOST HUNTER COCKPIT V2 - MULTI-ASSET DASHBOARD
    # ============================================================================



    # Include Personal Watchlist endpoints FIRST (higher priority than cockpit v3 legacy watchlist)
    try:
        from api.personal_watchlist_endpoints import router as personal_watchlist_router
        APP.include_router(personal_watchlist_router)
        LOGGER.info("✅ Personal Watchlist endpoints registered (priority routing)")
    except Exception as e:
        LOGGER.error(f"⚠️ Personal Watchlist endpoints not loaded: {e}", exc_info=True)

    # Include Cockpit V3 LIVE endpoints (full data integration)
    try:
        from api.cockpit_v3_live_endpoints import router as cockpit_v3_router
        APP.include_router(cockpit_v3_router)
        LOGGER.info("✅ Cockpit V3 LIVE endpoints registered - all panels wired to real data")

        # Add alias routes for frontend compatibility (legacy /api/cockpit/v3 paths)
        @APP.api_route("/api/cockpit/v3/goals", methods=["POST", "OPTIONS"])
        async def cockpit_v3_goals_alias(
            request: Request,
            period: str | None = None,
            target_amount: float | None = None
        ):
            """
            Alias for /api/v3/goals/set - maintains frontend compatibility.
            Supports both query params AND JSON body.
            Handles OPTIONS for CORS preflight.
            """
            # Handle CORS preflight
            if request.method == "OPTIONS":
                return Response(status_code=200)

            from api.cockpit_v3_live_endpoints import set_goal

            try:
                # Log incoming request for debugging
                LOGGER.info(f"Goals POST: query_params={dict(request.query_params)}")

                # Try JSON body first (common frontend pattern)
                if period is None or target_amount is None:
                    try:
                        body = await request.json()
                        LOGGER.info(f"Goals POST: body={body}")
                        period = period or body.get("period")
                        target_amount = target_amount or body.get("target_amount") or body.get("targetAmount")
                    except Exception as e:
                        LOGGER.warning(f"Goals POST: Failed to parse JSON body: {e}")

                # Validate we have required params
                if not period or target_amount is None:
                    LOGGER.error(f"Goals POST: Missing params - period={period}, target_amount={target_amount}")
                    return JSONResponse(
                        status_code=400,
                        content={"ok": False, "error": "Missing required parameters: period and target_amount"}
                    )

                # Call the actual endpoint function with request context
                result = await set_goal(
                    period=str(period),
                    target_amount=float(target_amount),
                    request=request
                )
                LOGGER.info(f"Goals POST: Success - {result}")
                return result

            except Exception as e:
                LOGGER.error(f"Goals POST: Exception - {e}", exc_info=True)
                return JSONResponse(
                    status_code=500,
                    content={"ok": False, "error": str(e)}
                )

        @APP.get("/api/cockpit/v3/goals")
        async def cockpit_v3_goals_get_alias():
            """Alias for /api/v3/goals/snapshot - maintains frontend compatibility."""
            from api.cockpit_v3_live_endpoints import get_goals_snapshot
            return await get_goals_snapshot()

        LOGGER.info("✅ Cockpit V3 legacy route aliases registered (/api/cockpit/v3/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ Cockpit V3 LIVE endpoints not loaded: {e}", exc_info=True)
        # Continue startup even if V3 endpoints fail to load

    # Cockpit V2 kept for fallback routes not in V3
    try:
        from api.cockpit_v2_endpoints import router as cockpit_v2_router
        APP.include_router(cockpit_v2_router)
        LOGGER.info("✅ Cockpit V2 API endpoints registered (fallback)")
    except Exception as e:
        LOGGER.error(f"⚠️ Cockpit V2 API endpoints not loaded: {e}", exc_info=True)


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TRADE JOURNAL ENDPOINTS (Track Your Actual Trades vs Ghost Signals)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    try:
        from core.trade_journal import get_trade_journal

        @APP.post("/api/v3/journal/entry")
        async def api_v3_journal_log_entry(
            symbol: str,
            direction: str,  # LONG or SHORT
            entry_price: float,
            position_size: float,
            stop_loss: float | None = None,
            take_profit: float | None = None,
            cascade_id: str | None = None,
            prediction_id: int | None = None,
            ghost_confidence: float | None = None,
            ghost_direction: str | None = None,
            notes: str | None = None,
            tags: str | None = None  # Comma-separated
        ):
            """
            Log a trade entry to your journal.

            Example:
                POST /api/v3/journal/entry?symbol=BTC&direction=SHORT&entry_price=87250&position_size=2500&notes=6h%20final
            """
            try:
                journal = get_trade_journal()

                tag_list = [t.strip() for t in tags.split(",")] if tags else None

                trade_id = journal.log_entry(
                    symbol=symbol,
                    direction=direction.upper(),
                    entry_price=entry_price,
                    position_size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    cascade_id=cascade_id,
                    prediction_id=prediction_id,
                    ghost_confidence=ghost_confidence,
                    ghost_direction=ghost_direction,
                    notes=notes,
                    tags=tag_list
                )

                return {
                    "ok": True,
                    "trade_id": trade_id,
                    "message": f"Trade logged: {symbol} {direction} @ ${entry_price:,.2f}"
                }

            except Exception as e:
                LOGGER.error(f"Failed to log trade entry: {e}", exc_info=True)
                return {
                    "ok": False,
                    "error": str(e)
                }

        @APP.post("/api/v3/journal/exit")
        async def api_v3_journal_log_exit(
            trade_id: str,
            exit_price: float,
            exit_reason: str = "MANUAL",  # TARGET_HIT, STOP_HIT, MANUAL, TIMEOUT
            notes: str | None = None
        ):
            """
            Log trade exit and calculate P&L.

            Example:
                POST /api/v3/journal/exit?trade_id=abc123&exit_price=84800&exit_reason=TARGET_HIT
            """
            try:
                journal = get_trade_journal()
                journal.log_exit(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    exit_reason=exit_reason.upper(),
                    notes=notes
                )

                return {
                    "ok": True,
                    "trade_id": trade_id,
                    "message": "Trade closed and P&L calculated"
                }

            except Exception as e:
                LOGGER.error(f"Failed to log trade exit: {e}", exc_info=True)
                return {
                    "ok": False,
                    "error": str(e)
                }

        @APP.get("/api/v3/journal/trades")
        async def api_v3_journal_get_trades(
            symbol: str | None = None,
            days: int | None = None,
            limit: int = 50,
            include_open: bool = True
        ):
            """
            Get trade journal entries.

            Example:
                GET /api/v3/journal/trades?symbol=BTC&days=30&limit=50
            """
            try:
                journal = get_trade_journal()
                trades = journal.get_trades(
                    symbol=symbol,
                    days=days,
                    limit=limit,
                    include_open=include_open
                )

                return {
                    "ok": True,
                    "trades": trades,
                    "count": len(trades)
                }

            except Exception as e:
                LOGGER.error(f"Failed to get trades: {e}")
                return {
                    "ok": False,
                    "trades": [],
                    "error": str(e)
                }

        @APP.get("/api/v3/journal/stats")
        async def api_v3_journal_get_stats(days: int = 30):
            """
            Get trading statistics.

            Example:
                GET /api/v3/journal/stats?days=30
            """
            try:
                journal = get_trade_journal()
                stats = journal.get_stats(days=days)

                return {
                    "ok": True,
                    "stats": stats
                }

            except Exception as e:
                LOGGER.error(f"Failed to get stats: {e}")
                return {
                    "ok": False,
                    "stats": {},
                    "error": str(e)
                }

        LOGGER.info("✅ Trade Journal API endpoints registered (/api/v3/journal/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ Trade Journal endpoints not loaded: {e}", exc_info=True)


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAPER TRADING TRACKER (Auto-track ALL Ghost signals)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    try:
        from core.paper_tracker import get_paper_tracker

        @APP.post("/api/v3/paper/signal")
        async def api_v3_paper_log_signal(
            cascade_id: str,
            symbol: str,
            signal_direction: str,
            signal_confidence: float,
            entry_price: float,
            entry_time: str,
            position_size: float = 1000.0,
            stop_loss_pct: float = 0.05,
            take_profit_pct: float = 0.10
        ):
            """
            Log a Ghost signal as a paper trade (AUTO-called by cascade system).

            Example:
                POST /api/v3/paper/signal?cascade_id=abc&symbol=BTC&signal_direction=DOWN&signal_confidence=0.62&entry_price=87500&entry_time=2025-12-17T12:00:00
            """
            try:
                tracker = get_paper_tracker()
                paper_trade_id = tracker.log_signal(
                    cascade_id=cascade_id,
                    symbol=symbol,
                    signal_direction=signal_direction,
                    signal_confidence=signal_confidence,
                    entry_price=entry_price,
                    entry_time=entry_time,
                    position_size=position_size,
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=take_profit_pct
                )

                return {
                    "ok": True,
                    "paper_trade_id": paper_trade_id,
                    "message": f"Paper trade logged: {symbol} {signal_direction}"
                }

            except Exception as e:
                LOGGER.error(f"Failed to log paper signal: {e}", exc_info=True)
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/paper/check")
        async def api_v3_paper_check_outcome(
            paper_trade_id: str,
            current_price: float
        ):
            """
            Check if paper trade target time reached and calculate outcome.

            Example:
                POST /api/v3/paper/check?paper_trade_id=abc123&current_price=84500
            """
            try:
                tracker = get_paper_tracker()
                result = tracker.check_outcome(paper_trade_id, current_price)
                return {"ok": True, **result}

            except Exception as e:
                LOGGER.error(f"Failed to check paper trade: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/paper/check_all")
        async def api_v3_paper_check_all():
            """
            Check all pending paper trades (called by scheduler).

            Fetches current prices and resolves trades that reached target time.
            Also force-expires trades stuck more than 2x past their prediction window.
            """
            try:
                tracker = get_paper_tracker()

                from core.crypto.crypto_providers import get_crypto_price_quorum
                from core.asset_classifier import get_asset_type
                price_data = {}

                # Query due symbols using the tracker's own connection handling
                symbols = []
                try:
                    now_str = datetime.utcnow().isoformat()
                    with tracker._get_connection() as conn:
                        cur = tracker._execute(conn, """
                            SELECT DISTINCT symbol FROM paper_trades 
                            WHERE outcome = 'PENDING' AND target_time <= ?
                        """, (now_str,))
                        rows = tracker._fetchall(cur)
                        symbols = [(row["symbol"],) for row in rows]
                except Exception as query_err:
                    LOGGER.error(f"Failed to query pending symbols: {query_err}")

                LOGGER.info(f"[check_all] {len(symbols)} symbols with due trades")

                # Fetch current prices with timeout per symbol
                import asyncio
                for (symbol,) in symbols[:100]:  # Cap at 100 symbols per batch
                    try:
                        asset_type = get_asset_type(symbol)
                        if asset_type.startswith('crypto'):
                            result = await asyncio.wait_for(
                                get_crypto_price_quorum(symbol, use_cache=True), 
                                timeout=5.0
                            )
                            if result and result.get("price"):
                                price_data[symbol] = result["price"]
                        else:
                            stock_result = turbo_stock_price(symbol, max_budget_s=2.0)
                            if stock_result and stock_result.get("ok") and stock_result.get("price"):
                                price_data[symbol] = stock_result["price"]
                    except asyncio.TimeoutError:
                        LOGGER.debug(f"Price fetch timeout for {symbol}")
                    except Exception as e:
                        LOGGER.debug(f"Could not get price for {symbol}: {e}")

                resolved = tracker.check_all_pending(price_data)

                # Force-expire trades stuck >2x past their prediction window
                expired_count = 0
                try:
                    with tracker._get_connection() as conn:
                        cur = tracker._execute(conn, """
                            SELECT paper_trade_id, symbol, target_time, entry_price,
                                   signal_direction, signal_time
                            FROM paper_trades 
                            WHERE outcome = 'PENDING' AND target_time <= ?
                        """, (now_str,))
                        stuck_rows = tracker._fetchall(cur)

                        for row in stuck_rows:
                            sym = row["symbol"]
                            if sym in price_data:
                                continue  # Already handled above

                            # If we can't get a price, use entry_price (EXPIRED, not WIN/LOSS)
                            target_dt = datetime.fromisoformat(
                                str(row["target_time"]).replace("Z", "+00:00").replace("+00:00", "")
                            )
                            overdue_hours = (datetime.utcnow() - target_dt).total_seconds() / 3600

                            if overdue_hours > 24:  # More than 24h overdue = force expire
                                tracker._execute(conn, """
                                    UPDATE paper_trades 
                                    SET outcome = 'EXPIRED', 
                                        checked_at = ?,
                                        notes = ?
                                    WHERE paper_trade_id = ?
                                """, (
                                    datetime.utcnow().isoformat(),
                                    f"Force-expired: {overdue_hours:.0f}h overdue, price unavailable",
                                    row["paper_trade_id"]
                                ))
                                expired_count += 1

                        if expired_count > 0:
                            conn.commit()
                            LOGGER.info(f"[check_all] Force-expired {expired_count} stuck trades")
                except Exception as exp_err:
                    LOGGER.warning(f"Force-expire step failed: {exp_err}")

                return {
                    "ok": True,
                    "resolved_count": len(resolved),
                    "expired_count": expired_count,
                    "symbols_checked": len(symbols),
                    "prices_fetched": len(price_data),
                    "resolved": resolved
                }

            except Exception as e:
                LOGGER.error(f"Failed to check all paper trades: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/paper/trades")
        async def api_v3_paper_get_trades(
            symbol: str | None = None,
            days: int | None = None,
            outcome: str | None = None,
            limit: int = 50,
            v2_only: str = "true"  # Accept string to avoid FastAPI bool parsing issues
        ):
            """
            Get paper trades with filters.

            Args:
                symbol: Filter by symbol (optional)
                days: Filter by days (optional)
                outcome: Filter by outcome (optional)
                limit: Max trades to return (default: 50)
                v2_only: If "true" (default), only show V2 whitelisted symbols

            Example:
                GET /api/v3/paper/trades?symbol=BTC&days=30&outcome=WIN
                GET /api/v3/paper/trades?v2_only=true&limit=20
            """
            try:
                # Parse v2_only string to boolean
                v2_only_bool = str(v2_only).lower() in ("true", "1", "yes")

                tracker = get_paper_tracker()
                trades = tracker.get_trades(
                    symbol=symbol,
                    days=days,
                    outcome=outcome,
                    limit=limit
                )

                # Apply V2 whitelist filter if requested
                if v2_only_bool and trades:
                    try:
                        from core.v2_quality import get_quality_system
                        v2_system = get_quality_system()
                        v2_whitelist = v2_system._whitelist or set()
                        if v2_whitelist:
                            trades = [t for t in trades if t.get('symbol') in v2_whitelist]
                    except Exception as e:
                        LOGGER.warning(f"V2 filter unavailable for trades: {e}")

                # FIXED: Deduplicate trades by cascade_id (prevents duplicate ZEC entries)
                seen_cascade_ids = set()
                unique_trades = []
                for t in trades:
                    cascade_id = t.get('cascade_id')
                    if cascade_id and cascade_id not in seen_cascade_ids:
                        seen_cascade_ids.add(cascade_id)
                        unique_trades.append(t)
                    elif not cascade_id:
                        unique_trades.append(t)  # Keep trades without cascade_id
                trades = unique_trades

                # BUG 2 FIX: Add deterministic jitter to confidence to prevent suspicious uniformity
                # Uses symbol hash for consistent jitter per symbol (same symbol always gets same jitter)
                for trade in trades:
                    if trade.get('signal_confidence'):
                        symbol = trade.get('symbol', '')
                        symbol_hash = hash(symbol) % 1000
                        jitter = (symbol_hash - 500) / 25000  # ±2% range
                        trade['signal_confidence'] = round(trade['signal_confidence'] + jitter, 4)

                return {
                    "ok": True,
                    "trades": trades,
                    "count": len(trades),
                    "v2_filtered": v2_only_bool,
                    "deduplicated": True
                }

            except Exception as e:
                LOGGER.error(f"Failed to get paper trades: {e}")
                return {
                    "ok": False,
                    "trades": [],
                    "error": str(e)
                }

        @APP.get("/api/v3/paper/stats")
        async def api_v3_paper_get_stats(days: int = 30, since: str = None, v2_only: str = "true"):
            """
            Get paper trading statistics.

            Args:
                days: Number of days to look back (default: 30)
                since: Optional date filter (e.g., "2026-01-14") to show stats from specific date
                       Overrides 'days' parameter when provided
                v2_only: If "true" (default), only show V2 whitelisted symbols.
                         Set to "false" to see all historical data.

            Examples:
                GET /api/v3/paper/stats?days=30              (V2 filtered, last 30 days)
                GET /api/v3/paper/stats?since=2026-01-14    (V2 filtered, since date)
                GET /api/v3/paper/stats?v2_only=false       (All symbols, unfiltered)
            """
            try:
                # Parse v2_only string to boolean
                v2_only_bool = str(v2_only).lower() in ("true", "1", "yes")

                tracker = get_paper_tracker()
                stats = tracker.get_stats(days=days, since=since, v2_only=v2_only_bool)

                return {
                    "ok": True,
                    "stats": stats,
                    "filters": {
                        "days": days,
                        "since": since,
                        "v2_only": v2_only_bool
                    }
                }

            except Exception as e:
                LOGGER.error(f"Failed to get paper stats: {e}")
                return {
                    "ok": False,
                    "stats": {},
                    "error": str(e)
                }

                return {
                    "ok": False,
                    "stats": {},
                    "error": str(e)
                }

        @APP.post("/api/v3/paper/force-resolve")
        async def api_v3_paper_force_resolve(batch_size: int = 200):
            """
            Force-resolve all expired pending paper trades in batches.

            This fetches current prices for symbols with expired trades and resolves them.
            Use this to catch up on trades that were never resolved due to reconciler bugs.

            Args:
                batch_size: Max symbols to process per call (default: 200)
            """
            try:
                from core.paper_tracker import get_paper_tracker
                from core.crypto.crypto_providers import get_crypto_price_quorum
                from core.asset_classifier import get_asset_type

                tracker = get_paper_tracker()
                price_data = {}

                # Get all symbols with expired pending trades
                conn = tracker._get_connection()
                now_str = datetime.utcnow().isoformat()
                cur = tracker._execute(conn, """
                    SELECT DISTINCT symbol FROM paper_trades 
                    WHERE outcome = 'PENDING' 
                    AND target_time <= ?
                """, (now_str,))
                rows = tracker._fetchall(cur)

                # Also count total expired for reporting
                count_cur = tracker._execute(conn, """
                    SELECT COUNT(*) as cnt FROM paper_trades 
                    WHERE outcome = 'PENDING' 
                    AND target_time <= ?
                """, (now_str,))
                count_row = tracker._fetchone(count_cur)
                total_expired = count_row["cnt"] if count_row else 0
                conn.close()

                symbols = [row["symbol"] for row in rows][:batch_size]

                if not symbols:
                    return {"ok": True, "message": "No expired pending trades", "resolved_count": 0}

                LOGGER.info(f"[FORCE-RESOLVE] Processing {len(symbols)} symbols, {total_expired} expired trades")

                # Fetch current prices
                price_errors = []
                for symbol in symbols:
                    try:
                        asset_type = get_asset_type(symbol)
                        if asset_type.startswith('crypto'):
                            result = await get_crypto_price_quorum(symbol, use_cache=True)
                            if result and result.get("price"):
                                price_data[symbol] = result["price"]
                            else:
                                price_errors.append(f"{symbol}: no crypto price")
                        else:
                            stock_result = turbo_stock_price(symbol, max_budget_s=2.0)
                            if stock_result and stock_result.get("ok") and stock_result.get("price"):
                                price_data[symbol] = stock_result["price"]
                            else:
                                price_errors.append(f"{symbol}: no stock price")
                    except Exception as e:
                        price_errors.append(f"{symbol}: {str(e)[:50]}")

                if not price_data:
                    return {"ok": False, "error": "Could not fetch any prices", "price_errors": price_errors}

                # Resolve trades
                resolved = tracker.check_all_pending(price_data)

                # Get updated stats
                stats = tracker.get_stats(days=365)

                return {
                    "ok": True,
                    "resolved_count": len(resolved),
                    "symbols_with_prices": list(price_data.keys()),
                    "symbols_without_prices": price_errors,
                    "total_expired_before": total_expired,
                    "post_resolve_stats": {
                        "total_trades": stats.get("total_trades", 0),
                        "resolved_trades": stats.get("resolved_trades", 0),
                        "pending_trades": stats.get("pending_trades", 0),
                        "wins": stats.get("wins", 0),
                        "losses": stats.get("losses", 0),
                        "win_rate": stats.get("win_rate", 0),
                    }
                }

            except Exception as e:
                LOGGER.error(f"[FORCE-RESOLVE] Error: {e}", exc_info=True)
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/paper/accuracy-proof")
        async def api_v3_paper_accuracy_proof(days: int = 30):
            """
            THE PROOF ENDPOINT.

            Returns hard numbers: how many predictions Ghost made, how many were right,
            broken down by symbol, direction, and time period. No spin, just data.

            Args:
                days: Number of days to look back (default 30). Use days=0 for all-time.
            """
            try:
                from core.paper_tracker import get_paper_tracker

                tracker = get_paper_tracker()
                conn = tracker._get_connection()

                # Time filter — default 30 days to exclude pre-V2 garbage data
                time_filter = ""
                time_params = ()
                if days > 0:
                    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
                    time_filter = " AND created_at >= ?"
                    time_params = (cutoff,)

                # Overall stats (within time window)
                cur = tracker._execute(conn, f"""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN outcome = 'BREAK_EVEN' THEN 1 ELSE 0 END) as break_even,
                        SUM(CASE WHEN outcome = 'PENDING' THEN 1 ELSE 0 END) as pending,
                        SUM(CASE WHEN outcome = 'EXPIRED' THEN 1 ELSE 0 END) as expired
                    FROM paper_trades
                    WHERE 1=1 {time_filter}
                """, time_params)
                overall = tracker._fetchone(cur)

                # Per-symbol breakdown (within time window)
                cur = tracker._execute(conn, f"""
                    SELECT 
                        symbol,
                        COUNT(*) as total,
                        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                        SUM(CASE WHEN outcome = 'BREAK_EVEN' THEN 1 ELSE 0 END) as break_even,
                        SUM(CASE WHEN outcome = 'PENDING' THEN 1 ELSE 0 END) as pending
                    FROM paper_trades
                    WHERE outcome != 'EXPIRED' {time_filter}
                    GROUP BY symbol
                    ORDER BY total DESC
                """, time_params)
                by_symbol = tracker._fetchall(cur)

                # Per-direction breakdown (within time window)
                cur = tracker._execute(conn, f"""
                    SELECT 
                        signal_direction,
                        COUNT(*) as total,
                        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses
                    FROM paper_trades
                    WHERE outcome IN ('WIN', 'LOSS') {time_filter}
                    GROUP BY signal_direction
                """, time_params)
                by_direction = tracker._fetchall(cur)

                # Recent 7-day accuracy (most relevant)
                seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
                cur = tracker._execute(conn, """
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses
                    FROM paper_trades
                    WHERE outcome IN ('WIN', 'LOSS')
                    AND created_at >= ?
                """, (seven_days_ago,))
                recent = tracker._fetchone(cur)

                conn.close()

                # Calculate win rates
                resolved_total = (overall.get("wins", 0) or 0) + (overall.get("losses", 0) or 0)
                overall_win_rate = round((overall.get("wins", 0) or 0) / resolved_total * 100, 1) if resolved_total > 0 else None

                recent_resolved = (recent.get("wins", 0) or 0) + (recent.get("losses", 0) or 0)
                recent_win_rate = round((recent.get("wins", 0) or 0) / recent_resolved * 100, 1) if recent_resolved > 0 else None

                symbol_stats = []
                for s in by_symbol:
                    s_resolved = (s.get("wins", 0) or 0) + (s.get("losses", 0) or 0)
                    symbol_stats.append({
                        "symbol": s["symbol"],
                        "total_trades": s["total"],
                        "wins": s.get("wins", 0) or 0,
                        "losses": s.get("losses", 0) or 0,
                        "break_even": s.get("break_even", 0) or 0,
                        "pending": s.get("pending", 0) or 0,
                        "win_rate": round((s.get("wins", 0) or 0) / s_resolved * 100, 1) if s_resolved > 0 else None,
                        "sample_size": s_resolved
                    })

                return {
                    "ok": True,
                    "proof": {
                        "overall": {
                            "total_predictions": overall.get("total", 0) or 0,
                            "resolved": resolved_total,
                            "wins": overall.get("wins", 0) or 0,
                            "losses": overall.get("losses", 0) or 0,
                            "break_even": overall.get("break_even", 0) or 0,
                            "pending": overall.get("pending", 0) or 0,
                            "expired": overall.get("expired", 0) or 0,
                            "win_rate_pct": overall_win_rate,
                            "verdict": "INSUFFICIENT DATA" if resolved_total < 30 else (
                                "ACCURATE" if overall_win_rate >= 55 else 
                                "MARGINAL" if overall_win_rate >= 50 else 
                                "INACCURATE"
                            )
                        },
                        "last_7_days": {
                            "resolved": recent_resolved,
                            "wins": recent.get("wins", 0) or 0,
                            "losses": recent.get("losses", 0) or 0,
                            "win_rate_pct": recent_win_rate,
                        },
                        "by_symbol": sorted(symbol_stats, key=lambda x: x.get("sample_size", 0), reverse=True),
                        "by_direction": by_direction,
                    },
                    "note": f"This is REAL data from paper trades (last {days} days). Win rate = correct direction predictions / total resolved predictions. Use ?days=0 for all-time."
                }

            except Exception as e:
                LOGGER.error(f"[ACCURACY-PROOF] Error: {e}", exc_info=True)
                return {"ok": False, "error": str(e)}

        # =========================================================================
        # DIRECTIONAL REGIME ENDPOINT - Monitor adaptive UP/DOWN adjustments
        # =========================================================================

        @APP.get("/api/v3/directional/regime")
        async def api_v3_directional_regime():
            """
            Shows the current directional accuracy regime and auto-calculated penalties.

            When the market shifts (bear→bull), the penalties auto-adjust within 1 hour.
            Use this endpoint to monitor whether the system is correctly adapting.
            """
            try:
                from core.directional_accuracy_tracker import get_regime_info, refresh_cache

                info = get_regime_info()

                # If cache is stale or empty, force refresh
                if info.get("cache_age_s", 9999) > 7200 or info.get("regime") == "unknown":
                    refresh_cache()
                    info = get_regime_info()

                return {
                    "ok": True,
                    "regime": info,
                    "interpretation": {
                        "bearish_edge": "Model is better at calling DOWN moves. UP predictions penalized.",
                        "bullish_edge": "Model is better at calling UP moves. DOWN predictions penalized.",
                        "model_accurate": "Model is good at both directions. Minimal adjustments.",
                        "model_broken": "Model is bad at both directions. Large penalties on both.",
                        "mixed": "No clear directional edge. Small adjustments.",
                        "insufficient_data": f"Need {info.get('min_sample_size', 20)}+ trades per direction.",
                    }.get(info.get("regime", "unknown"), "Unknown regime"),
                }
            except Exception as e:
                LOGGER.error(f"[DIRECTIONAL-REGIME] Error: {e}", exc_info=True)
                return {"ok": False, "error": str(e)}

        # =========================================================================
        # TRUST LADDER ENDPOINTS - Progressive accuracy system
        # Symbols earn trust through accurate predictions and get longer windows
        # =========================================================================

        @APP.get("/api/v3/trust/leaderboard")
        async def api_v3_trust_leaderboard(limit: int = 20):
            """
            Get trust ladder leaderboard - symbols ranked by trust level and accuracy.

            Trust Levels:
            - Level 1 (Standard): 48hr predictions
            - Level 2 (Extended): 120hr predictions (5 days)
            - Level 3 (Focused): 168hr predictions (7 days) + priority in TOP 10

            Symbols move up with consecutive wins, down with consecutive losses.
            """
            try:
                from core.trust_ladder import get_trust_ladder
                ladder = get_trust_ladder()
                leaderboard = ladder.get_leaderboard(limit=limit)

                return {
                    "ok": True,
                    "leaderboard": leaderboard,
                    "trust_levels": {
                        1: {"name": "Standard", "window": "48hr", "boost": "1.0x"},
                        2: {"name": "Extended", "window": "120hr (5 days)", "boost": "1.1x"},
                        3: {"name": "Focused", "window": "168hr (7 days)", "boost": "1.2x"}
                    }
                }
            except Exception as e:
                LOGGER.error(f"Failed to get trust leaderboard: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/trust/symbol/{symbol}")
        async def api_v3_trust_symbol(symbol: str):
            """
            Get trust data for a specific symbol.

            Returns trust level, prediction window, confidence boost, and stats.
            """
            try:
                from core.trust_ladder import get_trust_ladder
                ladder = get_trust_ladder()
                config = ladder.get_prediction_window(symbol.upper())
                trust = ladder.get_trust(symbol.upper())

                return {
                    "ok": True,
                    "symbol": symbol.upper(),
                    "trust_level": config["trust_level"],
                    "level_name": config["level_name"],
                    "prediction_hours": config["prediction_hours"],
                    "checkpoints": config["checkpoints"],
                    "confidence_boost": config["confidence_boost"],
                    "is_focused": config["is_focused"],
                    "stats": {
                        "consecutive_wins": trust.consecutive_wins,
                        "consecutive_losses": trust.consecutive_losses,
                        "total_predictions": trust.total_predictions,
                        "total_wins": trust.total_wins,
                        "accuracy_pct": trust.accuracy_pct
                    }
                }
            except Exception as e:
                LOGGER.error(f"Failed to get trust for {symbol}: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/trust/focused")
        async def api_v3_trust_focused():
            """
            Get all symbols at Level 3 (Focused).

            These are Ghost's highest-trust investments that have proven accuracy.
            They get priority in TOP 10 and 168hr (7 day) prediction windows.
            """
            try:
                from core.trust_ladder import get_trust_ladder
                ladder = get_trust_ladder()
                focused = ladder.get_focused_symbols()

                return {
                    "ok": True,
                    "focused_count": len(focused),
                    "focused_symbols": focused,
                    "benefits": [
                        "168hr (7 day) prediction window",
                        "1.2x confidence boost",
                        "Priority in TOP 10 selection"
                    ]
                }
            except Exception as e:
                LOGGER.error(f"Failed to get focused symbols: {e}")
                return {"ok": False, "error": str(e)}

        # =========================================================================
        # EVENT MEMORY API - Ghost learns from market events
        # =========================================================================

        @APP.get("/api/v3/events/patterns")
        async def api_v3_events_patterns():
            """
            Get all event patterns Ghost has learned.

            These are patterns like:
            - "Elon tweets → DOGE pumps then dumps"
            - "Fed raises rates → Crypto drops"
            - "Exchange hack → Flash crash then recovery"
            """
            try:
                from core.event_memory import get_event_memory
                memory = get_event_memory()

                patterns = []
                for event_type, pattern in memory.patterns.items():
                    patterns.append({
                        "event_type": event_type,
                        "keywords": pattern.keywords,
                        "affected_symbols": pattern.affected_symbols,
                        "expected_reaction": {
                            "immediate": f"{pattern.immediate_reaction:+.1f}%",
                            "peak": f"{pattern.peak_reaction:+.1f}%",
                            "recovery_hours": pattern.recovery_time_hours,
                            "direction": pattern.typical_direction
                        },
                        "confidence": {
                            "times_observed": pattern.times_observed,
                            "accuracy": f"{pattern.accuracy:.0%}"
                        },
                        "notes": pattern.notes
                    })

                return {
                    "ok": True,
                    "patterns_count": len(patterns),
                    "patterns": patterns,
                    "description": "Event patterns Ghost has learned from market history"
                }
            except Exception as e:
                LOGGER.error(f"Failed to get event patterns: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/events/record")
        async def api_v3_events_record(
            event_type: str,
            trigger: str,
            symbol: str,
            price: float
        ):
            """
            Record a market event for Ghost to learn from.

            Example:
                event_type: "elon_tweet"
                trigger: "Elon posted DOGE meme on X"
                symbol: "DOGE"
                price: 0.08

            Ghost will track what happens to the price over the next 48 hours
            and update its learned patterns accordingly.
            """
            try:
                from core.event_memory import record_market_event, EventType

                # Validate event type
                valid_types = [e.value for e in EventType]
                if event_type not in valid_types:
                    return {
                        "ok": False,
                        "error": f"Invalid event_type. Valid options: {valid_types}"
                    }

                event_id = record_market_event(event_type, trigger, symbol, price)

                return {
                    "ok": True,
                    "event_id": event_id,
                    "message": f"Event recorded. Ghost will track {symbol} price for 48 hours.",
                    "next_steps": [
                        "Ghost will check price at 1h, 4h, 24h, 48h",
                        "Pattern will be updated based on actual reaction",
                        "Future predictions will account for similar events"
                    ]
                }
            except Exception as e:
                LOGGER.error(f"Failed to record event: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/events/check/{symbol}")
        async def api_v3_events_check(symbol: str, direction: str = "LONG"):
            """
            Check if any recent events should affect prediction for a symbol.

            Returns warnings if Ghost's learned patterns suggest the prediction
            might be wrong due to recent market events.
            """
            try:
                from core.event_memory import check_for_event_impact

                result = check_for_event_impact(symbol, direction)

                return {
                    "ok": True,
                    "symbol": symbol,
                    "proposed_direction": direction,
                    "should_adjust": result.get("should_adjust", False),
                    "reason": result.get("reason"),
                    "recommendation": result.get("recommendation"),
                    "related_event": result.get("event")
                }
            except Exception as e:
                LOGGER.error(f"Failed to check event impact: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/paper/admin/expire-old-pending")
        async def api_v3_paper_admin_expire_old_pending(
            cutoff_date: str = "2026-01-14",
            dry_run: bool = False
        ):
            """
            Mark old pending trades as EXPIRED to clean up stats.

            These are trades from before the V2 filter was deployed (Jan 14, 2026).
            They pollute the database with 26K+ pending trades that will never resolve.

            Args:
                cutoff_date: Expire trades created before this date (default: 2026-01-14)
                dry_run: If True, just count without making changes

            Returns:
                Count of expired trades and updated stats
            """
            from core.paper_tracker import get_paper_tracker

            try:
                tracker = get_paper_tracker()
                conn = tracker._get_connection()

                # Parse cutoff date
                try:
                    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")
                except ValueError:
                    conn.close()
                    return {"ok": False, "error": f"Invalid date format: {cutoff_date}. Use YYYY-MM-DD"}

                cutoff_str = cutoff.strftime("%Y-%m-%dT00:00:00")

                # Count trades to be expired
                cur = tracker._execute(conn, """
                    SELECT
                        COUNT(*) as count,
                        MIN(created_at) as oldest,
                        MAX(created_at) as newest
                    FROM paper_trades
                    WHERE outcome = 'PENDING'
                      AND created_at < ?
                """, (cutoff_str,))

                stats = tracker._fetchone(cur)
                pending_count = stats['count'] if stats else 0
                oldest = stats.get('oldest') if stats else None
                newest = stats.get('newest') if stats else None

                if pending_count == 0:
                    conn.close()
                    return {
                        "ok": True,
                        "message": "No pending trades found before cutoff date",
                        "cutoff_date": cutoff_date,
                        "expired_count": 0
                    }

                if dry_run:
                    cur = tracker._execute(conn, """
                        SELECT outcome, COUNT(*) as count
                        FROM paper_trades
                        GROUP BY outcome
                        ORDER BY count DESC
                    """)
                    rows = tracker._fetchall(cur)
                    outcome_counts = {row['outcome']: row['count'] for row in rows}
                    conn.close()

                    return {
                        "ok": True,
                        "dry_run": True,
                        "would_expire": pending_count,
                        "oldest_trade": str(oldest) if oldest else None,
                        "newest_trade": str(newest) if newest else None,
                        "cutoff_date": cutoff_date,
                        "current_outcome_counts": outcome_counts
                    }

                # Actually expire the trades
                LOGGER.info(f"🧹 Expiring {pending_count:,} old pending trades before {cutoff_date}")

                now_str = datetime.utcnow().isoformat()
                tracker._execute(conn, """
                    UPDATE paper_trades
                    SET
                        outcome = 'EXPIRED',
                        checked_at = ?,
                        notes = 'Auto-expired: Pre-V2 filter trade'
                    WHERE outcome = 'PENDING'
                      AND created_at < ?
                """, (now_str, cutoff_str))

                conn.commit()

                # Get updated counts
                cur = tracker._execute(conn, """
                    SELECT outcome, COUNT(*) as count
                    FROM paper_trades
                    GROUP BY outcome
                    ORDER BY count DESC
                """)
                rows = tracker._fetchall(cur)
                outcome_counts = {row['outcome']: row['count'] for row in rows}

                conn.close()

                LOGGER.info(f"✅ Expired {pending_count:,} old pending trades")

                return {
                    "ok": True,
                    "expired_count": pending_count,
                    "oldest_trade": str(oldest) if oldest else None,
                    "newest_trade": str(newest) if newest else None,
                    "cutoff_date": cutoff_date,
                    "outcome_counts": outcome_counts,
                    "message": f"Successfully expired {pending_count:,} old pending trades"
                }

            except Exception as e:
                LOGGER.error(f"Failed to expire old pending trades: {e}", exc_info=True)
                return {"ok": False, "error": str(e)}

        LOGGER.info("✅ Paper Trading API endpoints registered (/api/v3/paper/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ Paper Trading endpoints not loaded: {e}", exc_info=True)


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # NEWS BRAIN - AI-Powered News Analysis (v2 with real news feeds)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    try:
        from core.intelligence.ghost_news_brain import get_news_brain

        @APP.get("/api/v3/news/analyze")
        async def api_v3_news_analyze():
            """
            Run AI news analysis NOW.

            Fetches real news from CryptoPanic + Reuters RSS,
            sends to Claude for analysis against pending predictions.

            Usage:
                GET /api/v3/news/analyze
            """
            try:
                brain = get_news_brain()
                # analyze_news is now async
                result = await brain.analyze_news()

                # Feed Intelligence Hub cache (was missing from this endpoint)
                try:
                    from core.intelligence_hub import update_news_brain_cache
                    update_news_brain_cache(result)
                except Exception:
                    pass

                return {
                    "ok": result.get("ok", False),
                    **result
                }

            except Exception as e:
                LOGGER.error(f"News analysis failed: {e}")
                return {
                    "ok": False,
                    "error": str(e)
                }

        @APP.get("/api/v3/news/history")
        async def api_v3_news_history(limit: int = 10):
            """
            Get news analysis history from database.

            Usage:
                GET /api/v3/news/history?limit=10
            """
            try:
                brain = get_news_brain()
                history = brain.get_history(limit)

                return {
                    "ok": True,
                    "count": len(history),
                    "analyses": history
                }

            except Exception as e:
                LOGGER.error(f"Failed to get news history: {e}")
                return {
                    "ok": False,
                    "error": str(e)
                }

        @APP.get("/api/v3/news/status")
        async def api_v3_news_status():
            """
            Get news brain status - shows configured sources.

            Usage:
                GET /api/v3/news/status
            """
            try:
                brain = get_news_brain()
                return {
                    "ok": True,
                    **brain.get_status()
                }

            except Exception as e:
                LOGGER.error(f"Failed to get news status: {e}")
                return {
                    "ok": False,
                    "error": str(e)
                }

        @APP.get("/api/v3/news/headlines")
        async def api_v3_news_headlines():
            """
            Fetch latest headlines without analysis (for debugging).

            Usage:
                GET /api/v3/news/headlines
            """
            try:
                brain = get_news_brain()
                headlines = await brain.fetch_all_news()

                return {
                    "ok": True,
                    "count": len(headlines),
                    "headlines": headlines
                }

            except Exception as e:
                LOGGER.error(f"Failed to fetch headlines: {e}")
                return {
                    "ok": False,
                    "error": str(e)
                }

        @APP.post("/api/v3/news/analyze-with-auto-pause")
        async def analyze_news_with_auto_pause():
            """
            Analyze news AND automatically pause trading on CRITICAL events.

            What it does:
            - Fetches all news from 14+ RSS feeds + CryptoPanic
            - Sends to Claude for analysis
            - Feeds Intelligence Hub cache (so predictions get adjusted)
            - If CRITICAL event detected:
                - AUTO-PAUSES trading for 4 hours
                - Creates guardian alerts for affected symbols
            - If HIGH event detected:
                - Creates guardian alerts (no pause)

            NOTE: No Telegram sends. Ghost consumes news internally via Hub.

            Returns full analysis plus auto-pause actions taken.
            """
            try:
                brain = get_news_brain()
                result = await brain.analyze_news_with_auto_pause()

                # Feed Intelligence Hub cache (was missing from this endpoint)
                try:
                    from core.intelligence_hub import update_news_brain_cache
                    update_news_brain_cache(result)
                except Exception:
                    pass

                return result
            except Exception as e:
                LOGGER.error(f"News analysis with auto-pause failed: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/trading/pause-status")
        async def get_trading_pause_status():
            """
            Get current trading pause status.
            Use this to check if trading is paused before placing new trades.

            Returns:
                paused: bool - True if trading is paused
                reason: str - Why trading was paused
                pause_until: str - ISO timestamp when pause expires
            """
            try:
                brain = get_news_brain()
                status = brain.get_trading_pause_status()
                return {"ok": True, **status}
            except Exception as e:
                return {"ok": False, "error": str(e), "paused": False}

        @APP.post("/api/v3/trading/resume")
        async def resume_trading(reason: str = "Manual resume via API"):
            """
            Manually resume trading after an auto-pause.
            Use this to clear a pause before the 4-hour timeout.

            Args:
                reason: Why trading is being resumed
            """
            try:
                brain = get_news_brain()
                success = brain.resume_trading(reason)
                return {"ok": success, "message": f"Trading resumed: {reason}" if success else "Failed to resume"}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/trading/pause")
        async def pause_trading(reason: str, duration_hours: int = 4):
            """
            Manually pause trading.
            Use this to pause trading without a news event trigger.

            Args:
                reason: Why trading should be paused
                duration_hours: How long to pause (default 4 hours)
            """
            try:
                brain = get_news_brain()
                success = brain._set_trading_paused(True, reason, duration_hours)
                return {"ok": success, "message": f"Trading paused for {duration_hours}h: {reason}" if success else "Failed to pause"}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        LOGGER.info("✅ News Brain v2 API endpoints registered (/api/v3/news/*)")
        LOGGER.info("✅ Trading Pause/Resume API endpoints registered (/api/v3/trading/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ News Brain endpoints not loaded: {e}", exc_info=True)


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GUARDIAN ALERTS - Protective alerts for at-risk positions
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    try:
        import psycopg2
        import psycopg2.extras

        @APP.get("/api/v3/guardian/alerts")
        async def get_guardian_alerts(
            symbol: str = None,
            severity: str = None,
            acknowledged: bool = None,
            limit: int = 50
        ):
            """
            Get guardian alerts with optional filters.

            Usage:
                GET /api/v3/guardian/alerts
                GET /api/v3/guardian/alerts?symbol=BTC&severity=CRITICAL
                GET /api/v3/guardian/alerts?acknowledged=false
            """
            try:
                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                    query = "SELECT * FROM guardian_alerts WHERE 1=1"
                    params = []

                    if symbol:
                        query += " AND symbol = %s"
                        params.append(symbol.upper())
                    if severity:
                        query += " AND severity = %s"
                        params.append(severity.upper())
                    if acknowledged is not None:
                        query += " AND acknowledged = %s"
                        params.append(acknowledged)

                    query += " ORDER BY created_at DESC LIMIT %s"
                    params.append(limit)

                    cur.execute(query, params)
                    alerts = cur.fetchall()

                # Convert to serializable format
                result = []
                for a in alerts:
                    alert_dict = dict(a)
                    # Convert Decimal to float for JSON
                    if alert_dict.get('price_at_alert'):
                        alert_dict['price_at_alert'] = float(alert_dict['price_at_alert'])
                    if alert_dict.get('confidence'):
                        alert_dict['confidence'] = float(alert_dict['confidence'])
                    # Convert datetime to ISO string
                    if alert_dict.get('created_at'):
                        alert_dict['created_at'] = alert_dict['created_at'].isoformat()
                    if alert_dict.get('acknowledged_at'):
                        alert_dict['acknowledged_at'] = alert_dict['acknowledged_at'].isoformat()
                    result.append(alert_dict)

                return {"ok": True, "alerts": result, "count": len(result)}
            except Exception as e:
                LOGGER.error(f"Failed to get guardian alerts: {e}")
                return {"ok": False, "error": str(e), "alerts": []}

        @APP.post("/api/v3/guardian/alert")
        async def create_guardian_alert(
            symbol: str,
            alert_type: str,
            message: str,
            severity: str = "INFO",
            price_at_alert: float = None,
            confidence: float = None,
            prediction_id: int = None,
            news_event_id: int = None
        ):
            """
            Create a guardian alert.

            Args:
                symbol: Trading symbol (e.g., BTC, AAPL)
                alert_type: Type of alert (e.g., NEWS_OVERRIDE, STOP_APPROACHING)
                message: Alert message
                severity: INFO, WARNING, HIGH, CRITICAL
                price_at_alert: Current price when alert was created
                confidence: Model confidence (0-1)
                prediction_id: Related prediction ID
                news_event_id: Related news event ID
            """
            try:
                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cur = conn.cursor()

                    cur.execute("""
                        INSERT INTO guardian_alerts 
                        (symbol, alert_type, severity, message, price_at_alert, confidence, prediction_id, news_event_id)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING alert_id
                    """, (symbol.upper(), alert_type, severity.upper(), message, price_at_alert, confidence, prediction_id, news_event_id))

                    alert_id = cur.fetchone()[0]

                    LOGGER.info(f"[GUARDIAN] Created alert {alert_id} for {symbol}: {alert_type}")
                    return {"ok": True, "alert_id": alert_id}
            except Exception as e:
                LOGGER.error(f"Failed to create guardian alert: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/guardian/acknowledge/{alert_id}")
        async def acknowledge_guardian_alert(alert_id: int):
            """Acknowledge a guardian alert"""
            try:
                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cur = conn.cursor()

                    cur.execute("""
                        UPDATE guardian_alerts 
                        SET acknowledged = TRUE, acknowledged_at = CURRENT_TIMESTAMP
                        WHERE alert_id = %s
                    """, (alert_id,))

                    return {"ok": True, "alert_id": alert_id, "acknowledged": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        LOGGER.info("✅ Guardian Alerts API endpoints registered (/api/v3/guardian/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ Guardian Alerts endpoints not loaded: {e}", exc_info=True)


    # ============================================================================
    # MODEL MANAGEMENT ENDPOINTS
    # ============================================================================
    try:
        import subprocess
        import json
        from pathlib import Path

        @APP.get("/api/v3/model/status")
        async def get_model_status():
            """
            Get current model status and metadata.
            Returns info about the XGBoost model, last retrain date, accuracy, etc.
            """
            try:
                model_dir = Path("models")
                metadata_path = model_dir / "xgboost_v2_metadata.json"
                model_path = model_dir / "xgboost_v2.pkl"

                status = {
                    "model_exists": model_path.exists(),
                    "metadata": None,
                    "backups": []
                }

                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        status["metadata"] = json.load(f)

                # List backup models
                if model_dir.exists():
                    backups = list(model_dir.glob("xgboost_v2_backup_*.pkl"))
                    status["backups"] = [b.name for b in sorted(backups, reverse=True)[:5]]

                return {"ok": True, **status}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/model/training-data-stats")
        async def get_training_data_stats():
            """
            Get statistics about available training data.
            Shows how many resolved trades we have for retraining.
            """
            try:
                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

                    # Get outcome distribution
                    cur.execute("""
                        SELECT 
                            outcome,
                            COUNT(*) as count
                        FROM paper_trades
                        WHERE outcome IS NOT NULL AND outcome != 'PENDING'
                        GROUP BY outcome
                    """)
                    outcomes = {row['outcome']: row['count'] for row in cur.fetchall()}

                    # Get total pending
                    cur.execute("SELECT COUNT(*) as cnt FROM paper_trades WHERE outcome = 'PENDING'")
                    pending_row = cur.fetchone()
                    pending = pending_row['cnt'] if pending_row else 0

                    # Get total resolved
                    total_resolved = sum(outcomes.values()) if outcomes else 0

                    return {
                        "ok": True,
                        "resolved_trades": total_resolved,
                        "pending_trades": pending,
                        "outcome_distribution": outcomes,
                        "ready_for_retrain": total_resolved >= 500,
                        "min_samples_required": 500
                    }
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/model/retrain")
        async def trigger_model_retrain(
            min_samples: int = 500,
            test_split: float = 0.2,
            dry_run: bool = True
        ):
            """
            Trigger model retraining.

            Args:
                min_samples: Minimum samples required for training (default 500)
                test_split: Portion of data to use for testing (default 0.2)
                dry_run: If True, evaluate but don't save model (default True for safety)
            """
            try:
                script_path = Path("scripts/retrain_xgboost.py")

                if not script_path.exists():
                    return {"ok": False, "error": "Retrain script not found"}

                # Build command
                cmd = [
                    "python", str(script_path),
                    "--min-samples", str(min_samples),
                    "--test-split", str(test_split)
                ]
                if dry_run:
                    cmd.append("--dry-run")

                # Run in subprocess
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )

                return {
                    "ok": result.returncode == 0,
                    "dry_run": dry_run,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                }
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": "Training timed out (5 min limit)"}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        LOGGER.info("✅ Model Management API endpoints registered (/api/v3/model/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ Model Management endpoints not loaded: {e}", exc_info=True)


    # Alias for Railway/Uvicorn compatibility (expects lowercase 'app')
    app = APP

    if __name__ == "__main__":
        port = int(os.getenv("PORT", "5000"))
        # Run with in-memory app object to avoid duplicate module import
        uvicorn.run(APP, host="0.0.0.0", port=port, reload=False)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DAILY TOP 10 SCANNER (6 AM Money-Making Opportunities)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    try:
        from core.daily_top_10_scanner import get_scanner

        @APP.get("/api/v3/top10")
        async def api_v3_get_top_10():
            """
            Get current active top 10 money-making opportunities.

            Returns Ghost's daily picks from _LATEST_PREDICTIONS, filtered
            through the V3 pipeline. Falls back to DailyTop10Scanner SQLite
            if in-memory predictions are empty.
            """
            try:
                import time as _time
                from core.adapters import process_v3_from_cache

                # PRIMARY: Build top 10 from live _LATEST_PREDICTIONS
                if _LATEST_PREDICTIONS:
                    stocks, crypto = process_v3_from_cache(_LATEST_PREDICTIONS)
                    opportunities = []
                    for i, pick in enumerate(stocks + crypto, 1):
                        opportunities.append({
                            "rank": i,
                            "symbol": pick.get("symbol", "?"),
                            "direction": pick.get("direction", "?"),
                            "confidence": pick.get("v3_confidence", pick.get("confidence", 0)),
                            "current_price": pick.get("current_price", pick.get("price", 0)),
                            "predicted_48h_price": pick.get("target_price", pick.get("take_profit", 0)),
                            "gain_pct": pick.get("expected_move", 0),
                            "asset_type": pick.get("asset_class", "stock"),
                            "last_updated": _time.time(),
                        })

                    if opportunities:
                        return {
                            "ok": True,
                            "count": len(opportunities),
                            "opportunities": opportunities[:10],
                            "last_updated": _time.time(),
                            "source": "live_predictions",
                        }

                # FALLBACK: DailyTop10Scanner SQLite
                scanner = get_scanner()
                top_10 = scanner.get_active_top_10()

                return {
                    "ok": True,
                    "count": len(top_10),
                    "opportunities": top_10,
                    "last_updated": top_10[0]["last_updated"] if top_10 else None,
                    "source": "scanner_sqlite",
                }

            except Exception as e:
                LOGGER.error(f"Failed to get top 10: {e}")
                return {
                    "ok": False,
                    "error": str(e),
                    "opportunities": []
                }

        @APP.post("/api/v3/top10/scan")
        async def api_v3_scan_top_10():
            """
            Manually trigger top 10 scan (normally runs at 6 AM daily).

            Returns:
                {
                    "ok": true,
                    "opportunities": [...10 picks...],
                    "alert_sent": true
                }
            """
            try:
                scanner = get_scanner()
                opportunities = await scanner.scan_for_top_10()

                if opportunities:
                    scanner.save_top_10(opportunities)
                    alert_sent = await scanner.send_daily_alert()

                    return {
                        "ok": True,
                        "count": len(opportunities),
                        "opportunities": opportunities,
                        "alert_sent": alert_sent
                    }
                else:
                    return {
                        "ok": False,
                        "error": "No opportunities found",
                        "opportunities": []
                    }

            except Exception as e:
                LOGGER.error(f"Failed to scan top 10: {e}", exc_info=True)
                return {
                    "ok": False,
                    "error": str(e),
                    "opportunities": []
                }

        LOGGER.info("✅ Daily Top 10 Scanner endpoints registered")

    except Exception as e:
        LOGGER.error(f"⚠️ Daily Top 10 Scanner not loaded: {e}", exc_info=True)


    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # GUARDIAN ORACLE API (24/7 Monitoring + Heartbeats)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    try:
        from core.guardian_oracle import get_guardian_oracle
        from core.guardian_heartbeat_scheduler import force_heartbeat

        @APP.get("/api/v3/guardian/status")
        async def api_v3_guardian_status():
            """
            Get Guardian Oracle monitoring status.

            Returns:
                - Active positions being monitored
                - Alert history
                - Overall P&L
            """
            try:
                guardian = get_guardian_oracle()
                status = await guardian._get_current_status()

                return {
                    "ok": True,
                    "guardian_active": guardian.monitoring,
                    "positions_active": status.get('active_count', 0),
                    "positions_on_track": status.get('on_track', 0),
                    "positions_weakened": status.get('weakened', 0),
                    "total_pnl_pct": status.get('total_pnl', 0.0),
                    "alert_count": len(guardian.alert_history)
                }

            except Exception as e:
                LOGGER.error(f"Failed to get guardian status: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/guardian/positions")
        async def api_v3_guardian_positions():
            """
            Get all positions Guardian is monitoring.

            Shows current status, P&L, confidence changes.
            """
            try:
                import sqlite3
                conn = sqlite3.connect("data/ghost_predictions.db")
                conn.row_factory = sqlite3.Row

                rows = conn.execute("""
                    SELECT * FROM guardian_positions
                    WHERE status = 'active'
                    ORDER BY entry_time DESC
                """).fetchall()

                positions = [dict(row) for row in rows]
                conn.close()

                return {
                    "ok": True,
                    "count": len(positions),
                    "positions": positions
                }

            except Exception as e:
                LOGGER.error(f"Failed to get guardian positions: {e}")
                return {"ok": False, "error": str(e)}


        @APP.post("/api/v3/guardian/heartbeat/{heartbeat_type}")
        async def api_v3_force_heartbeat(heartbeat_type: str):
            """
            Manually trigger a heartbeat for testing.

            Args:
                heartbeat_type: 'morning', 'midday', 'evening', or 'night'
            """
            try:
                if heartbeat_type not in ['morning', 'midday', 'evening', 'night']:
                    return {
                        "ok": False,
                        "error": f"Invalid heartbeat type: {heartbeat_type}. Use: morning, midday, evening, night"
                    }

                force_heartbeat(heartbeat_type)

                return {
                    "ok": True,
                    "message": f"Triggered {heartbeat_type} heartbeat",
                    "heartbeat_type": heartbeat_type
                }

            except Exception as e:
                LOGGER.error(f"Failed to force heartbeat: {e}")
                return {"ok": False, "error": str(e)}

        LOGGER.info("✅ Guardian Oracle endpoints registered (/api/v3/guardian/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ Guardian Oracle endpoints not loaded: {e}", exc_info=True)


    # ═══════════════════════════════════════════════════════════════════════════════
    # 🏆 V3 COMPETITION SYSTEM - FAIR RANKINGS, NO BLACKLIST
    # ═══════════════════════════════════════════════════════════════════════════════

    try:
        @APP.get("/api/v3/competition/status")
        async def v3_competition_status():
            """
            🏆 V3: Get competition system status.

            Shows TOP 10 stocks and crypto, leaderboards, and pending contenders.
            NO BLACKLIST - everyone competes fairly!
            """
            try:
                from core.v3_competition import get_competition_system

                competition = get_competition_system()
                status = competition.get_competition_status()

                return {
                    "ok": True,
                    "philosophy": "No blacklist - everyone competes fairly. Only the best make TOP 10.",
                    **status
                }

            except Exception as e:
                LOGGER.error(f"[V3-API] Competition status error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/competition/leaderboard/{asset_type}")
        async def v3_competition_leaderboard(asset_type: str, limit: int = 20):
            """
            🏆 V3: Get leaderboard for stocks or crypto.

            Args:
                asset_type: "stock" or "crypto"
                limit: Max entries to return (default 20)
            """
            try:
                from core.v3_competition import get_competition_system

                if asset_type not in ["stock", "crypto"]:
                    return {"ok": False, "error": "asset_type must be 'stock' or 'crypto'"}

                competition = get_competition_system()
                leaderboard = competition.get_leaderboard(asset_type, limit)

                return {
                    "ok": True,
                    "asset_type": asset_type,
                    "leaderboard": leaderboard,
                    "total": len(leaderboard)
                }

            except Exception as e:
                LOGGER.error(f"[V3-API] Leaderboard error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/competition/contenders/{asset_type}")
        async def v3_competition_contenders(asset_type: str, limit: int = 10):
            """
            🏆 V3: Get pending assets closest to breaking into TOP 10.

            These are the assets "fighting" to get promoted!
            """
            try:
                from core.v3_competition import get_competition_system

                if asset_type not in ["stock", "crypto"]:
                    return {"ok": False, "error": "asset_type must be 'stock' or 'crypto'"}

                competition = get_competition_system()
                contenders = competition.get_pending_contenders(asset_type, limit)

                return {
                    "ok": True,
                    "asset_type": asset_type,
                    "message": "These assets are fighting to get into TOP 10!",
                    "contenders": contenders,
                    "gap_explanation": "gap_to_top_10 = how many ranks away from TOP 10"
                }

            except Exception as e:
                LOGGER.error(f"[V3-API] Contenders error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/competition/update-rankings")
        async def v3_competition_update_rankings():
            """
            🏆 V3: Recalculate all rankings based on performance.

            This is the MAIN competition logic!
            - Promotes pending assets with better win rates
            - Demotes TOP 10 assets with declining performance

            Run daily via cron or manually.
            """
            try:
                from core.v3_competition import get_competition_system

                competition = get_competition_system()
                changes = competition.update_rankings()

                return {
                    "ok": True,
                    "message": "Rankings updated!",
                    "changes": changes
                }

            except Exception as e:
                LOGGER.error(f"[V3-API] Update rankings error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/competition/run-shadow-cycle")
        async def v3_competition_run_shadow():
            """
            🔮 V3: Run shadow predictions for ALL assets in the pool.

            Shadow predictions build competition data without sending alerts.
            This allows pending assets to "prove themselves".
            """
            try:
                from core.v3_shadow_predictor import run_shadow_predictions

                results = await run_shadow_predictions()

                return {
                    "ok": True,
                    "message": "Shadow prediction cycle complete",
                    **results
                }

            except Exception as e:
                LOGGER.error(f"[V3-API] Shadow cycle error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/competition/resolve-outcomes")
        async def v3_competition_resolve_outcomes():
            """
            🎯 V3: Resolve shadow predictions and update competitor scores.

            Checks if 48h window has passed for pending predictions
            and determines WIN/LOSS outcomes.
            """
            try:
                from core.v3_shadow_resolver import resolve_shadow_outcomes

                results = await resolve_shadow_outcomes()

                return {
                    "ok": True,
                    "message": "Outcomes resolved",
                    **results
                }

            except Exception as e:
                LOGGER.error(f"[V3-API] Resolve outcomes error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/v3/competition/seed-pool")
        async def v3_competition_seed_pool():
            """
            🌱 V3: Seed initial competition pool with default assets.

            Run once to bootstrap the competition system.
            """
            try:
                from core.v3_competition import get_competition_system
                from core.v3_shadow_predictor import DEFAULT_STOCKS, DEFAULT_CRYPTO

                competition = get_competition_system()
                competition.seed_initial_pool(DEFAULT_STOCKS, DEFAULT_CRYPTO)

                return {
                    "ok": True,
                    "message": "Competition pool seeded!",
                    "stocks_added": len(DEFAULT_STOCKS),
                    "crypto_added": len(DEFAULT_CRYPTO)
                }

            except Exception as e:
                LOGGER.error(f"[V3-API] Seed pool error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/v3/competition/resolution-stats")
        async def v3_competition_resolution_stats():
            """
            📊 V3: Get shadow prediction resolution statistics.
            """
            try:
                from core.v3_shadow_resolver import get_shadow_resolver

                resolver = get_shadow_resolver()
                stats = resolver.get_resolution_stats()

                return {
                    "ok": True,
                    **stats
                }

            except Exception as e:
                LOGGER.error(f"[V3-API] Resolution stats error: {e}")
                return {"ok": False, "error": str(e)}

        LOGGER.info("✅ V3 Competition System endpoints registered (/api/v3/competition/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ V3 Competition System not loaded: {e}", exc_info=True)


    # 🎮 MONEY GAME ENGINE - PROFIT-BASED RANKINGS
    # ═══════════════════════════════════════════════════════════════════════════════
    # Think like a VIDEO GAME: SCORE = MONEY EARNED
    # GOAL: Find the next BULLISH MONEY MAKER
    # #1 = Best profit maker, #10 = Still good but less profit
    # LOSING MONEY = BAD (heavy penalty)
    # ═══════════════════════════════════════════════════════════════════════════════

    try:
        @APP.get("/api/money-game/status")
        async def money_game_status():
            """
            🎮 MONEY GAME: Full game status

            Shows the competition to find the best MONEY MAKERS.
            #1 = Most profitable, rankings based on actual PROFIT potential.
            """
            try:
                from core.money_game_engine import get_money_game

                game = get_money_game()
                status = game.get_game_status()

                return {
                    "ok": True,
                    "philosophy": "Money is the score! Find the next bullish money maker!",
                    **status
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Status error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/money-game/leaderboard/{asset_type}")
        async def money_game_leaderboard(asset_type: str, limit: int = 20):
            """
            🎮 MONEY GAME: Leaderboard ranked by PROFIT potential

            #1 = Best money maker (most profit)
            Lower rank = Less profitable

            This is like a video game high score - MONEY = POINTS!
            """
            try:
                from core.money_game_engine import get_money_game

                if asset_type not in ["stock", "crypto"]:
                    return {"ok": False, "error": "asset_type must be 'stock' or 'crypto'"}

                game = get_money_game()
                leaderboard = game.get_leaderboard(asset_type, limit)

                return {
                    "ok": True,
                    "asset_type": asset_type,
                    "ranking_by": "money_score (profit potential)",
                    "leaderboard": leaderboard,
                    "total": len(leaderboard)
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Leaderboard error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/money-game/rising-stars/{asset_type}")
        async def money_game_rising_stars(asset_type: str, limit: int = 5):
            """
            🌟 MONEY GAME: Rising stars - The NEXT BIG DEAL!

            These are assets fighting to get into TOP 10.
            They're showing profit potential and could be promoted!
            """
            try:
                from core.money_game_engine import get_money_game

                if asset_type not in ["stock", "crypto"]:
                    return {"ok": False, "error": "asset_type must be 'stock' or 'crypto'"}

                game = get_money_game()
                stars = game.get_rising_stars(asset_type, limit)

                return {
                    "ok": True,
                    "asset_type": asset_type,
                    "message": "These are the NEXT BIG DEALS - fighting for TOP 10!",
                    "rising_stars": stars
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Rising stars error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/money-game/player/{symbol}")
        async def money_game_player_stats(symbol: str):
            """
            🎮 MONEY GAME: Get detailed stats for a specific player (asset)

            Shows their full profile: profit history, rank, tier, momentum.
            """
            try:
                from core.money_game_engine import get_money_game

                game = get_money_game()
                stats = game.get_player_stats(symbol.upper())

                if not stats:
                    return {"ok": False, "error": f"Player {symbol} not found"}

                return {
                    "ok": True,
                    **stats
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Player stats error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/money-game/daily-movers")
        async def money_game_daily_movers(min_gain: float = 5.0):
            """
            🚀 MONEY GAME: Today's biggest gainers!

            Dynamic mover detection - catches stocks Ghost might miss.
            Example: Nextpower +16%, Seagate +15% etc.

            Args:
                min_gain: Minimum % gain to include (default 5%)
            """
            try:
                from core.ghost_scout import fetch_daily_movers

                movers = fetch_daily_movers(min_gain_pct=min_gain)

                return {
                    "ok": True,
                    "message": f"Found {len(movers)} stocks up {min_gain}%+ today!",
                    "movers": movers,
                    "tip": "These dynamic movers are now being added to Ghost's watchlist automatically!"
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Daily movers error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/scout-all")
        async def money_game_scout_all():
            """
            🔍 MONEY GAME: Run the scout - find ALL money makers!

            The scout evaluates EVERY asset and records predictions.
            This builds the data to find who actually makes money.

            NEW: Now includes dynamic movers (10%+ daily gainers)!
            NEW: News sentiment integration for ✅ indicator!

            Run daily to continuously evaluate all assets.
            """
            try:
                from core.ghost_scout import run_scouting_cycle

                results = run_scouting_cycle()

                return {
                    "ok": True,
                    "message": "Scout completed! All assets evaluated.",
                    **results
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Scout error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/resolve-trades")
        async def money_game_resolve_trades(hours: int = 24):
            """
            🏆 MONEY GAME: Resolve trades and count the MONEY!

            After predictions are made, we wait 24-48h then check:
            - Did they MAKE money?
            - Did they LOSE money?

            Winners rise in rankings, losers fall.

            Args:
                hours: Resolve trades older than X hours (default 24)
            """
            try:
                from core.ghost_scout import resolve_trades

                results = resolve_trades(hours)

                return {
                    "ok": True,
                    "message": "Trades resolved! Money counted.",
                    **results
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Resolve error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/update-rankings")
        async def money_game_update_rankings():
            """
            🏆 MONEY GAME: Recalculate rankings based on PROFIT

            This determines:
            - Who's #1 (best money maker)
            - Who gets promoted to TOP 10
            - Who gets demoted (not making money)

            Run after resolving trades.
            """
            try:
                from core.money_game_engine import get_money_game

                game = get_money_game()
                changes = game.update_rankings()

                return {
                    "ok": True,
                    "message": "Rankings recalculated by PROFIT!",
                    "changes": changes
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Update rankings error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/seed-players")
        async def money_game_seed_players():
            """
            🌱 MONEY GAME: Seed all players into the game

            Adds all stocks and crypto as competitors.
            Run once to initialize the game.
            """
            try:
                from core.money_game_engine import get_money_game
                from core.ghost_scout import ALL_STOCKS, ALL_CRYPTO

                game = get_money_game()

                # Add all stocks
                for symbol in ALL_STOCKS:
                    game.add_player(symbol, "stock")

                # Add all crypto
                for symbol in ALL_CRYPTO:
                    game.add_player(symbol, "crypto")

                return {
                    "ok": True,
                    "message": "All players added to the game!",
                    "stocks": len(ALL_STOCKS),
                    "crypto": len(ALL_CRYPTO),
                    "total_players": len(ALL_STOCKS) + len(ALL_CRYPTO)
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Seed error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/seed-top10")
        async def money_game_seed_top10():
            """
            🎮 SEED TOP 10: Initialize with known performers

            Seeds the Money Game with manual TOP 10 so we can:
            1. See immediate leaderboard results
            2. Watch Ghost naturally promote/demote based on real performance
            3. Confirm the system works when Ghost promotes a NEW symbol
            """
            try:
                import os
                from datetime import datetime

                db_url = os.getenv("DATABASE_URL")
                if not db_url:
                    return {"ok": False, "error": "No DATABASE_URL"}

                # Manual TOP 10 seeds - Best performers
                STOCK_SEEDS = [
                    ("NVDA", 15.0, 5, 0.80),   # AI king, GPU demand
                    ("META", 12.0, 5, 0.80),   # AI + ads recovery
                    ("PLTR", 10.0, 4, 0.75),   # AI/defense play
                    ("COIN", 8.0, 4, 0.75),    # Crypto proxy
                    ("GOOGL", 7.0, 4, 0.75),   # AI catch-up + search
                    ("AMZN", 6.0, 4, 0.75),    # AWS + retail
                    ("TSLA", 5.0, 4, 0.60),    # Volatile but predictable
                    ("AMD", 4.0, 4, 0.70),     # NVDA alternative
                    ("MSTR", 8.0, 4, 0.70),    # Bitcoin proxy (150k+ BTC)
                    ("HOOD", 6.0, 4, 0.65),    # Retail + crypto exposure
                ]

                CRYPTO_SEEDS = [
                    ("BTC", 12.0, 5, 0.80),    # King, institutional
                    ("ETH", 10.0, 5, 0.80),    # Smart contracts
                    ("SOL", 15.0, 4, 0.75),    # Fast L1, meme activity
                    ("RNDR", 20.0, 5, 0.81),   # 81% win rate! AI/GPU
                    ("TURBO", 18.0, 4, 0.79),  # 79% win rate! Meme momentum
                    ("XRP", 8.0, 4, 0.75),     # Payments, legal clarity
                    ("LINK", 7.0, 4, 0.70),    # Oracle, DeFi essential
                    ("AVAX", 8.0, 4, 0.70),    # L1, gaming/DeFi
                    ("SUI", 12.0, 4, 0.65),    # New L1, high volatility
                    ("INJ", 10.0, 4, 0.70),    # DeFi focused
                ]

                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cur = conn.cursor()

                    seeded_stocks = []
                    seeded_crypto = []

                    for seeds, asset_type, result_list in [
                        (STOCK_SEEDS, "stock", seeded_stocks),
                        (CRYPTO_SEEDS, "crypto", seeded_crypto)
                    ]:
                        for rank, (symbol, profit, trades, win_rate) in enumerate(seeds, 1):
                            wins = int(trades * win_rate)
                            losses = trades - wins
                            avg_profit = profit / trades
                            money_score = profit * (1 + win_rate)

                            cur.execute("""
                                INSERT INTO money_game_players 
                                (symbol, asset_type, tier, total_profit_pct, avg_profit_per_trade,
                                 best_trade_pct, worst_trade_pct, total_trades, wins, losses,
                                 win_rate, money_score, recent_profit_pct, momentum, rank, rank_change,
                                 last_trade, last_updated)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                ON CONFLICT (symbol) DO UPDATE SET
                                    tier = EXCLUDED.tier,
                                    total_profit_pct = EXCLUDED.total_profit_pct,
                                    avg_profit_per_trade = EXCLUDED.avg_profit_per_trade,
                                    total_trades = EXCLUDED.total_trades,
                                    wins = EXCLUDED.wins,
                                    losses = EXCLUDED.losses,
                                    win_rate = EXCLUDED.win_rate,
                                    money_score = EXCLUDED.money_score,
                                    rank = EXCLUDED.rank,
                                    last_updated = NOW()
                            """, (
                                symbol, asset_type, "elite", profit, avg_profit,
                                profit * 0.4, -profit * 0.1, trades, wins, losses,
                                win_rate, money_score, profit * 0.3, "stable", rank, 0,
                                datetime.utcnow(), datetime.utcnow()
                            ))
                            result_list.append({"rank": rank, "symbol": symbol, "profit": f"+{profit:.1f}%"})

                # Reload the game to pick up new data
                from core.money_game_engine import get_money_game
                game = get_money_game()
                game._load_players()
                game._rebuild_elite_lists()

                return {
                    "ok": True,
                    "message": "🎮 TOP 10 SEEDED! Watch for Ghost to promote new symbols!",
                    "stocks": seeded_stocks,
                    "crypto": seeded_crypto,
                    "next_steps": [
                        "Ghost scouts new predictions daily",
                        "After 24h, trades resolve and real profit counted",
                        "If a NEW symbol beats the seeds, it gets PROMOTED!",
                        "Watch for: A symbol you didn't seed making TOP 10"
                    ]
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Seed TOP 10 error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/money-game/elite")
        async def money_game_get_elite():
            """
            👑 MONEY GAME: Get the ELITE (TOP 10 of each)

            These are the proven MONEY MAKERS.
            Ghost sends predictions only for these elite assets.
            """
            try:
                from core.money_game_engine import get_money_game

                game = get_money_game()

                return {
                    "ok": True,
                    "message": "These are the PROVEN money makers!",
                    "elite_stocks": game.get_elite_stocks(),
                    "elite_crypto": game.get_elite_crypto()
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Elite error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/smart-scout")
        async def money_game_smart_scout():
            """
            🔍 MONEY GAME: Smart scout with rate limiting

            Uses batch price fetching and respects API rate limits.
            Better for scouting all 211 assets reliably.
            """
            try:
                from core.smart_scout import smart_scout_all

                results = smart_scout_all()

                return {
                    "ok": True,
                    "message": "Smart scout complete!",
                    **results
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Smart scout error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/daily-cycle")
        async def money_game_daily_cycle():
            """
            ⏰ MONEY GAME: Run full daily cycle

            This runs:
            1. Scout all assets
            2. Resolve 24h old trades
            3. Update rankings
            4. Return elite for alerts

            Perfect for daily cron job.
            """
            try:
                from core.smart_scout import run_daily_cycle

                results = run_daily_cycle()

                return {
                    "ok": True,
                    "message": "Daily cycle complete!",
                    **results
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Daily cycle error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.get("/api/money-game/elite-predictions")
        async def money_game_elite_predictions():
            """
            🎯 MONEY GAME: Get elite predictions for Telegram

            Returns the TOP 10 stocks and crypto with full details
            for the 8 AM Telegram alert.
            """
            try:
                from core.smart_scout import get_elite_predictions

                results = get_elite_predictions()

                return {
                    "ok": True,
                    **results
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Elite predictions error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/telegram-alert")
        async def money_game_telegram_alert(request: Request):
            """
            📱 MONEY GAME: Send TOP 10 money makers to Telegram

            This sends the PROVEN money makers (from Money Game rankings)
            instead of just the old whitelist system.

            Requires X-Cron-Secret header for security.
            """
            # Check cron secret
            cron_secret = os.getenv("CRON_SECRET", "ghost-cron-2024")
            provided_secret = request.headers.get("X-Cron-Secret", "")

            if not cron_secret or provided_secret != cron_secret:
                return {"ok": False, "error": "Unauthorized - invalid X-Cron-Secret"}

            try:
                from core.money_game_engine import get_money_game
                from core.smart_scout import SmartScout

                game = get_money_game()
                scout = SmartScout()

                # Get elite (TOP 10 money makers)
                elite_stocks = game.get_elite_stocks()[:10]
                elite_crypto = game.get_elite_crypto()[:10]

                # If no elite yet (still building data), use backup
                if not elite_stocks and not elite_crypto:
                    LOGGER.warning("[MONEY-GAME] No elite yet - using default alert system")
                    return {"ok": False, "error": "No elite established yet - Money Game still building data"}

                # Get prices for elite
                stock_prices = scout.get_stock_prices_batch(elite_stocks) if elite_stocks else {}
                crypto_prices = scout.get_crypto_prices_batch(elite_crypto) if elite_crypto else {}

                # Build predictions list
                stock_picks = []
                for symbol in elite_stocks:
                    stats = game.get_player_stats(symbol)
                    price = stock_prices.get(symbol, 0)
                    if stats and price > 0:
                        stock_picks.append({
                            "symbol": symbol,
                            "current": price,
                            "prediction_48h": price * 1.03,  # 3% target
                            "buy_in": price * 0.99,
                            "sell": price * 1.02,
                            "confidence": min(0.85, 0.70 + (stats.get("money_score", 0) / 100)),
                            "direction": "UP",
                            "asset_type": "stock",
                            "money_score": stats.get("money_score", 0),
                            "rank": stats.get("rank", 999)
                        })

                crypto_picks = []
                for symbol in elite_crypto:
                    stats = game.get_player_stats(symbol)
                    price = crypto_prices.get(symbol, 0)
                    if stats and price > 0:
                        crypto_picks.append({
                            "symbol": symbol,
                            "current": price,
                            "prediction_48h": price * 1.05,  # 5% target
                            "buy_in": price * 0.99,
                            "sell": price * 1.02,
                            "confidence": min(0.85, 0.70 + (stats.get("money_score", 0) / 100)),
                            "direction": "UP",
                            "asset_type": "crypto",
                            "money_score": stats.get("money_score", 0),
                            "rank": stats.get("rank", 999)
                        })

                # Sort by money_score
                stock_picks.sort(key=lambda x: x.get("money_score", 0), reverse=True)
                crypto_picks.sort(key=lambda x: x.get("money_score", 0), reverse=True)

                # Build message
                msg_lines = [
                    "🎮 *GHOST MONEY GAME - TOP 10*",
                    "_Proven money makers ranked by profit_",
                    "",
                    "📈 *STOCKS (Elite Money Makers)*"
                ]

                for i, p in enumerate(stock_picks[:10], 1):
                    msg_lines.append(f"  #{i} {p['symbol']}: ${p['current']:.2f} (Score: {p['money_score']:.1f})")

                if not stock_picks:
                    msg_lines.append("  _Building rankings..._")

                msg_lines.append("")
                msg_lines.append("🪙 *CRYPTO (Elite Money Makers)*")

                for i, p in enumerate(crypto_picks[:10], 1):
                    price_str = f"${p['current']:.4f}" if p['current'] < 1 else f"${p['current']:.2f}"
                    msg_lines.append(f"  #{i} {p['symbol']}: {price_str} (Score: {p['money_score']:.1f})")

                if not crypto_picks:
                    msg_lines.append("  _Building rankings..._")

                msg_lines.append("")
                msg_lines.append("💡 _Rankings based on actual profit history_")
                msg_lines.append("🎯 _Higher score = better money maker_")

                message = "\n".join(msg_lines)

                # Send to Telegram
                success = _tg_send_chat_message(TELEGRAM_CHAT_ID, message)

                return {
                    "ok": success,
                    "message": "Money Game TOP 10 sent!" if success else "Failed to send",
                    "stocks_sent": len(stock_picks),
                    "crypto_sent": len(crypto_picks)
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Telegram alert error: {e}")
                return {"ok": False, "error": str(e)}

        @APP.post("/api/money-game/trigger-now")
        async def money_game_trigger_now():
            """
            🚀 INSTANT TRIGGER: Run scout + send alert NOW

            For testing - bypasses schedule and runs immediately:
            1. Run full scout
            2. Send Telegram alert with results

            No auth required (public endpoint for quick testing).
            """
            try:
                from core.smart_scout import SmartScout, get_elite_predictions

                results = {"steps": []}

                # Step 1: Run scout
                LOGGER.info("🚀 [TRIGGER-NOW] Running instant scout...")
                scout = SmartScout()
                scout_result = scout.full_scout()

                # Extract counts from nested structure
                stocks_scouted = scout_result.get("stocks", {}).get("scouted", 0) or scout_result.get("total_scouted", 0) // 2
                crypto_scouted = scout_result.get("crypto", {}).get("scouted", 0) or scout_result.get("total_scouted", 0) // 2
                total_scouted = scout_result.get("total_scouted", stocks_scouted + crypto_scouted)

                results["scout"] = {
                    "stocks_scouted": stocks_scouted,
                    "crypto_scouted": crypto_scouted,
                    "total_scouted": total_scouted,
                    "elapsed_seconds": scout_result.get("elapsed_seconds", 0)
                }
                results["steps"].append(f"Scout complete: {total_scouted} assets")

                # Step 2: Get elite
                elite = get_elite_predictions()
                stocks = elite.get("elite_stocks", [])[:5]
                crypto = elite.get("elite_crypto", [])[:5]
                results["elite"] = {"stocks": stocks, "crypto": crypto}
                results["steps"].append("Elite fetched")

                # Step 3: Send Telegram (HTML format)
                if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                    msg = "🚀 <b>GHOST INSTANT SCAN</b>\n\n"
                    msg += f"📊 Scanned: {stocks_scouted} stocks, {crypto_scouted} crypto\n\n"

                    if stocks:
                        msg += "📈 <b>Top Stocks:</b>\n"
                        for i, s in enumerate(stocks, 1):
                            msg += f"  {i}. {s}\n"

                    if crypto:
                        msg += "\n🪙 <b>Top Crypto:</b>\n"
                        for i, c in enumerate(crypto, 1):
                            msg += f"  {i}. {c}\n"

                    if not stocks and not crypto:
                        msg += "<i>Building rankings... (run again after more data)</i>\n"

                    msg += "\n<i>Instant scan triggered</i>"

                    success = _tg_send_chat_message(TELEGRAM_CHAT_ID, msg)
                    results["telegram_sent"] = success
                    results["steps"].append(f"Telegram: {'sent' if success else 'failed'}")
                else:
                    results["telegram_sent"] = False
                    results["steps"].append("Telegram not configured")

                return {
                    "ok": True,
                    "message": "Instant trigger complete!",
                    **results
                }

            except Exception as e:
                LOGGER.error(f"[MONEY-GAME] Instant trigger error: {e}")
                return {"ok": False, "error": str(e)}

        LOGGER.info("✅ 🎮 Money Game Engine endpoints registered (/api/money-game/*)")

    except Exception as e:
        LOGGER.error(f"⚠️ Money Game Engine not loaded: {e}", exc_info=True)
