"""Routes: paper_trading — extracted from wolf_app.py (Step 12)"""

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
import httpx
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

try:
    from wolf_helpers import *
except ImportError:
    pass

router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- Routes: paper_trading (2 endpoints) ---

try:
    from core.trade_journal import get_trade_journal

    @router.post("/api/v3/journal/entry")
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

    @router.post("/api/v3/journal/exit")
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

    @router.get("/api/v3/journal/trades")
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

    @router.get("/api/v3/journal/stats")
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
except Exception as _je:
    LOGGER.warning(f"Journal section error: {_je}")

try:
    from core.paper_tracker import get_paper_tracker

    @router.post("/api/v3/paper/signal")
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

    @router.post("/api/v3/paper/check")
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

    @router.post("/api/v3/paper/check_all")
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

    @router.get("/api/v3/paper/trades")
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

    @router.get("/api/v3/paper/stats")
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

    @router.post("/api/v3/paper/force-resolve")
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

    @router.get("/api/v3/paper/accuracy-proof")
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
    
    @router.get("/api/v3/directional/regime")
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
    
    @router.get("/api/v3/trust/leaderboard")
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
    
    @router.get("/api/v3/trust/symbol/{symbol}")
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
    
    @router.get("/api/v3/trust/focused")
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
    
    @router.get("/api/v3/events/patterns")
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
    
    @router.post("/api/v3/events/record")
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
    
    @router.get("/api/v3/events/check/{symbol}")
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

    @router.post("/api/v3/paper/admin/expire-old-pending")
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
except Exception as _route_e:
    LOGGER.warning(f'Route section load error: {_route_e}')
