"""Routes: forecast — extracted from wolf_app.py (Step 12)"""
# fmt: off
# ruff: noqa

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, RedirectResponse

try:
    import httpx
except ImportError:
    httpx = None

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

try:
    from wolf_helpers import (
        AUTH_DEP, SECURITY_SCHEME, WOLF, WOLF_SQLITE_PATH,
        _is_truthy, _json500, with_cap,
        AlertTemplateBody, AlertToggle, AlertConfigBody,
        RuntimeConfigBody, ControlBody, ModeBody, TrainBody,
        AgentControlBody, CashBody, PositionAddBody, PositionsImportBody,
        WatchlistImportBody, TradeRequest, PredFeedbackBody,
        AddPositionBody, OrderPlaceBody,
        _PredictRunBody, _RecordPriceBody, _ScoreBody, _BacktestBody,
        ChatRequest, AiDecision, TelegramUpdate,
    )
    from fastapi.security import HTTPAuthorizationCredentials
except Exception as _wh_e:
    import logging as _l
    _l.getLogger("ghost").warning(f"wolf_helpers import partial: {_wh_e}")
    AUTH_DEP = None
    WOLF = "WOLF"
    WOLF_SQLITE_PATH = "data/wolf.db"


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

router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- 11 endpoints ---

@router.get("/forecast/48h")
async def get_forecast_48h(symbol: str = WOLF, limit: int = 50):
    """
    Get 48-hour forecast series for a symbol.

    Query params:
    - symbol: Stock symbol (default: WOLF)
    - limit: Max number of forecast points (default: 50)

    Returns:
    {
      "symbol": "WOLF",
      "series": [
        {
          "t": 1739145600,
          "now": 34.13,
          "mid": 35.40,
          "lo": 33.8,
          "hi": 36.7,
          "conf": 0.62,
          "model": "gpt-4o"
        }
      ]
    }
    """
    try:
        series = _get_forecast_48h_series(symbol, limit)
        return {
            "symbol": symbol,
            "series": series,
            "count": len(series),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/48h/metrics")
async def get_forecast_48h_metrics(symbol: str = WOLF, window: int = 30):
    """
    Get accuracy metrics for 48-hour forecasts.

    Query params:
    - symbol: Stock symbol (default: WOLF)
    - window: Number of recent forecasts to evaluate (default: 30)

    Returns:
    {
      "symbol": "WOLF",
      "window": 30,
      "mape48h": 0.081,
      "mae48h": 2.63,
      "hit_rate_band": 0.73,
      "direction_hit": 0.67,
      "bias": "over",
      "bias_bps": 58,
      "last_verified_at": 1739232000,
      "count": 25
    }
    """
    try:
        metrics = _compute_forecast_48h_metrics(symbol, window)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/48h/generate")
async def post_generate_forecast_48h(symbol: str = WOLF):
    """
    Generate a new 48-hour forecast immediately.

    Body params:
    - symbol: Stock symbol (default: WOLF)

    Returns:
    {
      "ok": true,
      "forecast_id": 123,
      "symbol": "WOLF",
      "ts_issued": 1739145600,
      "price_now": 34.13,
      "price_pred_mid": 35.40,
      "price_pred_lo": 33.8,
      "price_pred_hi": 36.7,
      "pnl_pred_mid": -250.50,
      "confidence": 0.75,
      "model": "simple-vol"
    }
    """
    try:
        # Check if price is available
        if symbol == WOLF:
            price, _, _ = get_wolf_price()
        else:
            price = None

        if not price or price <= 0:
            raise HTTPException(
                status_code=503,
                detail="live price unavailable - cannot generate forecast",
            )

        result = _generate_48h_forecast(symbol)

        if not result.get("ok"):
            raise HTTPException(
                status_code=500, detail=result.get("error", "forecast generation failed")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/forecast/score")
async def api_forecast_score(
    body: _ScoreBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    conn = _forecast_db_conn()
    if conn is None:
        return {"ok": False, "error": "db"}
    try:
        conn.row_factory = __import__("sqlite3").Row  # type: ignore
        cur = conn.cursor()
        cur.execute("SELECT * FROM forecast_runs WHERE id=?", (int(body.forecast_id),))
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": "not-found"}
        rowd = dict(row)
        symbol = str(rowd.get("symbol") or WOLF)
        actual = _realized_since(symbol, int(rowd.get("as_of_ts") or 0))
        # trim to through_ts
        actual = [(ts, price) for ts, price in actual if ts <= int(body.through_ts)]
        map, rmse, bias_pct, hit_peak = _compute_forecast_scores(rowd, actual)
        cur.execute(
            "INSERT INTO forecast_scores(forecast_id, scored_through_ts, map, rmse, bias, hit_peak, notes) VALUES(?,?,?,?,?,?,?) ON CONFLICT(forecast_id) DO UPDATE SET scored_through_ts=excluded.scored_through_ts, map=excluded.map, rmse=excluded.rmse, bias=excluded.bias, hit_peak=excluded.hit_peak, notes=excluded.notes",
            (
                int(body.forecast_id),
                int(body.through_ts),
                map,
                rmse,
                bias_pct,
                int(hit_peak),
                "auto",
            ),
        )
        conn.commit()
        return {
            "ok": True,
            "map": map,
            "rmse": rmse,
            "bias_pct": bias_pct,
            "hit_peak": bool(hit_peak),
        }
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"ok": False, "error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.post("/api/forecast/backtest")
async def api_forecast_backtest(
    body: _BacktestBody | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    sym = (body.symbol if body else None) or WOLF
    conn = _forecast_db_conn()
    if conn is None:
        return {"ok": False, "error": "db"}
    try:
        conn.row_factory = __import__("sqlite3").Row  # type: ignore
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM forecast_runs WHERE symbol=? ORDER BY as_of_ts DESC LIMIT 1",
            (sym,),
        )
        row = cur.fetchone()
        if not row:
            return {"ok": False, "error": "no-forecast"}
        rowd = dict(row)
        actual = _realized_since(sym, int(rowd.get("as_of_ts") or 0))
        map, rmse, bias_pct, hit_peak = _compute_forecast_scores(rowd, actual)
        now_ts = int(time.time())
        # Safely coerce forecast id
        fid_any = rowd.get("id")
        try:
            fid = int(fid_any)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return {"ok": False, "error": "invalid-forecast-id"}
        cur.execute(
            "INSERT INTO forecast_scores(forecast_id, scored_through_ts, map, rmse, bias, hit_peak, notes) VALUES(?,?,?,?,?,?,?) ON CONFLICT(forecast_id) DO UPDATE SET scored_through_ts=excluded.scored_through_ts, map=excluded.map, rmse=excluded.rmse, bias=excluded.bias, hit_peak=excluded.hit_peak, notes=excluded.notes",
            (fid, now_ts, map, rmse, bias_pct, int(hit_peak), "backtest"),
        )
        map, rmse, bias_pct, hit_peak = _compute_forecast_scores(rowd, actual)
        now_ts = int(time.time())
        # Safely coerce forecast id and hit_peak with defaults
        fid_any = rowd.get("id")
        try:
            fid = int(fid_any) if fid_any is not None else 0
        except (TypeError, ValueError):
            fid = 0
        cur.execute(
            "INSERT INTO forecast_scores(forecast_id, scored_through_ts, map, rmse, bias, hit_peak, notes) VALUES(?,?,?,?,?,?,?) ON CONFLICT(forecast_id) DO UPDATE SET scored_through_ts=excluded.scored_through_ts, map=excluded.map, rmse=excluded.rmse, bias=excluded.bias, hit_peak=excluded.hit_peak, notes=excluded.notes",
            (
                fid,
                now_ts,
                map,
                rmse,
                bias_pct,
                int(hit_peak) if hit_peak is not None else 0,
                "backtest",
            ),
        )
        # Update rolling stats (last 7/30 for symbol)
        cur.execute(
            "SELECT map, bias, rmse FROM forecast_scores WHERE forecast_id IN (SELECT id FROM forecast_runs WHERE symbol=? ORDER BY as_of_ts DESC LIMIT 30)",
            (sym,),
        )
        arr = [(r[0], r[1], r[2]) for r in cur.fetchall()]
        last7 = arr[:7]

        def _avg(idx: int, A: list[tuple]):
            vals = [float(x[idx]) for x in A if x[idx] is not None]
            return (sum(vals) / len(vals)) if vals else None

        m7, b7, r7 = _avg(0, last7), _avg(1, last7), _avg(2, last7)
        m30, b30, r30 = _avg(0, arr), _avg(1, arr), _avg(2, arr)
        cur.execute(
            "INSERT INTO model_stats(symbol, mape_7, mape_30, bias_7, bias_30, rmse_7, rmse_30, updated_ts) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(symbol) DO UPDATE SET mape_7=excluded.mape_7, mape_30=excluded.mape_30, bias_7=excluded.bias_7, bias_30=excluded.bias_30, rmse_7=excluded.rmse_7, rmse_30=excluded.rmse_30, updated_ts=excluded.updated_ts",
            (sym, m7, m30, b7, b30, r7, r30, now_ts),
        )
        conn.commit()
        return {
            "ok": True,
            "map": map,
            "rmse": rmse,
            "bias_pct": bias_pct,
            "hit_peak": bool(hit_peak),
        }
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"ok": False}
    finally:
        try:
            conn.close()
        except Exception:
            pass


@router.post("/api/forecast/record")
async def api_forecast_record(payload: dict[str, Any]):
    """Store a new 48h forecast for later comparison."""
    try:
        fcst_id = f"fcst-{int(time.time())}-{payload.get('hours', 48)}h"
        FORECAST_STORE[fcst_id] = {
            "symbol": payload.get("symbol", WOLF),
            "as_o": time.time(),
            "hours": payload.get("hours", 48),
            "path_mid": payload.get("path_mid", []),
            "path_lo": payload.get("path_lo", []),
            "path_hi": payload.get("path_hi", []),
            "metadata": payload.get("metadata", {}),
        }
        FORECAST_ACTUALS[fcst_id] = []
        return {"ok": True, "forecast_id": fcst_id}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, 500)


@router.get("/api/forecast/overlay")
async def api_forecast_overlay(symbol: str = WOLF, hours: int = 48):
    """Return predicted vs actual price overlay for charting (MVP JSON schema)."""
    try:
        symbol = symbol.upper()

        # Find most recent forecast
        matching = [
            fid
            for fid, f in FORECAST_STORE.items()
            if f.get("symbol", "").upper() == symbol and f.get("hours") == hours
        ]
        if not matching:
            return {"enabled": False, "reason": "no_forecast"}

        forecast_id = max(matching, key=lambda fid: FORECAST_STORE[fid].get("as_of", 0))
        fcst = FORECAST_STORE[forecast_id]
        actuals = FORECAST_ACTUALS.get(forecast_id, [])

        # Compute basic metrics
        metrics = _compute_forecast_metrics(fcst, actuals)

        return {
            "label": "Ghost Predictions",
            "symbol": symbol,
            "forecast_id": forecast_id,
            "as_o": fcst.get("as_o"),
            "coverage_h": hours,
            "enabled": True,
            "path_predicted": {
                "mid": fcst.get("path_mid", []),
                "lo": fcst.get("path_lo", []),
                "hi": fcst.get("path_hi", []),
            },
            "path_actual": actuals,
            "metrics": metrics,
        }
    except Exception as e:
        return JSONResponse({"enabled": False, "error": str(e)}, 500)


@router.get("/api/forecast/stream")  # Renamed to avoid duplicate with /api/cockpit/stream
async def api_forecast_stream(request: Request):
    """
    Server-Sent Events (SSE) endpoint for real-time two-line overlay updates.
    Emits 'forecast_update' events when prices tick or forecast regenerates.
    """

    async def event_generator():
        last_update = 0
        update_interval = 10  # Update every 10 seconds
        start_time = time.time()

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                print("[SSE forecast] Client disconnected, closing stream")
                break
            # TTL: Close stream after 30 minutes
            if time.time() - start_time > 1800:
                print("[SSE forecast] Stream TTL expired (30 min), closing")
                break
            try:
                now_ts = int(time.time())

                # Only send updates if enough time has passed
                if now_ts - last_update >= update_interval:
                    # Check if we should skip due to anomaly/manual
                    manual_active = STATE.get("manual_price_override") is not None
                    anomaly_active = False
                    try:
                        if isinstance(PRICE_DIAG, dict) and PRICE_DIAG.get("anomaly"):
                            anomaly_active = True
                    except Exception:
                        pass

                    # Build two-line data
                    two_line_data = None
                    if not manual_active and not anomaly_active:
                        try:
                            two_line_data = _build_two_line_forecast(WOLF)
                        except Exception as e:
                            print(f"[SSE] Failed to build two-line overlay: {e}")

                    # Send SSE event
                    if two_line_data:
                        data = json.dumps(
                            {
                                "type": "forecast_update",
                                "ts": now_ts,
                                "data": two_line_data,
                            }
                        )
                        yield f"event: forecast_update\ndata: {data}\n\n"
                    else:
                        # Send heartbeat even if no data
                        yield f"event: heartbeat\ndata: {json.dumps({'ts': now_ts})}\n\n"

                    last_update = now_ts

                # Sleep briefly to avoid tight loop
                await asyncio.sleep(1)

            except Exception as e:
                print(f"[SSE] Error in event generator: {e}")
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/forecast/two_line")
async def api_forecast_two_line(symbol: str = WOLF):
    try:
        data = _build_two_line_forecast(symbol)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/48h/recent")
async def api_forecast_recent(symbol: str = WOLF, limit: int = 10):
    try:
        rows = _recent_forecasts_view(symbol, n=max(1, min(50, int(limit))))
        return {"rows": rows, "symbol": symbol}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/forecast/multi_horizon")
async def api_forecast_multi_horizon(symbol: str = "WOLF"):
    """
    APEX Multi-Horizon Brain: Generate forecasts for 3 time horizons
    - NOWCAST: 1 hour ahead (ultra-short term)
    - SWING: 48 hours ahead (short-term technical)
    - POSITION: 1 week ahead (medium-term trend)

    Returns:
        {
            "symbol": str,
            "timestamp": int,
            "forecasts": {
                "nowcast": {...},
                "swing": {...},
                "position": {...}
            },
            "consensus": {
                "action": str,
                "confidence": float,
                "weighted_return": float,
                "risk_level": str,
                "agreement": str
            }
        }
    """
    from core.multi_horizon_forecaster import get_multi_horizon_forecaster

    if symbol.upper() != WOLF:
        return {"error": f"Symbol {symbol} not supported", "supported": [WOLF]}, 404

    try:
        forecaster = get_multi_horizon_forecaster()
        result = forecaster.forecast_all_horizons(WOLF)
        return result
    except Exception as e:
        LOGGER.error(f"Multi-horizon forecast failed: {e}", exc_info=True)
        return {"error": f"Multi-horizon forecast failed: {str(e)}"}, 500


