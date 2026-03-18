"""
Ghost Protocol — V4 Picks Route
================================
Today's trade picks — reads from ghost_tracked_picks DB table.
Falls back to in-memory predictions if DB is empty.
"""

import logging
import os
import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter

router = APIRouter(tags=["picks"])
LOGGER = logging.getLogger("ghost.routes.picks")


@router.get("/api/v4/picks")
async def api_v4_picks():
    """
    Today's trade picks — reads from ghost_tracked_picks (same source as Telegram).
    Falls back to _LATEST_PREDICTIONS if tracked picks table is empty.
    Every number here matches what the user received in Telegram.
    """
    try:
        now = time.time()
        picks = []
        source = "none"

        # ── PRIMARY: ghost_tracked_picks (same DB Telegram writes to) ──
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            try:
                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT symbol, asset_type, direction, entry_price, target_price,
                               stop_price, prediction_48h, confidence, entry_time,
                               expires_at, status
                        FROM ghost_tracked_picks
                        ORDER BY entry_time DESC
                        LIMIT 50
                    """)
                    rows = cur.fetchall()
                    cur.close()

                for r in rows:
                    symbol = r[0]
                    asset_type = (r[1] or "crypto").lower()
                    db_direction = r[2] or "UP"
                    entry = float(r[3]) if r[3] else 0
                    target = float(r[4]) if r[4] else (float(r[6]) if r[6] else 0)
                    stop = float(r[5]) if r[5] else 0
                    conf = float(r[7]) if r[7] else 0
                    entry_time = r[8]
                    expires_at = r[9]
                    status = (r[10] or "active").lower()

                    if not entry or entry <= 0:
                        continue

                    # Derive direction from entry vs target prices (source of truth)
                    if entry > 0 and target > 0:
                        direction = "UP" if target > entry else "DOWN"
                    else:
                        direction = db_direction

                    is_buy = direction in ("UP", "BUY")

                    # Gain %
                    if is_buy and target and entry:
                        gain_pct = (target - entry) / entry * 100
                    elif not is_buy and target and entry:
                        gain_pct = (entry - target) / entry * 100
                    else:
                        gain_pct = 0

                    # Done by
                    done_by = "--"
                    if expires_at:
                        try:
                            if isinstance(expires_at, (int, float)):
                                exp_dt = datetime.fromtimestamp(expires_at, tz=UTC)
                            else:
                                exp_dt = expires_at
                                if exp_dt.tzinfo is None:
                                    exp_dt = exp_dt.replace(tzinfo=UTC)
                            is_stock = asset_type in ("stock", "stocks")
                            if is_stock and exp_dt.weekday() >= 5:
                                days_ahead = 7 - exp_dt.weekday()
                                exp_dt += timedelta(days=days_ahead)
                            done_by = exp_dt.strftime("%a %b %-d")
                        except Exception:
                            done_by = "--"

                    picks.append({
                        "symbol": symbol,
                        "direction": direction,
                        "confidence": round(conf, 1),
                        "entry_price": round(entry, 6),
                        "target_price": round(target, 6) if target else None,
                        "stop_loss": round(stop, 6) if stop else None,
                        "gain_pct": round(gain_pct, 1),
                        "done_by": done_by,
                        "type": asset_type,
                        "market": asset_type,
                        "status": status,
                        "whitelisted": False,
                    })
                if picks:
                    source = "ghost_tracked_picks"
            except Exception as db_err:
                LOGGER.error(f"[V4] ghost_tracked_picks query failed: {db_err}")

        # ── FALLBACK: in-memory predictions ──
        if not picks:
            try:
                # Import from wolf_app at call time to avoid circular imports
                import wolf_app as _wa
                with _wa._LATEST_PREDICTIONS_LOCK:
                    for symbol, pred in _wa._LATEST_PREDICTIONS.items():
                        if not isinstance(pred, dict):
                            continue
                        conf = pred.get("confidence", 0)
                        if conf < 50:
                            continue
                        direction = pred.get("direction", "FLAT")
                        if direction == "FLAT":
                            continue
                        entry = pred.get("entry_price") or pred.get("price_at_prediction") or pred.get("price", 0)
                        target = pred.get("target_price") or pred.get("take_profit", 0)
                        stop = pred.get("stop_loss", 0)
                        if not entry or entry <= 0:
                            continue
                        if entry > 0 and target > 0:
                            direction = "UP" if target > entry else "DOWN"
                        is_up = direction == "UP"
                        if is_up and target and entry:
                            gain_pct = (target - entry) / entry * 100
                        elif not is_up and target and entry:
                            gain_pct = (entry - target) / entry * 100
                        else:
                            gain_pct = 0
                        horizon_h = pred.get("horizon_h", 48)
                        run_at = pred.get("run_at", now)
                        deadline_ts = run_at + (horizon_h * 3600)
                        deadline_dt = datetime.fromtimestamp(deadline_ts, tz=UTC)
                        market = pred.get("market", "crypto")
                        if market == "stock" and deadline_dt.weekday() >= 5:
                            days_ahead = 7 - deadline_dt.weekday()
                            deadline_dt += timedelta(days=days_ahead)
                        done_by = deadline_dt.strftime("%a %b %-d")
                        age_h = (now - run_at) / 3600
                        if age_h > horizon_h * 2:
                            continue
                        picks.append({
                            "symbol": symbol,
                            "direction": direction,
                            "confidence": round(conf, 1),
                            "entry_price": round(entry, 6),
                            "target_price": round(target, 6) if target else None,
                            "stop_loss": round(stop, 6) if stop else None,
                            "gain_pct": round(gain_pct, 1),
                            "done_by": done_by,
                            "type": market,
                            "market": market,
                            "status": "active",
                            "whitelisted": False,
                        })
                if picks:
                    source = "latest_predictions"
            except Exception:
                pass

        picks.sort(key=lambda p: p["confidence"], reverse=True)

        return {
            "ok": True,
            "picks": picks,
            "count": len(picks),
            "timestamp": int(now),
            "source": source,
        }
    except Exception as e:
        LOGGER.error(f"[V4] Picks endpoint failed: {e}", exc_info=True)
        return {"ok": False, "picks": [], "count": 0, "error": str(e)}
