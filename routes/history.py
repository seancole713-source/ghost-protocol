"""
Ghost Protocol — V4 History Route
===================================
Full resolved prediction history from ghost_predictions (source of truth).
"""

import logging
import os
import time

from fastapi import APIRouter

router = APIRouter(tags=["history"])
LOGGER = logging.getLogger("ghost.routes.history")


@router.get("/api/v4/history")
async def api_v4_history(days: int = 90, limit: int = 500):
    """
    Full resolved prediction history from ghost_predictions (source of truth).
    Returns actual_price, correct/incorrect, eval_ts — the REAL record.
    This is the same table that drives the accuracy numbers.
    """
    try:
        from core.db_pool import get_sync_connection

        with get_sync_connection() as conn:
            # Safety: reset any aborted transaction state from previous pool user
            try:
                conn.rollback()
            except Exception:
                pass
            cur = conn.cursor()
            cutoff_ts = int(time.time()) - (days * 86400)

            # Try with target_price column for direction re-derivation
            has_target_col = True
            try:
                cur.execute("""
                    SELECT symbol, predicted_direction, current_price, outcome_price,
                           predicted_pct, outcome_pct, correct, predicted_at,
                           checked_at, confidence, target_price
                    FROM ghost_predictions
                    WHERE correct IS NOT NULL
                      AND (eval_version IS NULL OR eval_version NOT LIKE 'skip%%')
                    ORDER BY checked_at DESC NULLS LAST, predicted_at DESC
                    LIMIT %s
                """, (limit,))
            except Exception:
                # FIX (Mar 13, 2026): MUST rollback before retry!
                # Without this, PostgreSQL stays in aborted transaction state
                # and the fallback query fails with "current transaction is aborted"
                conn.rollback()
                has_target_col = False
                cur = conn.cursor()  # get fresh cursor after rollback
                cur.execute("""
                    SELECT symbol, predicted_direction, current_price, outcome_price,
                           predicted_pct, outcome_pct, correct, predicted_at,
                           checked_at, confidence
                    FROM ghost_predictions
                    WHERE correct IS NOT NULL
                      AND (eval_version IS NULL OR eval_version NOT LIKE 'skip%%')
                    ORDER BY checked_at DESC NULLS LAST, predicted_at DESC
                    LIMIT %s
                """, (limit,))
            rows = cur.fetchall()
            cur.close()

        trades = []
        for r in rows:
            symbol = r[0]
            direction = r[1] or "UP"
            entry_price = float(r[2]) if r[2] else 0
            target_price_val = float(r[10]) if has_target_col and len(r) > 10 and r[10] else 0

            # Re-derive direction from prices (fixes 216 historical mismatches)
            if entry_price > 0 and target_price_val > 0:
                direction = "UP" if target_price_val > entry_price else "DOWN"
            elif r[4] and float(r[4]) != 0:  # predicted_pct fallback
                direction = "UP" if float(r[4]) > 0 else "DOWN"

            exit_price = float(r[3]) if r[3] else 0
            expected_move = float(r[4]) if r[4] else 0
            actual_move_pct = float(r[5]) if r[5] else 0
            correct = r[6]  # 1 or 0
            predicted_at = r[7]  # unix timestamp
            eval_ts = r[8]      # unix timestamp (checked_at)
            confidence = float(r[9]) if r[9] else 0

            # Derive market from symbol
            from core.asset_classification import is_crypto_symbol as _hist_is_crypto
            market = "crypto" if _hist_is_crypto(symbol) else "stock"

            # Calculate P&L based on actual move
            pnl = 0
            if entry_price and actual_move_pct:
                is_up = direction == "UP"
                if is_up:
                    pnl = entry_price * (actual_move_pct / 100)
                else:
                    pnl = entry_price * (-actual_move_pct / 100)

            outcome = "win" if correct == 1 else "loss"

            trades.append({
                "symbol": symbol,
                "direction": direction,
                "entry_price": round(entry_price, 6) if entry_price else None,
                "exit_price": round(exit_price, 6) if exit_price else None,
                "pnl": round(pnl, 4),
                "actual_move_pct": round(actual_move_pct, 2),
                "outcome": outcome,
                "confidence": round(confidence, 1),
                "market": market,
                "type": market,
                "predicted_at": predicted_at,
                "resolved_at": eval_ts,
            })

        # Summary stats
        wins = sum(1 for t in trades if t["outcome"] == "win")
        losses = len(trades) - wins
        total_pnl = sum(t["pnl"] for t in trades)
        win_rate = round(wins / len(trades) * 100, 1) if trades else 0

        return {
            "ok": True,
            "trades": trades,
            "count": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
            "source": "ghost_predictions",
        }
    except Exception as e:
        LOGGER.error(f"[V4] History endpoint failed: {e}", exc_info=True)
        return {"ok": False, "trades": [], "count": 0, "error": str(e)}
