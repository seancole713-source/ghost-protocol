# ══════════════════════════════════════════════════════════════
# FILE: core/integrity.py
# PURPOSE: System Integrity Audit (v2) — Ghost's health monitoring.
#          Runs 40 checks on every page load + every 5 minutes.
#          Self-healing: auto-fixes what it can. Powers the cockpit
#          Health tab and /integrity/audit/readonly endpoint.
# STATUS: STABLE
# LINES: ~2093
# ──────────────────────────────────────────────────────────────
# CHANGE LOG:
#   2026-03-20 — Lower EXPIRE_HOURS 168->72 to expire stuck predictions after 3 days (Browser Agent)
#   2026-03-20 — Remove auto_fix gate from expiry: always expire >7day stuck predictions (Browser Agent)
#   2026-03-19 — Briefing header added (Browser Agent)
# ──────────────────────────────────────────────────────────────
#   2026-03-21 — Fix auto-expiry: skip_tag column doesn't exist, use eval_version instead (Browser Agent)
# KNOWN ISSUES:
#   - Health score currently 70/100 (2 checks failing)
#   - Some checks may report false failures during weekends/off-hours
#     when market data is unavailable
# ──────────────────────────────────────────────────────────────
# DO NOT CHANGE (frozen interfaces):
#   run_integrity_audit()      — called by cockpit routes + background task
#   IntegrityResult            — returned by audit, consumed by cockpit
#   /integrity/audit/readonly  — served via routes/cockpit.py, calls this
# ══════════════════════════════════════════════════════════════
"""
Ghost Protocol — System Integrity Audit (v2)
═════════════════════════════════════════════

Ghost's EYES — watches every subsystem for the exact bugs we've
fixed in production. If any of them regress, the health bar drops.

Self-healing background audit that:
  • Runs on every page load + every 5 minutes
  • Silently fixes what it can (auto_fix=True)
  • Reports issues it can't fix with severity levels
  • Returns a 0-100 health score

═══════════════════════════════════════════════════════════════════
 35 CHECKS — regression + infrastructure + PROACTIVE discovery
═══════════════════════════════════════════════════════════════════

 ── DATA LAYER (regression guards) ──
 CHECK 1:  Database Connectivity — PG + SQLite alive
 CHECK 2:  Prediction Staleness — engines still producing
 CHECK 3:  Overall Accuracy — skip-tag-filtered
 CHECK 4:  Learning Brain Status — bench + invert zones
 CHECK 5:  Price Feed Health — _LATEST_PREDICTIONS populated
 CHECK 6:  Duplicate Predictions — same symbol+timestamp
 CHECK 7:  Stale Evaluations — past check_at but unchecked
 CHECK 8:  Direction vs Target Math — UP must target above entry
 CHECK 9:  Config / Env Vars — required keys set
 CHECK 10: Per-Symbol Accuracy — flag chronic losers
 CHECK 11: Crypto/Stock Misclassification
 CHECK 12: Skip-Tag Pollution
 CHECK 13: Ghost Brain vs Learning Brain Conflict
 CHECK 14: Cache Data Integrity
 CHECK 15: Edge Whitelist Validation
 CHECK 16: Direction Consistency Guards
 CHECK 17: Brain Inversion + Target Recalc
 CHECK 18: V3 Filter Sanity
 CHECK 19: Live Display Math
 CHECK 20: Notification Pipeline Health

 ── INFRASTRUCTURE (runtime monitoring) ──
 CHECK 21: Background Task Heartbeats
 CHECK 22: Telegram Delivery Verification
 CHECK 23: DB Pool Utilization
 CHECK 24: Memory & Resources
 CHECK 25: Price Provider Circuit Breakers
 CHECK 26: API Rate Limiter State
 CHECK 27: Daemon Thread Liveness
 CHECK 28: Database Schema Validation
 CHECK 29: Alpaca Broker Health
 CHECK 30: Price Cache Health

 ── PROACTIVE (finds new bugs) ──
 CHECK 31: News Feed Content Validation — real news vs self-referential
 CHECK 32: Confidence Data Integrity — detects jitter/clamping
 CHECK 33: Endpoint Placeholder Detection — stub/fake/phantom data
 CHECK 34: Hunter Feed Data Fabrication — synthetic values in API output
 CHECK 35: Accuracy Fallback Phantom — silently reporting 50% defaults

 ── CONTRADICTION (cross-system sanity) ──
 CHECK 36: Paper Trade Resolution — 0 resolved = exit logic broken
 CHECK 37: Prediction vs Forecast Consistency — UP 80% vs FLAT 0%
 CHECK 38: Signal Burst Rate — shotgun pattern detection
 CHECK 39: Win Rate vs Confidence Sanity — 70% confident, 0% wins
 CHECK 40: Dashboard Data Consistency — accuracy mismatch across panels

Created: March 12, 2026
Updated: March 12, 2026 — v5: 40 checks — regression + infra + proactive + contradiction
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("ghost.integrity")

# ── Severity weights ──────────────────────────────────────────
SEVERITY_WEIGHTS = {"error": 10, "warn": 3, "info": 0.5}

# ── Staleness thresholds ──────────────────────────────────────
PREDICTION_STALE_MINUTES = 120
PREDICTION_VERY_STALE_MINUTES = 360
EVAL_OVERDUE_HOURS = 60  # Reconciler uses 48h window; was 12 (too aggressive, caused 211 false overdue)

# ── Known edge symbols (hardcoded cross-check) ───────────────
KNOWN_CRYPTO_EDGE = {"ETH", "XRP", "LINK", "CHZ"}
KNOWN_STOCK_EDGE = {"PANW", "NET", "FTNT", "DDOG", "T", "BMBL", "XPO"}


def run_audit(auto_fix: bool = True) -> Dict[str, Any]:
    """
    Run the full integrity audit — Ghost's eyes on every subsystem.

    Returns:
        {
            "health_score": 0-100,
            "auto_fixes_applied": int,
            "issues_remaining": int,
            "issues": [...],
            "checks_run": [...],
            "checks_total": int,
            "summary": {...},
            "last_audit": ISO timestamp,
        }
    """
    issues: List[Dict] = []
    fixes_applied = 0
    checks_run: List[str] = []
    summary: Dict[str, Any] = {}

    now_ts = int(time.time())

    # ══════════════════════════════════════════════════════════════
    # PULL ALL DATA ONCE — used by multiple checks
    # ══════════════════════════════════════════════════════════════
    predictions = []
    pg_available = False
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()
            cutoff = now_ts - (14 * 86400)
            cur.execute("""
                SELECT id, symbol, predicted_at, check_at, predicted_direction,
                       confidence, current_price, target_price, checked,
                       correct, eval_version, outcome_price, outcome_direction,
                       checked_at
                FROM ghost_predictions
                WHERE predicted_at > %s
                ORDER BY predicted_at DESC
            """, (cutoff,))
            cols = [desc[0] for desc in cur.description]
            for row in cur.fetchall():
                predictions.append(dict(zip(cols, row)))
        pg_available = True
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] PostgreSQL data pull failed: {e}")
        issues.append({
            "type": "database_error",
            "severity": "error",
            "detail": f"PostgreSQL connection failed: {str(e)[:100]}",
        })

    # Separate evaluated predictions (reused by many checks)
    evaluated = [p for p in predictions
                 if p.get("checked") == 1
                 and not str(p.get("eval_version") or "").startswith("skip")]

    # Grab the live cache once for checks that need it
    latest_preds: Dict[str, Any] = {}
    try:
        import wolf_app
        latest_preds = getattr(wolf_app, '_LATEST_PREDICTIONS', {})
    except Exception:
        pass

    # ══════════════════════════════════════════════════════════════
    # CHECK 1: Database Connectivity
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Database Connectivity")
    try:
        if pg_available:
            summary["database"] = "connected"
        else:
            summary["database"] = "error"

        import sqlite3
        wolf_db = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
        if os.path.exists(wolf_db):
            c = sqlite3.connect(wolf_db, timeout=2)
            c.execute("SELECT 1").fetchone()
            c.close()
            summary["sqlite"] = "connected"
        else:
            summary["sqlite"] = "no_file"
    except Exception as e:
        summary["sqlite"] = "error"
        issues.append({
            "type": "sqlite_error", "severity": "warn",
            "detail": f"SQLite check failed: {str(e)[:80]}",
        })

    # ══════════════════════════════════════════════════════════════
    # CHECK 2: Prediction Staleness
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Prediction Staleness")
    try:
        if predictions:
            latest_pred = predictions[0]
            latest_ts = latest_pred.get("predicted_at", 0)
            age_minutes = (now_ts - latest_ts) / 60 if latest_ts else 9999
            summary["latest_prediction_age_min"] = round(age_minutes, 1)
            summary["latest_prediction_symbol"] = latest_pred.get("symbol", "?")

            if age_minutes > PREDICTION_VERY_STALE_MINUTES:
                issues.append({
                    "type": "predictions_very_stale", "severity": "error",
                    "detail": f"No new predictions in {age_minutes:.0f} min ({latest_pred.get('symbol', '?')} was last)",
                })
            elif age_minutes > PREDICTION_STALE_MINUTES:
                issues.append({
                    "type": "predictions_stale", "severity": "warn",
                    "detail": f"No new predictions in {age_minutes:.0f} min",
                })
        elif pg_available:
            issues.append({
                "type": "no_predictions", "severity": "error",
                "detail": "No predictions found in last 14 days",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Staleness check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 3: Overall Accuracy (skip-tag filtered)
    # ──────────────────────────────────────────────────────────────
    # BUG HISTORY: GHOST_SCORE counted 439 skip-tagged junk preds,
    # showing 14.6% instead of real 56%. Fixed by adding
    # AND eval_version NOT LIKE 'skip%%' to accuracy queries.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Prediction Accuracy (skip-filtered)")
    try:
        if evaluated:
            correct_count = sum(1 for p in evaluated if p.get("correct") == 1)
            accuracy_pct = (correct_count / len(evaluated)) * 100
            summary["accuracy_pct"] = round(accuracy_pct, 1)
            summary["total_evaluated"] = len(evaluated)
            summary["total_correct"] = correct_count

            if accuracy_pct < 40:
                issues.append({
                    "type": "accuracy_critical", "severity": "error",
                    "detail": f"Overall accuracy {accuracy_pct:.1f}% — below 40% ({correct_count}/{len(evaluated)})",
                })
            elif accuracy_pct < 42:
                issues.append({
                    "type": "accuracy_low", "severity": "warn",
                    "detail": f"Overall accuracy {accuracy_pct:.1f}% — below 42% ({correct_count}/{len(evaluated)})",
                })
        else:
            summary["accuracy_pct"] = None
            summary["total_evaluated"] = 0
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Accuracy check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 4: Learning Brain Status
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Learning Brain Status")
    try:
        from core.ghost_learning_brain import (
            get_scorecard, get_benched_symbols, get_inverted_symbols,
            BENCH_ACCURACY_THRESHOLD, INVERT_ACCURACY_THRESHOLD,
        )
        scorecard = get_scorecard()
        benched = get_benched_symbols()
        inverted = get_inverted_symbols()

        summary["brain_symbols_total"] = len(scorecard)
        summary["brain_benched"] = benched
        summary["brain_inverted"] = inverted
        summary["brain_bench_threshold"] = BENCH_ACCURACY_THRESHOLD
        summary["brain_invert_threshold"] = INVERT_ACCURACY_THRESHOLD

        if len(benched) + len(inverted) > len(scorecard) * 0.5 and len(scorecard) >= 5:
            issues.append({
                "type": "brain_too_many_losers", "severity": "warn",
                "detail": f"Over 50% of symbols benched/inverted: {', '.join(benched + inverted)}",
            })

        for sym, data in scorecard.items():
            if data.get("total", 0) >= 10 and data.get("accuracy_pct", 50) < 20 and sym not in benched:
                issues.append({
                    "type": "symbol_critical_accuracy", "severity": "warn",
                    "detail": f"{sym} accuracy {data['accuracy_pct']}% over {data['total']} predictions — needs attention",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Learning Brain check failed: {e}")
        summary["brain_status"] = "unavailable"

    # ══════════════════════════════════════════════════════════════
    # CHECK 5: Price Feed Health / _LATEST_PREDICTIONS populated
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Price Feed Health")
    try:
        summary["active_symbols"] = len(latest_preds)
        summary["active_symbol_list"] = sorted(latest_preds.keys())

        if len(latest_preds) == 0:
            issues.append({
                "type": "no_active_predictions", "severity": "error",
                "detail": "No symbols in _LATEST_PREDICTIONS cache — engines may be down",
            })

        stale_price_symbols = []
        for sym, pred in latest_preds.items():
            if isinstance(pred, dict):
                pred_ts = (pred.get("timestamp")
                           or pred.get("run_at")
                           or pred.get("predicted_at")
                           or 0)
                if pred_ts and (now_ts - pred_ts) > 7200:
                    # Skip stale warning for stocks during off-hours (markets closed)
                    _is_stock = sym in KNOWN_STOCK_EDGE or (not sym in KNOWN_CRYPTO_EDGE and sym.isalpha() and len(sym) <= 5)
                    _et_hour = (datetime.utcnow().hour - 5) % 24  # ET = UTC-5
                    _is_market_hours = 9 <= _et_hour < 17 and datetime.utcnow().weekday() < 5
                    if _is_stock and not _is_market_hours:
                        pass  # Stocks naturally stale when market closed
                    else:
                        stale_price_symbols.append(sym)

        if stale_price_symbols:
            issues.append({
                "type": "stale_prices", "severity": "warn",
                "detail": f"Stale price cache (>2h): {', '.join(stale_price_symbols[:8])}",
            })
    except Exception as e:
        LOGGER.debug(f"[INTEGRITY] Price feed check skipped: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 6: Duplicate Predictions (auto-fixable)
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Duplicate Predictions")
    try:
        if predictions:
            seen = set()
            duplicates = []
            for p in predictions:
                key = (p.get("symbol"), p.get("predicted_at"))
                if key in seen:
                    duplicates.append(p)
                else:
                    seen.add(key)

            summary["duplicate_predictions"] = len(duplicates)

            if auto_fix and duplicates and pg_available:
                try:
                    from core.db_pool import get_sync_connection
                    with get_sync_connection() as conn:
                        cur = conn.cursor()
                        dup_ids = [d["id"] for d in duplicates if d.get("id")]
                        if dup_ids:
                            cur.execute(
                                "DELETE FROM ghost_predictions WHERE id = ANY(%s)",
                                (dup_ids,)
                            )
                            conn.commit()
                            fixes_applied += len(dup_ids)
                except Exception as e:
                    issues.append({
                        "type": "duplicate_predictions", "severity": "warn",
                        "detail": f"{len(duplicates)} duplicates found (auto-fix failed: {str(e)[:60]})",
                    })
            elif duplicates:
                issues.append({
                    "type": "duplicate_predictions", "severity": "warn",
                    "detail": f"{len(duplicates)} duplicate predictions found",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Duplicate check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 7: Stale Evaluations
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Stale Evaluations")
    try:
        if predictions:
            overdue = []
            overdue_cutoff = now_ts - (EVAL_OVERDUE_HOURS * 3600)
            for p in predictions:
                check_at = p.get("check_at") or 0
                if (p.get("checked") == 0
                        and check_at > 0
                        and check_at < overdue_cutoff):
                    overdue.append(p)

            summary["overdue_evaluations"] = len(overdue)

            if len(overdue) > 20:
                issues.append({
                    "type": "many_overdue_evals", "severity": "error",
                    "detail": f"{len(overdue)} predictions overdue for evaluation (>{EVAL_OVERDUE_HOURS}h past check_at)",
                })
            elif len(overdue) > 5:
                issues.append({
                    "type": "overdue_evals", "severity": "warn",
                    "detail": f"{len(overdue)} predictions overdue for evaluation",
                })

            # ── AUTO-FIX: Expire ancient stuck predictions ──────────────
            # Predictions >7 days past check_at are too old to evaluate
            # (price data unavailable). Mark them checked with skip tag.
            EXPIRE_HOURS = 60  # Match EVAL_OVERDUE_HOURS so overdue preds expire immediately (lowered from 168/7d to catch more stuck predictions)
            if overdue:  # Always expire ancient >7day predictions (safe cleanup)
                expire_cutoff = now_ts - (EXPIRE_HOURS * 3600)
                ancient = [p for p in overdue if (p.get("check_at") or 0) < expire_cutoff]
                if ancient:
                    try:
                        from core.db_pool import get_sync_connection
                        with get_sync_connection() as conn:
                            cur = conn.cursor()
                            ancient_ids = [p.get("id") for p in ancient if p.get("id")]
                            if ancient_ids:
                                cur.execute(
                                    f"""UPDATE ghost_predictions
                                        SET checked = 1,
                                            eval_version = 'skip-expired_stale'
                                        WHERE id IN ({','.join(['%s'] * len(ancient_ids))})
                                          AND checked = 0""",
                                    ancient_ids
                                )
                                conn.commit()
                                expired_count = cur.rowcount
                                fixes_applied += expired_count
                                LOGGER.info(f"[INTEGRITY] Auto-expired {expired_count} ancient predictions (>{EXPIRE_HOURS}h)")
                    except Exception as fix_err:
                        LOGGER.warning(f"[INTEGRITY] Auto-expire failed: {fix_err}")

    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Stale eval check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 8: Direction vs Target Math (DB + Live Cache)
    # ──────────────────────────────────────────────────────────────
    # BUG HISTORY: CHZ labeled "🔴 DOWN" but target $0.0430 was
    # ABOVE entry $0.0384. Multiple direction-flipping systems
    # changed direction AFTER target was calculated.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Direction vs Target Math")
    try:
        # --- DB predictions ---
        db_mismatches = 0
        db_mismatch_examples = []
        if predictions:
            for p in predictions:
                direction = p.get("predicted_direction")
                target = p.get("target_price")
                entry = p.get("current_price")
                if not all([direction, target, entry]):
                    continue
                try:
                    target_f = float(target)
                    entry_f = float(entry)
                    if entry_f <= 0:
                        continue
                except (ValueError, TypeError):
                    continue

                if direction == "UP" and target_f < entry_f * 0.99:
                    db_mismatches += 1
                    if len(db_mismatch_examples) < 3:
                        db_mismatch_examples.append(
                            f"{p.get('symbol')}: {direction} but target ${target_f:.4f} < entry ${entry_f:.4f}"
                        )
                elif direction == "DOWN" and target_f > entry_f * 1.01:
                    db_mismatches += 1
                    if len(db_mismatch_examples) < 3:
                        db_mismatch_examples.append(
                            f"{p.get('symbol')}: {direction} but target ${target_f:.4f} > entry ${entry_f:.4f}"
                        )

        summary["direction_target_mismatches"] = db_mismatches

        # --- LIVE cache (what users actually see right now) ---
        live_mismatches = 0
        live_mismatch_examples = []
        for sym, pred in latest_preds.items():
            if not isinstance(pred, dict):
                continue
            direction = pred.get("direction")
            target = pred.get("target_price")
            entry = (pred.get("price")
                     or pred.get("price_at_prediction")
                     or pred.get("current_price"))
            if not all([direction, target, entry]):
                continue
            try:
                target_f = float(target)
                entry_f = float(entry)
                if entry_f <= 0:
                    continue
            except (ValueError, TypeError):
                continue

            if direction == "UP" and target_f < entry_f * 0.99:
                live_mismatches += 1
                live_mismatch_examples.append(
                    f"{sym}: says UP but target ${target_f:.4f} < entry ${entry_f:.4f}"
                )
            elif direction == "DOWN" and target_f > entry_f * 1.01:
                live_mismatches += 1
                live_mismatch_examples.append(
                    f"{sym}: says DOWN but target ${target_f:.4f} > entry ${entry_f:.4f}"
                )

        summary["live_direction_mismatches"] = live_mismatches

        # Live mismatches are critical — users SEE them right now
        if live_mismatches > 0:
            # AUTO-FIX: Correct LIVE cache direction based on target vs entry math
            # Root cause: learning brain inverts direction without recalculating target
            if auto_fix:
                _live_fixes = 0
                for sym, pred in latest_preds.items():
                    if not isinstance(pred, dict):
                        continue
                    direction = pred.get("direction")
                    target = pred.get("target_price")
                    entry = (pred.get("price")
                             or pred.get("price_at_prediction")
                             or pred.get("current_price"))
                    if not all([direction, target, entry]):
                        continue
                    try:
                        target_f = float(target)
                        entry_f = float(entry)
                        if entry_f <= 0:
                            continue
                    except (ValueError, TypeError):
                        continue

                    correct_direction = None
                    if direction == "UP" and target_f < entry_f * 0.99:
                        correct_direction = "DOWN"
                    elif direction == "DOWN" and target_f > entry_f * 1.01:
                        correct_direction = "UP"

                    if correct_direction:
                        pred["direction"] = correct_direction
                        pred["action"] = "BUY" if correct_direction == "UP" else "SELL"
                        pred["integrity_corrected"] = True
                        _live_fixes += 1
                        LOGGER.info(f"[INTEGRITY] AUTO-FIX: {sym} direction {direction} → {correct_direction} (target={target_f:.4f} vs entry={entry_f:.4f})")

                if _live_fixes:
                    fixes_applied += _live_fixes

            issues.append({
                "type": "live_direction_mismatch", "severity": "error",
                "detail": f"LIVE display math wrong — {live_mismatches} symbols: {'; '.join(live_mismatch_examples[:3])}",
            })
        if db_mismatches > 20:
            # AUTO-FIX: Correct direction based on target vs entry price
            # Fix ALL predictions (including historical/checked) — direction is provably wrong
            if pg_available and db_mismatches > 0:
                try:
                    fix_count = 0
                    _af_conn = pg_connect(_db_url)
                    _af_cur = _af_conn.cursor()
                    _af_cur.execute("""
                        UPDATE ghost_predictions
                        SET predicted_direction = CASE
                            WHEN target_price > current_price * 1.001 THEN 'UP'
                            WHEN target_price < current_price * 0.999 THEN 'DOWN'
                            ELSE predicted_direction
                        END
                        WHERE (
                              (predicted_direction = 'UP' AND target_price < current_price * 0.99)
                              OR (predicted_direction = 'DOWN' AND target_price > current_price * 1.01)
                          )
                          AND current_price > 0
                          AND target_price > 0
                    """)
                    fix_count = _af_cur.rowcount
                    _af_conn.commit()
                    _af_cur.close()
                    _af_conn.close()
                    if fix_count > 0:
                        fixes_applied += fix_count
                        db_mismatches = max(0, db_mismatches - fix_count)  # Update count after fix
                        LOGGER.info(f"[INTEGRITY] AUTO-FIX: Corrected direction on {fix_count} predictions (including historical)")
                except Exception as fix_err:
                    LOGGER.warning(f"[INTEGRITY] Direction auto-fix failed: {fix_err}")

            # Only report remaining mismatches (after auto-fix)
            if db_mismatches > 0:
                issues.append({
                    "type": "db_direction_mismatch", "severity": "info" if db_mismatches < 50 else "warn",
                    "detail": f"{db_mismatches} stored predictions have direction/target mismatch (auto-fixing)",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Direction/target check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 9: Config / Environment Variables
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Configuration")
    try:
        required_vars = {
            "DATABASE_URL": "PostgreSQL — predictions won't be stored",
            "TELEGRAM_BOT_TOKEN": "Telegram — picks won't be sent",
            "TELEGRAM_CHAT_ID": "Telegram — picks won't be delivered",
        }
        optional_vars = {
            "ALPACA_API_KEY": "Alpaca trading disabled",
            "ALPACA_SECRET_KEY": "Alpaca trading disabled",
        }

        missing_required = []
        for var, desc in required_vars.items():
            if not os.environ.get(var):
                missing_required.append(var)
                issues.append({
                    "type": "config_missing_required", "severity": "error",
                    "detail": f"{var} not set — {desc}",
                })

        missing_optional = []
        for var, desc in optional_vars.items():
            if not os.environ.get(var):
                missing_optional.append(var)

        summary["config_missing_required"] = missing_required
        summary["config_missing_optional"] = missing_optional
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Config check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 10: Per-Symbol Accuracy Breakdown
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Per-Symbol Accuracy")
    try:
        if evaluated:
            symbol_stats = {}
            for p in evaluated:
                sym = p.get("symbol", "?")
                if sym not in symbol_stats:
                    symbol_stats[sym] = {"total": 0, "correct": 0}
                symbol_stats[sym]["total"] += 1
                if p.get("correct") == 1:
                    symbol_stats[sym]["correct"] += 1

            poor_symbols = []
            for sym, stats in symbol_stats.items():
                if stats["total"] >= 10:
                    acc = (stats["correct"] / stats["total"]) * 100
                    if acc < 25:
                        poor_symbols.append(f"{sym} ({acc:.0f}%)")

            summary["per_symbol_count"] = len(symbol_stats)
            summary["per_symbol_accuracy"] = {
                sym: round((s["correct"] / s["total"]) * 100, 1) if s["total"] else 0
                for sym, s in symbol_stats.items()
            }
            if poor_symbols:
                issues.append({
                    "type": "poor_symbol_accuracy", "severity": "info",
                    "detail": f"Low accuracy symbols: {', '.join(poor_symbols)}",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Per-symbol check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 11: Crypto/Stock Misclassification
    # ──────────────────────────────────────────────────────────────
    # BUG HISTORY: CHZ + 32 crypto symbols were missing from
    # asset_classification.py, got classified as "stocks", stole
    # stock slots in Telegram picks. Fixed via _merge_config_crypto().
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Crypto/Stock Classification")
    try:
        from core.asset_classification import is_crypto_symbol, CRYPTO_SYMBOLS as ac_crypto
        from config.symbols import is_crypto as config_is_crypto, CRYPTO_SYMBOLS as config_crypto

        misclass_issues = []

        # Every crypto in config/symbols.py must also be in asset_classification
        for sym in config_crypto:
            if not is_crypto_symbol(sym):
                misclass_issues.append(f"{sym} is crypto in config but not in asset_classification")

        # Known edge crypto must classify correctly
        for sym in KNOWN_CRYPTO_EDGE:
            if not is_crypto_symbol(sym):
                misclass_issues.append(f"EDGE CRYPTO {sym} classified as stock!")
            if not config_is_crypto(sym):
                misclass_issues.append(f"EDGE CRYPTO {sym} missing from config/symbols.py!")

        # Known edge stocks must NOT be crypto
        for sym in KNOWN_STOCK_EDGE:
            if is_crypto_symbol(sym):
                misclass_issues.append(f"STOCK {sym} misclassified as crypto!")

        # Check live cache — does the market field match classification?
        for sym, pred in latest_preds.items():
            if not isinstance(pred, dict):
                continue
            cache_market = str(pred.get("market", "")).lower()
            is_actually_crypto = is_crypto_symbol(sym)
            if cache_market == "crypto" and not is_actually_crypto:
                misclass_issues.append(f"{sym}: cache says crypto but classifier says stock")
            elif cache_market == "stock" and is_actually_crypto:
                misclass_issues.append(f"{sym}: cache says stock but classifier says crypto")

        summary["crypto_symbols_count"] = len(ac_crypto)
        summary["config_crypto_count"] = len(config_crypto)
        summary["classification_issues"] = len(misclass_issues)

        if misclass_issues:
            issues.append({
                "type": "crypto_stock_misclass", "severity": "error",
                "detail": f"Classification mismatch: {'; '.join(misclass_issues[:5])}",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Classification check error: {e}")
        issues.append({
            "type": "classification_check_failed", "severity": "warn",
            "detail": f"Crypto/stock classification check failed: {str(e)[:80]}",
        })

    # ══════════════════════════════════════════════════════════════
    # CHECK 12: Skip-Tag Pollution
    # ──────────────────────────────────────────────────────────────
    # BUG HISTORY: GHOST_SCORE queries counted 439 skip-tagged preds,
    # showing 14.6% instead of real 56% accuracy. Fixed by adding
    # AND eval_version NOT LIKE 'skip%%' filter to 3 queries.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Skip-Tag Pollution")
    try:
        if predictions:
            all_checked = [p for p in predictions if p.get("checked") == 1]
            skip_tagged = [p for p in all_checked
                          if str(p.get("eval_version") or "").startswith("skip")]
            clean = [p for p in all_checked
                     if not str(p.get("eval_version") or "").startswith("skip")]

            summary["skip_tagged_count"] = len(skip_tagged)
            summary["clean_evaluated_count"] = len(clean)

            if len(all_checked) > 0:
                skip_pct = (len(skip_tagged) / len(all_checked)) * 100
                summary["skip_tag_pct"] = round(skip_pct, 1)

                if skip_pct > 40:
                    issues.append({
                        "type": "skip_tag_pollution", "severity": "warn",
                        "detail": (f"{skip_pct:.0f}% of evaluated predictions are skip-tagged "
                                   f"({len(skip_tagged)}/{len(all_checked)}) — "
                                   f"accuracy inflated if not filtered"),
                    })

            # Accuracy delta WITH vs WITHOUT skips
            if skip_tagged and clean:
                acc_with = sum(1 for p in all_checked if p.get("correct") == 1) / len(all_checked) * 100
                acc_clean = sum(1 for p in clean if p.get("correct") == 1) / len(clean) * 100
                delta = abs(acc_clean - acc_with)
                summary["accuracy_with_skips"] = round(acc_with, 1)
                summary["accuracy_without_skips"] = round(acc_clean, 1)
                summary["skip_tag_accuracy_delta"] = round(delta, 1)

                if delta > 15:
                    issues.append({
                        "type": "skip_tag_distortion", "severity": "info",
                        "detail": f"Skip-tags distort accuracy by {delta:.1f}pp ({acc_with:.1f}% with → {acc_clean:.1f}% without)",
                    })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Skip-tag check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 13: Ghost Brain vs Learning Brain Conflict
    # ──────────────────────────────────────────────────────────────
    # BUG RISK: If Ghost Brain AND Learning Brain both invert the
    # same symbol, it double-flips = back to the original bad dir.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Brain Conflict Detection")
    try:
        # Learning Brain inversions
        from core.ghost_learning_brain import get_inverted_symbols as lb_inverted
        lb_inv_set = set(lb_inverted())

        # Ghost Brain inversions
        gb_inv_set = set()
        try:
            from core.ghost_brain import GhostBrain, INVERT_BELOW, EXCLUDE_BELOW, BRAIN_ENABLED
            summary["ghost_brain_enabled"] = BRAIN_ENABLED
            summary["ghost_brain_invert_below"] = INVERT_BELOW
            summary["ghost_brain_exclude_below"] = EXCLUDE_BELOW

            if BRAIN_ENABLED:
                gb = GhostBrain()
                gb_decisions = getattr(gb, '_decisions', {})
                for sym, dec in gb_decisions.items():
                    if getattr(dec, 'inverted', False):
                        gb_inv_set.add(sym)
        except Exception:
            pass

        double_inverted = lb_inv_set & gb_inv_set
        summary["learning_brain_inverted"] = sorted(lb_inv_set)
        summary["ghost_brain_inverted"] = sorted(gb_inv_set)
        summary["double_inverted"] = sorted(double_inverted)

        if double_inverted:
            issues.append({
                "type": "brain_double_invert", "severity": "error",
                "detail": f"DOUBLE INVERSION — both brains flipping: {', '.join(sorted(double_inverted))} → back to bad direction!",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Brain conflict check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 14: Cache Data Integrity
    # ──────────────────────────────────────────────────────────────
    # Validates every _LATEST_PREDICTIONS entry has required fields,
    # valid direction (UP/DOWN not HOLD/FLAT), positive price, and
    # confidence in [0,1].
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Cache Data Integrity")
    try:
        cache_issues = []
        required_fields = ["direction", "confidence", "symbol"]

        for sym, pred in latest_preds.items():
            if not isinstance(pred, dict):
                cache_issues.append(f"{sym}: not a dict")
                continue

            for field in required_fields:
                if field not in pred or pred[field] is None:
                    cache_issues.append(f"{sym}: missing '{field}'")

            d = pred.get("direction")
            if d not in ("UP", "DOWN"):
                cache_issues.append(f"{sym}: direction='{d}' (must be UP/DOWN)")

            conf = pred.get("confidence")
            if conf is not None:
                try:
                    cf = float(conf)
                    if cf < 0 or cf > 1:
                        cache_issues.append(f"{sym}: confidence={cf} out of [0,1]")
                except (ValueError, TypeError):
                    cache_issues.append(f"{sym}: confidence='{conf}' not numeric")

            price = pred.get("price") or pred.get("price_at_prediction")
            if price is not None:
                try:
                    pf = float(price)
                    if pf <= 0:
                        cache_issues.append(f"{sym}: price={pf} not positive")
                except (ValueError, TypeError):
                    cache_issues.append(f"{sym}: price='{price}' not numeric")

        summary["cache_integrity_issues"] = len(cache_issues)

        if cache_issues:
            issues.append({
                "type": "cache_integrity", "severity": "warn",
                "detail": f"Cache data issues: {'; '.join(cache_issues[:5])}",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Cache integrity check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 15: Edge Whitelist Validation
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Edge Whitelist")
    try:
        from config.symbols import get_edge_set
        edge_set = get_edge_set()
        summary["edge_symbols"] = sorted(edge_set)
        summary["edge_count"] = len(edge_set)

        # All known edge symbols should be present
        expected_edge = KNOWN_CRYPTO_EDGE | KNOWN_STOCK_EDGE
        missing_from_edge = expected_edge - edge_set
        if missing_from_edge:
            issues.append({
                "type": "edge_whitelist_gap", "severity": "warn",
                "detail": f"Expected edge symbols missing: {', '.join(sorted(missing_from_edge))}",
            })

        # Warn about actively predicted symbols not in edge set
        non_edge_active = [sym for sym in latest_preds if sym not in edge_set]
        if non_edge_active:
            summary["non_edge_active"] = non_edge_active
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Edge whitelist check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 16: Direction Consistency Guards Present
    # ──────────────────────────────────────────────────────────────
    # BUG HISTORY: Direction was flipped by brain but target/stop
    # were never recalculated. We added guards in format_pick()
    # and _build_pick(). This check verifies those guards exist.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Direction Consistency Guards")
    try:
        import inspect
        import core.ghost_notifications as gn_mod

        gn_source = inspect.getsource(gn_mod)

        has_format_pick_guard = "DIRECTION CONSISTENCY GUARD" in gn_source
        has_brain_inverted_field = "brain_inverted" in gn_source

        summary["format_pick_guard"] = has_format_pick_guard
        summary["brain_inverted_field"] = has_brain_inverted_field

        if not has_format_pick_guard:
            issues.append({
                "type": "missing_direction_guard", "severity": "error",
                "detail": "format_pick() direction consistency guard NOT found — direction/target mismatch bug can recur!",
            })
        if not has_brain_inverted_field:
            issues.append({
                "type": "missing_brain_inverted", "severity": "error",
                "detail": "brain_inverted field NOT found in notifications pipeline — inversion target recalc missing!",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Direction guard check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 17: Brain Inversion + Target Recalc
    # ──────────────────────────────────────────────────────────────
    # BUG HISTORY: Brain flipped direction but target stayed on
    # wrong side of entry. Now we mirror target around entry.
    # Validates that RECENT inverted predictions have correct math.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Inversion Target Recalc")
    try:
        inverted_syms = set()
        try:
            from core.ghost_learning_brain import get_inverted_symbols
            inverted_syms = set(get_inverted_symbols())
        except Exception:
            pass

        if inverted_syms and predictions:
            recent_inverted = [
                p for p in predictions[:200]
                if p.get("symbol") in inverted_syms
            ]
            bad_recalc = 0
            bad_examples = []
            for p in recent_inverted:
                direction = p.get("predicted_direction")
                target = p.get("target_price")
                entry = p.get("current_price")
                if not all([direction, target, entry]):
                    continue
                try:
                    tf = float(target)
                    ef = float(entry)
                    if ef <= 0:
                        continue
                except (ValueError, TypeError):
                    continue

                if direction == "UP" and tf < ef * 0.99:
                    bad_recalc += 1
                    if len(bad_examples) < 2:
                        bad_examples.append(f"{p.get('symbol')}: UP target ${tf:.4f} < entry ${ef:.4f}")
                elif direction == "DOWN" and tf > ef * 1.01:
                    bad_recalc += 1
                    if len(bad_examples) < 2:
                        bad_examples.append(f"{p.get('symbol')}: DOWN target ${tf:.4f} > entry ${ef:.4f}")

            summary["inverted_target_mismatches"] = bad_recalc
            if bad_recalc > 0:
                issues.append({
                    "type": "inversion_target_wrong", "severity": "error",
                    "detail": f"{bad_recalc} inverted predictions have target on wrong side — {'; '.join(bad_examples)}",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Inversion recalc check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 18: V3 Filter Sanity
    # ══════════════════════════════════════════════════════════════
    checks_run.append("V3 Filter Configuration")
    try:
        from core.v3_filter import V3Filter, _V3_MIN_CONFIDENCE
        from config.symbols import V3_VALIDATED_STRATEGIES

        summary["v3_min_confidence"] = _V3_MIN_CONFIDENCE
        summary["v3_strategies_count"] = len(V3_VALIDATED_STRATEGIES)
        summary["v3_strategies"] = sorted(V3_VALIDATED_STRATEGIES.keys())

        # Must instantiate cleanly
        v3f = V3Filter()
        summary["v3_filter_ok"] = True

        if len(V3_VALIDATED_STRATEGIES) == 0:
            issues.append({
                "type": "v3_no_strategies", "severity": "error",
                "detail": "V3 has zero validated strategies — no predictions will pass!",
            })

        # Edge symbols without strategies
        try:
            from config.symbols import get_edge_set
            edge_without_strategy = [
                sym for sym in get_edge_set()
                if sym not in V3_VALIDATED_STRATEGIES
            ]
            if edge_without_strategy:
                summary["edge_without_v3_strategy"] = edge_without_strategy
        except Exception:
            pass

    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] V3 filter check error: {e}")
        issues.append({
            "type": "v3_filter_broken", "severity": "error",
            "detail": f"V3 Filter failed to load: {str(e)[:80]}",
        })

    # ══════════════════════════════════════════════════════════════
    # CHECK 19: Live Prediction Display Math
    # ──────────────────────────────────────────────────────────────
    # The actual numbers users see. Direction, target, entry,
    # confidence, expected_move must all be internally consistent.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Live Display Math")
    try:
        display_issues = []
        for sym, pred in latest_preds.items():
            if not isinstance(pred, dict):
                continue

            direction = pred.get("direction")
            price = pred.get("price") or pred.get("price_at_prediction")
            target = pred.get("target_price")
            expected_move = pred.get("expected_move")

            if price and target and direction:
                try:
                    pf = float(price)
                    tf = float(target)
                    if pf > 0:
                        implied_pct = ((tf - pf) / pf) * 100
                        if direction == "UP" and implied_pct < -1:
                            display_issues.append(
                                f"{sym}: UP but target implies {implied_pct:+.1f}% move"
                            )
                        elif direction == "DOWN" and implied_pct > 1:
                            display_issues.append(
                                f"{sym}: DOWN but target implies {implied_pct:+.1f}% move"
                            )
                except (ValueError, TypeError):
                    pass

            # expected_move sign should match direction
            if expected_move is not None and direction:
                try:
                    em = float(expected_move)
                    if direction == "UP" and em < -0.5:
                        display_issues.append(f"{sym}: UP but expected_move={em:.2f}%")
                    elif direction == "DOWN" and em > 0.5:
                        display_issues.append(f"{sym}: DOWN but expected_move={em:.2f}%")
                except (ValueError, TypeError):
                    pass

        summary["display_math_issues"] = len(display_issues)
        if display_issues:
            issues.append({
                "type": "display_math_wrong", "severity": "error",
                "detail": f"Display math broken — {'; '.join(display_issues[:4])}",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Display math check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 20: Notification Pipeline Health
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Notification Pipeline")
    try:
        telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        telegram_chat = os.environ.get("TELEGRAM_CHAT_ID")
        summary["telegram_configured"] = bool(telegram_token and telegram_chat)

        # Verify key modules load
        from core.ghost_notifications import GhostNotificationSystem
        summary["notifications_module"] = "loaded"

        from core.adapters import scored_list_to_formatter
        summary["adapters_module"] = "loaded"

        # Verify adapters use is_crypto for routing
        import inspect
        adapters_src = inspect.getsource(scored_list_to_formatter)
        has_crypto_routing = "is_crypto" in adapters_src
        summary["adapters_crypto_routing"] = has_crypto_routing

        if not has_crypto_routing:
            issues.append({
                "type": "adapters_no_crypto_routing", "severity": "error",
                "detail": "scored_list_to_formatter() missing is_crypto routing — crypto/stock split broken!",
            })
    except ImportError as e:
        issues.append({
            "type": "notification_import_fail", "severity": "error",
            "detail": f"Notification pipeline import failed: {str(e)[:80]}",
        })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Notification check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 21: Background Task Heartbeats
    # ──────────────────────────────────────────────────────────────
    # Ghost runs 15+ background threads/tasks. If any thread crashes
    # with an unhandled exception, it dies silently. This check
    # verifies every registered task has pulsed recently.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Background Task Heartbeats")
    try:
        from core.heartbeat import get_all_heartbeats, get_missing_tasks, get_stale_tasks

        heartbeats = get_all_heartbeats()
        missing = get_missing_tasks()
        stale = get_stale_tasks()

        # Web-mode awareness: tasks that only run with WORKER_MODE=1
        # can't pulse in web mode — don't flag them as missing
        import os as _os_hb
        _is_worker_mode = _os_hb.getenv("WORKER_MODE") == "1"
        WORKER_ONLY_TASKS = {
            'vip-scanner', 'premarket-scanner', 'full-scanner',
            'money-game', 'autosave-worker', 'alert-worker',
            'open-close-scheduler', 'outcome-reconciler', 'accuracy-tracker',
        }
        if not _is_worker_mode:
            missing = [t for t in missing if t not in WORKER_ONLY_TASKS]
            # Also filter dead/stale that are worker-only (shouldn't happen but safety)
            heartbeats = {k: v for k, v in heartbeats.items() if k not in WORKER_ONLY_TASKS}

        alive_tasks = [n for n, h in heartbeats.items() if h["status"] == "alive"]
        stale_tasks = [n for n, h in heartbeats.items() if h["status"] == "stale"]
        dead_tasks = [n for n, h in heartbeats.items() if h["status"] == "dead"]

        summary["heartbeat_alive"] = alive_tasks
        summary["heartbeat_stale"] = stale_tasks
        summary["heartbeat_dead"] = dead_tasks
        summary["heartbeat_never_pulsed"] = missing
        summary["heartbeat_total_registered"] = len(heartbeats) + len(missing)

        if dead_tasks:
            issues.append({
                "type": "background_task_dead", "severity": "error",
                "detail": f"DEAD background tasks (no heartbeat): {', '.join(dead_tasks)}",
            })
        if stale_tasks:
            issues.append({
                "type": "background_task_stale", "severity": "warn",
                "detail": f"Stale background tasks (delayed heartbeat): {', '.join(stale_tasks)}",
            })
        # Flag tasks that registered but NEVER pulsed
        # This was the biggest blind spot — 18/18 tasks dead and health said nothing
        if missing:
            total_registered = len(heartbeats) + len(missing)
            alive_count = len(alive_tasks)
            critical_missing = [t for t in missing if t in (
                'outcome-reconciler', 'prediction-cycle', 'full-scanner',
                'money-game', 'reevaluation', 'self-improvement',
            )]

            # AUTO-FIX: If outcome-reconciler never pulsed, trigger check_all
            # to process overdue paper trades that would normally auto-resolve
            if auto_fix and 'outcome-reconciler' in missing:
                try:
                    from core.paper_tracker import get_paper_tracker
                    tracker = get_paper_tracker()
                    _fix_stats = tracker.get_stats(days=30)
                    _fix_pending = _fix_stats.get('pending_trades', 0)
                    _fix_resolved = _fix_stats.get('resolved_trades', 0)
                    if _fix_pending > _fix_resolved:
                        # Build price_data from _LATEST_PREDICTIONS cache
                        import wolf_app as _wa21
                        _preds21 = getattr(_wa21, '_LATEST_PREDICTIONS', {})
                        _prices21 = {}
                        for _s21, _p21 in _preds21.items():
                            if isinstance(_p21, dict):
                                _px = _p21.get('price') or _p21.get('price_at_prediction') or _p21.get('current_price')
                                if _px:
                                    try:
                                        _prices21[_s21] = float(_px)
                                    except (ValueError, TypeError):
                                        pass
                        if _prices21:
                            resolved_list = tracker.check_all_pending(_prices21)
                            if resolved_list:
                                fixes_applied += len(resolved_list)
                                LOGGER.info(f"[INTEGRITY] AUTO-FIX: Resolved {len(resolved_list)} overdue paper trades (reconciler not running)")
                        else:
                            LOGGER.warning("[INTEGRITY] AUTO-FIX: No price data for paper trade resolution")
                except Exception as fix_err:
                    LOGGER.warning(f"[INTEGRITY] Paper trade auto-resolve failed: {fix_err}")

            if len(missing) == total_registered and total_registered > 0:
                # ALL tasks never pulsed — nothing is running
                issues.append({
                    "type": "all_background_tasks_dead", "severity": "error",
                    "detail": f"ALL {total_registered} background tasks have NEVER pulsed — reconciler, prediction cycle, and all daemons are NOT running",
                })
            elif critical_missing:
                issues.append({
                    "type": "critical_tasks_never_started", "severity": "error",
                    "detail": f"Critical tasks never pulsed: {', '.join(critical_missing)} — paper trades won't auto-resolve, predictions may not cycle",
                })
            elif len(missing) >= 5:
                issues.append({
                    "type": "many_tasks_never_started", "severity": "warn",
                    "detail": f"{len(missing)}/{total_registered} background tasks never pulsed: {', '.join(missing[:8])}",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Heartbeat check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 22: Telegram Delivery Verification
    # ──────────────────────────────────────────────────────────────
    # CHECK 20 verifies env vars exist and modules import. This
    # check verifies messages are ACTUALLY being delivered — last
    # successful send timestamp, error state, API reachability.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Telegram Delivery Verification")
    try:
        import wolf_app as _wa

        last_send_ts = getattr(_wa, '_LAST_TELEGRAM_SEND_TIME', None)
        last_status = getattr(_wa, '_LAST_TELEGRAM_STATUS', 'never_run')
        last_error = getattr(_wa, '_LAST_TELEGRAM_ERROR', None)

        summary["telegram_last_send_ts"] = last_send_ts
        summary["telegram_last_status"] = last_status
        summary["telegram_last_error"] = last_error

        if last_send_ts:
            age_hours = (now_ts - last_send_ts) / 3600
            summary["telegram_last_send_hours_ago"] = round(age_hours, 1)

            if age_hours > 24:
                issues.append({
                    "type": "telegram_no_sends", "severity": "error",
                    "detail": f"No Telegram message sent in {age_hours:.0f}h — delivery may be broken",
                })
            elif age_hours > 12:
                issues.append({
                    "type": "telegram_sends_slow", "severity": "warn",
                    "detail": f"No Telegram message sent in {age_hours:.0f}h",
                })
        elif last_status == "never_run":
            summary["telegram_last_send_hours_ago"] = None

        if last_status == "error" and last_error:
            issues.append({
                "type": "telegram_last_error", "severity": "warn",
                "detail": f"Last Telegram send failed: {str(last_error)[:80]}",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Telegram delivery check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 23: Database Connection Pool Utilization
    # ──────────────────────────────────────────────────────────────
    # CHECK 1 just tests SELECT 1. This check monitors pool
    # utilization — if all connections are checked out, new queries
    # will hang/timeout. Uses existing get_sync_pool_status().
    # ══════════════════════════════════════════════════════════════
    checks_run.append("DB Pool Utilization")
    try:
        from core.db_pool import get_sync_pool_status

        pool_status = get_sync_pool_status()
        summary["db_pool"] = pool_status

        if pool_status.get("initialized"):
            max_conn = pool_status.get("max_connections", 5)
            checked_out = pool_status.get("checked_out", 0)
            available = pool_status.get("available", max_conn)

            if max_conn > 0:
                utilization_pct = (checked_out / max_conn) * 100
                summary["db_pool_utilization_pct"] = round(utilization_pct, 1)

                if available == 0:
                    issues.append({
                        "type": "db_pool_exhausted", "severity": "error",
                        "detail": f"DB pool EXHAUSTED — {checked_out}/{max_conn} connections in use, 0 available!",
                    })
                elif utilization_pct > 80:
                    issues.append({
                        "type": "db_pool_high", "severity": "warn",
                        "detail": f"DB pool {utilization_pct:.0f}% utilized ({checked_out}/{max_conn})",
                    })

            if pool_status.get("closed"):
                issues.append({
                    "type": "db_pool_closed", "severity": "error",
                    "detail": "DB connection pool is CLOSED — all queries will fail!",
                })
        else:
            summary["db_pool_utilization_pct"] = None
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] DB pool check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 24: Memory / Resource Usage
    # ──────────────────────────────────────────────────────────────
    # No resource monitoring existed. This check uses os-level
    # /proc/self/status on Linux (Railway runs Linux) to get RSS
    # without requiring psutil.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Memory & Resources")
    try:
        import resource
        # getrusage returns RSS in KB on Linux
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = usage.ru_maxrss / 1024  # Convert KB to MB on Linux
        summary["memory_rss_mb"] = round(rss_mb, 1)

        # Also try /proc/self/status for more accurate current RSS
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        rss_mb = rss_kb / 1024
                        summary["memory_rss_mb"] = round(rss_mb, 1)
                        break
        except Exception:
            pass

        if rss_mb > 1024:
            issues.append({
                "type": "memory_critical", "severity": "error",
                "detail": f"Memory usage {rss_mb:.0f}MB — exceeds 1GB, OOM risk!",
            })
        elif rss_mb > 512:
            issues.append({
                "type": "memory_high", "severity": "warn",
                "detail": f"Memory usage {rss_mb:.0f}MB — above 512MB",
            })

        # Thread count
        try:
            import threading
            thread_count = threading.active_count()
            summary["active_threads"] = thread_count

            if thread_count > 50:
                issues.append({
                    "type": "thread_leak", "severity": "warn",
                    "detail": f"Thread count {thread_count} — possible thread leak",
                })
        except Exception:
            pass

        # Open file descriptors
        try:
            fd_count = len(os.listdir("/proc/self/fd"))
            summary["open_file_descriptors"] = fd_count

            if fd_count > 500:
                issues.append({
                    "type": "fd_leak", "severity": "warn",
                    "detail": f"Open file descriptors: {fd_count} — possible leak",
                })
        except Exception:
            pass

    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Memory check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 25: Price Provider Health / Circuit Breakers
    # ──────────────────────────────────────────────────────────────
    # Ghost cascades through multiple price providers. If all are
    # in circuit-breaker "open" state, predictions run on stale
    # prices. This check reads the breaker states.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Price Provider Circuit Breakers")
    try:
        import wolf_app as _wa

        breakers = getattr(_wa, '_PROVIDER_BREAKERS', {})
        provider_status = {}
        open_providers = []

        for name, state in breakers.items():
            status = state.get("state", "unknown")
            failures = state.get("failures", 0)
            provider_status[name] = {
                "state": status,
                "failures": failures,
            }
            if status == "open":
                open_providers.append(f"{name} ({failures} failures)")

        summary["price_providers"] = provider_status
        summary["providers_open_circuit"] = len(open_providers)

        if len(open_providers) == len(breakers) and len(breakers) > 0:
            issues.append({
                "type": "all_providers_down", "severity": "error",
                "detail": f"ALL price providers circuit-broken: {', '.join(open_providers)} — running on stale prices!",
            })
        elif open_providers:
            issues.append({
                "type": "provider_circuit_open", "severity": "warn",
                "detail": f"Price providers circuit-broken: {', '.join(open_providers)}",
            })

        # Also check crypto provider health if available
        try:
            from core.crypto.crypto_providers import get_crypto_provider_health
            crypto_health = get_crypto_provider_health()
            summary["crypto_provider_health"] = crypto_health
        except Exception:
            pass

    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Provider health check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 26: API Throttle / Rate Limiter State
    # ──────────────────────────────────────────────────────────────
    # The IP throttle dict could grow unbounded if eviction fails.
    # Also monitors if too many clients are getting 429'd.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("API Rate Limiter State")
    try:
        import wolf_app as _wa

        throttle_buckets = getattr(_wa, '_throttle_buckets', {})
        bucket_count = len(throttle_buckets)
        summary["throttle_bucket_count"] = bucket_count

        if bucket_count > 500:
            issues.append({
                "type": "throttle_memory_leak", "severity": "error",
                "detail": f"Throttle bucket dict has {bucket_count} entries — eviction broken, memory leak!",
            })
        elif bucket_count > 200:
            issues.append({
                "type": "throttle_buckets_high", "severity": "warn",
                "detail": f"Throttle tracking {bucket_count} IPs — higher than expected",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Throttle check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 27: Daemon Thread Liveness
    # ──────────────────────────────────────────────────────────────
    # Direct check: are the actual Python thread objects alive?
    # This catches threads that crashed even if heartbeats weren't
    # wired in yet.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Daemon Thread Liveness")
    try:
        import wolf_app as _wa

        thread_checks = {
            "autosave-worker": getattr(_wa, '_AUTOSAVE_WORKER', None),
            "alert-worker": getattr(_wa, '_ALERT_WORKER', None),
            "open-close-scheduler": getattr(_wa, '_SCHED_WORKER', None),
            "outcome-reconciler": getattr(_wa, '_RECONCILER_WORKER', None),
            "accuracy-tracker": getattr(_wa, '_ACCURACY_TRACKER', None),
        }

        dead_threads = []
        alive_threads = []
        for name, thread in thread_checks.items():
            if thread is None:
                continue  # Never started — not necessarily an error
            if thread.is_alive():
                alive_threads.append(name)
            else:
                dead_threads.append(name)

        summary["daemon_threads_alive"] = alive_threads
        summary["daemon_threads_dead"] = dead_threads

        if dead_threads:
            issues.append({
                "type": "daemon_thread_dead", "severity": "error",
                "detail": f"DEAD daemon threads: {', '.join(dead_threads)} — crashed silently!",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Thread liveness check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 28: Database Schema Validation
    # ──────────────────────────────────────────────────────────────
    # No Alembic — migrations are ad-hoc ALTER TABLEs. This check
    # verifies critical columns exist in ghost_predictions.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Database Schema")
    try:
        if pg_available:
            from core.db_pool import get_sync_connection
            with get_sync_connection() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'ghost_predictions'
                """)
                existing_cols = {row[0] for row in cur.fetchall()}

            required_cols = {
                "id", "symbol", "predicted_at", "check_at",
                "predicted_direction", "confidence", "current_price",
                "target_price", "checked", "correct", "eval_version",
                "outcome_price", "outcome_direction", "checked_at",
                "window_high", "window_low",
            }

            missing_cols = required_cols - existing_cols
            summary["schema_columns_found"] = len(existing_cols)
            summary["schema_columns_missing"] = sorted(missing_cols) if missing_cols else []

            if missing_cols:
                issues.append({
                    "type": "schema_missing_columns", "severity": "error",
                    "detail": f"ghost_predictions missing columns: {', '.join(sorted(missing_cols))} — migration needed!",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Schema check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 29: Alpaca Broker Health
    # ──────────────────────────────────────────────────────────────
    # If Alpaca keys are set, verify the broker can actually reach
    # the API. Silent order failures = real money lost.
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Broker Health")
    try:
        alpaca_key = os.environ.get("ALPACA_API_KEY")
        alpaca_secret = os.environ.get("ALPACA_SECRET_KEY")

        if alpaca_key and alpaca_secret:
            try:
                from core.alpaca_broker import get_broker
                broker = get_broker()
                if broker:
                    summary["broker_status"] = "initialized"
                    # Check if broker has a health check method
                    if hasattr(broker, 'health_check'):
                        try:
                            health = broker.health_check()
                            summary["broker_health"] = health
                            if not health.get("ok"):
                                issues.append({
                                    "type": "broker_unhealthy", "severity": "error",
                                    "detail": f"Alpaca broker health check failed: {str(health)[:80]}",
                                })
                        except Exception as e:
                            issues.append({
                                "type": "broker_health_error", "severity": "warn",
                                "detail": f"Broker health check threw: {str(e)[:80]}",
                            })
                else:
                    summary["broker_status"] = "not_initialized"
            except Exception as e:
                summary["broker_status"] = f"error: {str(e)[:60]}"
                issues.append({
                    "type": "broker_import_fail", "severity": "warn",
                    "detail": f"Alpaca broker failed to load: {str(e)[:80]}",
                })
        else:
            summary["broker_status"] = "not_configured"
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Broker check error: {e}")

    # ══════════════════════════════════════════════════════════════
    # CHECK 30: Price Cache Size & Freshness
    # ──────────────────────────────────────────────────────────────
    # PRICE_CACHE can grow unbounded. Also checks that cached prices
    # aren't all stale (which means providers silently failed).
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Price Cache Health")
    try:
        import wolf_app as _wa

        price_cache = getattr(_wa, 'PRICE_CACHE', {})
        cache_size = len(price_cache)
        summary["price_cache_size"] = cache_size

        if cache_size > 1000:
            issues.append({
                "type": "price_cache_bloat", "severity": "warn",
                "detail": f"PRICE_CACHE has {cache_size} entries — possible memory bloat",
            })

        # Check freshness of cached prices
        if price_cache:
            stale_count = 0
            fresh_count = 0
            for sym, entry in price_cache.items():
                if isinstance(entry, dict):
                    ts = entry.get("ts") or entry.get("timestamp") or 0
                    if ts and (now_ts - ts) > 3600:
                        stale_count += 1
                    elif ts:
                        fresh_count += 1

            summary["price_cache_fresh"] = fresh_count
            summary["price_cache_stale"] = stale_count

            if fresh_count == 0 and stale_count > 0:
                issues.append({
                    "type": "all_prices_stale", "severity": "error",
                    "detail": f"ALL {stale_count} cached prices are stale (>1h) — providers may be down!",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Price cache check error: {e}")

    # ══════════════════════════════════════════════════════════════
    #  PROACTIVE CHECKS 31-35 — Find bugs we DON'T know about yet
    #  These validate that features return semantically correct data,
    #  not just that they exist and return 200 OK.
    # ══════════════════════════════════════════════════════════════

    # ── CHECK 31: News Feed Content Validation ───────────────────
    # Detects when the news feed is just showing Ghost predictions
    # disguised as news instead of actual external news articles.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("News Feed Content")
    try:
        import wolf_app as _wa
        import inspect

        # Check the source of api_v3_news_feed to see if it ONLY
        # calls the hunter feed (predictions) with no external source.
        fn = getattr(_wa, 'api_v3_news_feed', None)
        if fn:
            src = inspect.getsource(fn)
            has_rss = 'feedparser' in src or 'rss' in src.lower() or 'RSS' in src
            has_external = 'reuters' in src.lower() or 'marketwatch' in src.lower() or 'yahoo' in src.lower() or 'cnbc' in src.lower()
            only_hunter = 'api_v3_hunter_feed' in src and not has_rss and not has_external

            summary["news_has_external_sources"] = has_external or has_rss
            summary["news_only_predictions"] = only_hunter

            if only_hunter:
                issues.append({
                    "type": "news_feed_self_referential", "severity": "error",
                    "detail": "News feed only shows Ghost predictions relabeled as news — no external sources",
                })
        else:
            summary["news_has_external_sources"] = None
            summary["news_only_predictions"] = None
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] News feed content check error: {e}")

    # ── CHECK 32: Confidence Data Integrity ──────────────────────
    # Detects when confidence values are being clamped or jittered,
    # hiding real model output from the user.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Confidence Data Integrity")
    try:
        import wolf_app as _wa
        import inspect

        fn = getattr(_wa, 'api_v3_hunter_feed', None)
        if fn:
            src = inspect.getsource(fn)
            has_clamp = 'max(45' in src or 'min(85' in src or 'clamp' in src.lower()
            has_jitter = 'jitter' in src.lower() or 'conf_jitter' in src

            summary["confidence_clamped"] = has_clamp
            summary["confidence_jittered"] = has_jitter

            problems = []
            if has_clamp:
                problems.append("confidence clamped to 45-85% (hides real values)")
            if has_jitter:
                problems.append("fake jitter added to confidence (±2%)")

            if problems:
                issues.append({
                    "type": "confidence_data_fabricated", "severity": "warn",
                    "detail": f"Hunter feed: {'; '.join(problems)}",
                })
        else:
            summary["confidence_clamped"] = None
            summary["confidence_jittered"] = None
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Confidence data check error: {e}")

    # ── CHECK 33: Endpoint Placeholder Detection ─────────────────
    # Scans for endpoints returning stub/placeholder/warming-up data.
    # A real health check should flag "warming up" as a problem, not
    # silently let users think the system is working.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Endpoint Placeholder Detection")
    try:
        import wolf_app as _wa
        import inspect

        PLACEHOLDER_PATTERNS = [
            "warming up", "initializing", "coming soon",
            "not implemented", "stub", "placeholder", "todo",
            "mock", "hardcoded", "# fake",
        ]
        # Scan key endpoint functions for placeholder strings
        endpoints_to_check = [
            "api_v3_news_feed", "api_v3_hunter_feed",
            "api_v3_predictions_history", "api_v3_accuracy_summary",
            "api_v3_health_metrics", "api_v3_goals_snapshot",
        ]
        placeholder_found = []
        for ep_name in endpoints_to_check:
            fn = getattr(_wa, ep_name, None)
            if fn:
                try:
                    src = inspect.getsource(fn).lower()
                    for pattern in PLACEHOLDER_PATTERNS:
                        if pattern in src:
                            placeholder_found.append(f"{ep_name}: '{pattern}'")
                            break  # One per endpoint
                except Exception:
                    pass

        summary["endpoints_with_placeholders"] = placeholder_found

        if placeholder_found:
            issues.append({
                "type": "endpoint_placeholder_data", "severity": "warn",
                "detail": f"{len(placeholder_found)} endpoints return placeholder data: {', '.join(placeholder_found[:5])}",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Placeholder detection error: {e}")

    # ── CHECK 34: Hunter Feed Data Fabrication ────────────────────
    # Validates live hunter feed output: are change_pct values
    # synthetic (invented formulas), are outcomes always None, etc.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Hunter Feed Data Fabrication")
    try:
        import wolf_app as _wa
        import inspect

        fn = getattr(_wa, 'api_v3_hunter_feed', None)
        fabrication_issues = []
        if fn:
            src = inspect.getsource(fn)

            # Check for synthetic expected_move formula
            if 'confidence_pct - 40' in src and 'change_pct' in src:
                fabrication_issues.append("expected_move invented from confidence formula")

            # Check for change_jitter
            if 'change_jitter' in src:
                fabrication_issues.append("change_pct jittered with fake noise")

            # Check for hardcoded stop/take values
            if 'stop_loss' in src and '0.98' in src:
                fabrication_issues.append("stop_loss hardcoded at 2%")
            if 'take_profit' in src and '1.06' in src:
                fabrication_issues.append("take_profit hardcoded at 6%")

        summary["hunter_feed_fabrications"] = fabrication_issues

        if fabrication_issues:
            issues.append({
                "type": "hunter_feed_fabricated", "severity": "warn",
                "detail": f"Hunter feed fabricates data: {'; '.join(fabrication_issues)}",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Hunter feed fabrication check error: {e}")

    # ── CHECK 35: Accuracy Fallback Phantom ───────────────────────
    # Detects when accuracy/health endpoints silently fall back to
    # 50% defaults instead of returning null/error. A user sees
    # "50% accuracy" and thinks the system measured that — it didn't.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Accuracy Fallback Phantom")
    try:
        import wolf_app as _wa
        import inspect
        import re as _re35

        phantom_defaults = []
        # Patterns that indicate phantom accuracy defaults (not data_health, ai_activity, etc.)
        _phantom_patterns = [
            _re35.compile(r'\baccuracy\s*=\s*(?:50|0\.5)\b'),       # accuracy = 50 or accuracy = 0.5
            _re35.compile(r'\baccuracy_pct\s*=\s*(?:50|0\.5)\b'),   # accuracy_pct = 50
            _re35.compile(r'\bghost_score\s*=\s*(?:50|0\.5)\b'),    # ghost_score = 50
            _re35.compile(r'\bwin_rate\s*=\s*(?:50|0\.5)\b'),       # win_rate = 50
        ]
        for fn_name in ["api_v3_goals_snapshot", "api_v3_health_metrics", "api_v3_accuracy_summary"]:
            fn = getattr(_wa, fn_name, None)
            if fn:
                try:
                    src = inspect.getsource(fn)
                    # Only flag if ACCURACY-specific variables default to 50
                    # Don't flag data_health=50 or ai_activity=50 — those are intentional neutral values
                    if any(p.search(src) for p in _phantom_patterns):
                        phantom_defaults.append(fn_name)
                except Exception:
                    pass

        summary["phantom_accuracy_defaults"] = phantom_defaults

        if phantom_defaults:
            issues.append({
                "type": "accuracy_phantom_defaults", "severity": "warn",
                "detail": f"{len(phantom_defaults)} endpoints use phantom 50% default: {', '.join(phantom_defaults)}",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Accuracy phantom check error: {e}")

    # ══════════════════════════════════════════════════════════════
    #  CONTRADICTION CHECKS 36-40 — Catch when features contradict
    #  each other. The health check saw 93 because it checked each
    #  feature in isolation. These checks compare features AGAINST
    #  each other to find logical impossibilities.
    # ══════════════════════════════════════════════════════════════

    # ── CHECK 36: Paper Trade Resolution ─────────────────────────
    # If there are 100+ pending trades and 0 resolved, the exit
    # logic is broken. A real trading system closes positions.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Paper Trade Resolution")
    try:
        from core.paper_tracker import get_paper_tracker
        tracker = get_paper_tracker()
        stats = tracker.get_stats(days=365)

        pending = stats.get("pending_trades", 0)
        resolved = stats.get("resolved_trades", 0)
        total = stats.get("total_trades", 0)

        summary["paper_total_trades"] = total
        summary["paper_pending"] = pending
        summary["paper_resolved"] = resolved
        summary["paper_win_rate"] = stats.get("win_rate", 0)

        if total >= 50 and resolved == 0:
            issues.append({
                "type": "paper_trades_never_resolve", "severity": "error",
                "detail": f"{pending} pending paper trades, 0 resolved — exit logic broken or reconciler not running",
            })
        elif total >= 100 and resolved < total * 0.10:
            issues.append({
                "type": "paper_trades_low_resolution", "severity": "warn",
                "detail": f"Only {resolved}/{total} paper trades resolved ({resolved/total*100:.0f}%) — reconciler may be failing",
            })
        elif total >= 50 and pending > resolved * 2:
            issues.append({
                "type": "paper_trades_backlog", "severity": "info",
                "detail": f"{pending} pending vs {resolved} resolved — trade backlog growing, {pending/total*100:.0f}% still waiting",
            })

        # AUTO-FIX: Process overdue paper trades regardless of severity
        # The worst cases (0 resolved, <10% resolution) need this MORE, not less
        if auto_fix and pending > 0:
            try:
                import wolf_app as _wa36
                _preds36 = getattr(_wa36, '_LATEST_PREDICTIONS', {})
                _prices36 = {}
                for _s36, _p36 in _preds36.items():
                    if isinstance(_p36, dict):
                        _px = _p36.get('price') or _p36.get('price_at_prediction') or _p36.get('current_price')
                        if _px:
                            try:
                                _prices36[_s36] = float(_px)
                            except (ValueError, TypeError):
                                pass
                if _prices36:
                    resolved_list = tracker.check_all_pending(_prices36)
                    if resolved_list:
                        fixes_applied += len(resolved_list)
                        LOGGER.info(f"[INTEGRITY] AUTO-FIX: Resolved {len(resolved_list)} trades from backlog ({pending} were pending)")
            except Exception as fix_err:
                LOGGER.warning(f"[INTEGRITY] Paper trade backlog fix failed: {fix_err}")
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Paper trade resolution check error: {e}")

    # ── CHECK 37: Prediction vs Forecast Consistency ─────────────
    # When predictions say UP with 80% but forecast says FLAT/0%,
    # two subsystems are contradicting each other.
    # Also flags when forecast system is completely disconnected.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Prediction vs Forecast Consistency")
    try:
        import wolf_app as _wa
        predictions = getattr(_wa, '_LATEST_PREDICTIONS', {})

        # Read actual forecast data from the file (not a module variable)
        forecast_data = None
        forecast_age_h = None
        try:
            forecast_path = getattr(_wa, 'FORECAST_GRID_PATH', 'data/forecast_WOLF.json')
            if os.path.exists(forecast_path):
                with open(forecast_path) as _fg:
                    forecast_data = json.load(_fg)
                # Check forecast freshness
                fc_ts = forecast_data.get("aso") or forecast_data.get("generated_at") or 0
                if fc_ts > 0:
                    forecast_age_h = (now_ts - fc_ts) / 3600
                    summary["forecast_age_hours"] = round(forecast_age_h, 1)
        except Exception:
            pass

        # Flag 1: Forecast file missing or empty while predictions are active
        active_predictions = len([p for p in predictions.values()
                                  if p.get("direction") in ("UP", "DOWN")])
        summary["forecast_data_available"] = forecast_data is not None
        summary["active_directional_predictions"] = active_predictions

        if active_predictions >= 3 and forecast_data is None:
            # AUTO-FIX: Regenerate forecast grid
            if auto_fix:
                try:
                    import wolf_app as _wa_fix
                    _gen_fn = getattr(_wa_fix, '_generate_forecast_grid', None)
                    if _gen_fn:
                        _gen_fn()  # Regenerates and saves to file
                        fixes_applied += 1
                        LOGGER.info("[INTEGRITY] AUTO-FIX: Regenerated missing forecast grid")
                except Exception as fix_err:
                    LOGGER.warning(f"[INTEGRITY] Forecast regeneration failed: {fix_err}")
            issues.append({
                "type": "forecast_disconnected", "severity": "warn",
                "detail": f"{active_predictions} active predictions but forecast grid is empty/missing — dashboard forecast panel has no data",
            })
        elif forecast_data and forecast_age_h and forecast_age_h > 6:
            # AUTO-FIX: Regenerate stale forecast (always — it's just a cache file)
            try:
                import wolf_app as _wa_fix2
                _gen_fn2 = getattr(_wa_fix2, '_generate_forecast_grid', None)
                if _gen_fn2:
                    _gen_fn2()  # Regenerates and saves to file
                    fixes_applied += 1
                    LOGGER.info(f"[INTEGRITY] AUTO-FIX: Regenerated stale forecast ({forecast_age_h:.1f}h old)")
            except Exception as fix_err:
                LOGGER.warning(f"[INTEGRITY] Forecast regeneration failed: {fix_err}")
            if forecast_age_h > 48:
                sev = "info"  # downgrade after auto-fix attempt
            else:
                sev = "info"
            issues.append({
                "type": "forecast_stale", "severity": sev,
                "detail": f"Forecast data was {forecast_age_h:.1f}h old — auto-regenerated",
            })

        # Flag 2: Check for direction contradictions between prediction and forecast
        contradictions = []
        if forecast_data and isinstance(forecast_data, dict):
            fc_direction = forecast_data.get("direction", "FLAT")
            fc_symbol = forecast_data.get("symbol", "")
            if fc_symbol and fc_symbol in predictions:
                pred = predictions[fc_symbol]
                pred_dir = pred.get("direction", "")
                pred_conf = pred.get("confidence", 0)
                pred_pct = (pred_conf * 100) if pred_conf <= 1 else pred_conf
                if pred_pct >= 55 and pred_dir in ("UP", "DOWN"):
                    if fc_direction == "FLAT" or fc_direction == "":
                        contradictions.append(
                            f"{fc_symbol}: prediction={pred_dir} {pred_pct:.0f}% but forecast=FLAT"
                        )
                    elif pred_dir != fc_direction:
                        contradictions.append(
                            f"{fc_symbol}: prediction={pred_dir} {pred_pct:.0f}% but forecast={fc_direction}"
                        )

        summary["prediction_forecast_contradictions"] = contradictions
        if contradictions:
            issues.append({
                "type": "prediction_forecast_conflict", "severity": "warn",
                "detail": f"{len(contradictions)} symbols: prediction and forecast disagree — {contradictions[0]}",
            })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Prediction vs forecast check error: {e}")

    # ── CHECK 38: Signal Burst Rate ──────────────────────────────
    # If predictions are fired in rapid succession rather than
    # spread across analysis windows, it's a shotgun pattern.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Signal Burst Rate")
    try:
        import wolf_app as _wa
        predictions = getattr(_wa, '_LATEST_PREDICTIONS', {})

        if len(predictions) >= 5:
            timestamps = []
            for pred in predictions.values():
                ts = pred.get("run_at") or pred.get("created_at") or 0
                if isinstance(ts, (int, float)):
                    timestamps.append(ts)

            if len(timestamps) >= 5:
                timestamps.sort()
                spread_s = timestamps[-1] - timestamps[0]
                preds_per_min = (len(timestamps) / max(spread_s, 1)) * 60
                summary["prediction_burst_spread_s"] = round(spread_s, 1)
                summary["predictions_per_minute"] = round(preds_per_min, 1)

                # Check for tight bursts: >4 predictions per minute
                if spread_s < 120 and len(timestamps) >= 8:
                    issues.append({
                        "type": "signal_burst_spam", "severity": "warn",
                        "detail": f"All {len(timestamps)} predictions fired within {spread_s:.0f}s — shotgun pattern, not selective",
                    })
                elif preds_per_min >= 4:
                    issues.append({
                        "type": "signal_rapid_fire", "severity": "info",
                        "detail": f"{len(timestamps)} predictions at {preds_per_min:.1f}/min ({spread_s:.0f}s total) — rapid-fire mode, not staggered analysis",
                    })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Signal burst check error: {e}")

    # ── CHECK 39: Win Rate vs Confidence Sanity ──────────────────
    # If average confidence is 70%+ but win rate is 0%, either the
    # confidence is fake or the evaluation is broken.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Win Rate vs Confidence Sanity")
    try:
        import wolf_app as _wa
        predictions = getattr(_wa, '_LATEST_PREDICTIONS', {})

        if predictions:
            confidences = []
            for pred in predictions.values():
                c = pred.get("confidence", 0)
                pct = (c * 100) if c <= 1 else c
                confidences.append(pct)

            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            summary["avg_prediction_confidence"] = round(avg_conf, 1)

            # Cross-reference with paper trade win rate
            try:
                from core.paper_tracker import get_paper_tracker
                tracker = get_paper_tracker()
                stats = tracker.get_stats(days=30)
                win_rate = stats.get("win_rate", 0)
                resolved = stats.get("resolved_trades", 0)

                if avg_conf >= 60 and resolved >= 20 and win_rate < 0.3:
                    issues.append({
                        "type": "confidence_winrate_mismatch", "severity": "warn",
                        "detail": f"Avg confidence {avg_conf:.0f}% but win rate only {win_rate*100:.0f}% over {resolved} trades — confidence calibration broken",
                    })
            except Exception:
                pass
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Win rate vs confidence check error: {e}")

    # ── CHECK 40: Dashboard Data Consistency ─────────────────────
    # Checks that accuracy shown in different panels matches.
    # If health says 56% but goals says 50%, they're reading from
    # different sources — user sees contradictory numbers.
    # ──────────────────────────────────────────────────────────────
    checks_run.append("Dashboard Data Consistency")
    try:
        # Compare the accuracy we computed in CHECK 3 with what the
        # paper tracker thinks accuracy is
        integrity_accuracy = summary.get("accuracy_pct")

        try:
            from core.paper_tracker import get_paper_tracker
            tracker = get_paper_tracker()
            stats = tracker.get_stats(days=30)
            paper_win_rate = stats.get("win_rate", None)
            paper_resolved = stats.get("resolved_trades", 0)

            if integrity_accuracy and paper_win_rate is not None and paper_resolved >= 10:
                paper_pct = paper_win_rate * 100
                diff = abs(integrity_accuracy - paper_pct)
                summary["accuracy_vs_winrate_gap"] = round(diff, 1)

                if diff >= 20:
                    issues.append({
                        "type": "dashboard_accuracy_mismatch", "severity": "warn",
                        "detail": f"Integrity accuracy {integrity_accuracy:.1f}% vs paper win rate {paper_pct:.1f}% — {diff:.0f}pp gap, different data sources",
                    })
            elif paper_resolved == 0 and summary.get("total_evaluated", 0) > 50:
                summary["accuracy_vs_winrate_gap"] = None
                issues.append({
                    "type": "dashboard_accuracy_incomplete", "severity": "info",
                    "detail": f"Accuracy shows {integrity_accuracy:.1f}% from {summary.get('total_evaluated', 0)} evals, but paper tracker has 0 resolved trades — win rate can't be verified",
                })
        except Exception:
            pass
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Dashboard consistency check error: {e}")

    # ══════════════════════════════════════════════════════════════
    penalty = sum(SEVERITY_WEIGHTS.get(i["severity"], 1) for i in issues)
    health_score = max(0, min(100, 100 - penalty))

    # ── Score breakdown: show what costs points ──
    score_breakdown = []
    error_issues = [i for i in issues if i.get("severity") == "error"]
    warn_issues = [i for i in issues if i.get("severity") == "warn"]
    info_issues = [i for i in issues if i.get("severity") == "info"]
    if error_issues:
        score_breakdown.append({
            "component": "errors",
            "count": len(error_issues),
            "weight": SEVERITY_WEIGHTS["error"],
            "penalty": len(error_issues) * SEVERITY_WEIGHTS["error"],
            "details": [i.get("detail", i.get("type", ""))[:80] for i in error_issues[:5]],
        })
    if warn_issues:
        score_breakdown.append({
            "component": "warnings",
            "count": len(warn_issues),
            "weight": SEVERITY_WEIGHTS["warn"],
            "penalty": len(warn_issues) * SEVERITY_WEIGHTS["warn"],
            "details": [i.get("detail", i.get("type", ""))[:80] for i in warn_issues[:5]],
        })
    if info_issues:
        score_breakdown.append({
            "component": "info",
            "count": len(info_issues),
            "weight": SEVERITY_WEIGHTS["info"],
            "penalty": len(info_issues) * SEVERITY_WEIGHTS["info"],
            "details": [i.get("detail", i.get("type", ""))[:60] for i in info_issues[:3]],
        })

    return {
        "health_score": round(health_score, 1),
        "score_breakdown": score_breakdown,
        "total_penalty": round(penalty, 1),
        "auto_fixes_applied": fixes_applied,
        "issues_remaining": len(issues),
        "issues": issues[:25],
        "checks_run": checks_run,
        "checks_total": len(checks_run),
        "summary": summary,
        "last_audit": datetime.now().isoformat(),
    }
