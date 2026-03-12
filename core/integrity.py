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
 20 CHECKS — mapped to every bug we found and fixed
═══════════════════════════════════════════════════════════════════

 CHECK 1:  Database Connectivity — PG + SQLite alive
 CHECK 2:  Prediction Staleness — engines still producing
 CHECK 3:  Overall Accuracy — skip-tag-filtered (Bug: GHOST_SCORE)
 CHECK 4:  Learning Brain Status — bench + invert zones
 CHECK 5:  Price Feed Health — _LATEST_PREDICTIONS populated
 CHECK 6:  Duplicate Predictions — same symbol+timestamp
 CHECK 7:  Stale Evaluations — past check_at but unchecked
 CHECK 8:  Direction vs Target Math — UP must target above entry
           (Bug: CHZ labeled DOWN but target above entry)
 CHECK 9:  Config / Env Vars — required keys set
 CHECK 10: Per-Symbol Accuracy — flag chronic losers
 CHECK 11: Crypto/Stock Misclassification — asset_classification.py
           must agree with config/symbols.py
           (Bug: CHZ + 32 crypto classified as stocks)
 CHECK 12: Skip-Tag Pollution — accuracy counts must exclude skips
           (Bug: 439 skip-tagged preds inflating denominator)
 CHECK 13: Ghost Brain vs Learning Brain Conflict — both brains
           shouldn't invert the same symbol (double-flip = back to bad)
 CHECK 14: Cache Data Integrity — _LATEST_PREDICTIONS entries must
           have valid direction, price, confidence, market type
 CHECK 15: Edge Whitelist Validation — active symbols in edge set
 CHECK 16: Direction Consistency Guards — format_pick() and
           _build_pick() guards must exist in ghost_notifications.py
           (Bug: brain flipped direction but target stayed wrong)
 CHECK 17: Brain Inversion + Target Recalc — inverted predictions
           must have targets on the correct side
 CHECK 18: V3 Filter Sanity — min_confidence, strategies loaded
 CHECK 19: Live Display Math — cached prediction numbers consistent
 CHECK 20: Notification Pipeline Health — imports + config

Created: March 12, 2026
Updated: March 12, 2026 — v2: expanded from 10 → 20 checks
"""

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
EVAL_OVERDUE_HOURS = 12

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
            elif accuracy_pct < 50:
                issues.append({
                    "type": "accuracy_low", "severity": "warn",
                    "detail": f"Overall accuracy {accuracy_pct:.1f}% — below 50% ({correct_count}/{len(evaluated)})",
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
            if data.get("total", 0) >= 10 and data.get("accuracy_pct", 50) < 20:
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
            issues.append({
                "type": "live_direction_mismatch", "severity": "error",
                "detail": f"LIVE display math wrong — {live_mismatches} symbols: {'; '.join(live_mismatch_examples[:3])}",
            })
        if db_mismatches > 20:
            issues.append({
                "type": "db_direction_mismatch", "severity": "warn",
                "detail": f"{db_mismatches} stored predictions have direction/target mismatch (historical)",
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
                    if acc < 35:
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

                if skip_pct > 30:
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

                if delta > 5:
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
        from core.ghost_notifications import GhostNotificationSystem

        gn_source = inspect.getsource(GhostNotificationSystem)

        has_format_pick_guard = (
            "DIRECTION CONSISTENCY GUARD" in gn_source
            or "direction" in gn_source.lower() and "consistency" in gn_source.lower()
        )
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
    # COMPUTE HEALTH SCORE
    # ══════════════════════════════════════════════════════════════
    penalty = sum(SEVERITY_WEIGHTS.get(i["severity"], 1) for i in issues)
    health_score = max(0, min(100, 100 - penalty))

    return {
        "health_score": round(health_score, 1),
        "auto_fixes_applied": fixes_applied,
        "issues_remaining": len(issues),
        "issues": issues[:25],
        "checks_run": checks_run,
        "checks_total": len(checks_run),
        "summary": summary,
        "last_audit": datetime.now().isoformat(),
    }
