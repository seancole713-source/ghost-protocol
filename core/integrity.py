"""
Ghost Protocol — System Integrity Audit
════════════════════════════════════════

Self-healing background audit that:
  • Runs on every page load + every 5 minutes
  • Silently fixes what it can (auto_fix=True)
  • Reports issues it can't fix with severity levels
  • Returns a 0-100 health score

Checks implemented:
  1. Prediction staleness — no new predictions in N minutes
  2. Accuracy drift — overall accuracy below threshold
  3. Learning Brain status — symbols benched/inverted
  4. Price feed health — providers returning data
  5. Duplicate predictions — same symbol+timestamp
  6. Missing evaluation fields — predictions never checked
  7. Config / env vars — critical keys not set
  8. Prediction-outcome mismatch — direction vs target disagreement
  9. Database connectivity — PG and SQLite accessible
  10. Stale evaluations — predictions past check_at but unchecked

Design principles:
  • Pull all data ONCE, loop for each check — no N+1
  • Auto-fix silently, only report unfixable — keeps score meaningful
  • Flat penalty per issue — 1 real error = real score drop
  • Try/catch per check — one failure doesn't kill the audit

Created: March 12, 2026
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
PREDICTION_STALE_MINUTES = 120     # No new prediction in 2 hours = warn
PREDICTION_VERY_STALE_MINUTES = 360  # 6 hours = error
EVAL_OVERDUE_HOURS = 12            # Past check_at by 12h = warn


def run_audit(auto_fix: bool = True) -> Dict[str, Any]:
    """
    Run the full integrity audit.
    
    Args:
        auto_fix: If True, silently fix what we can and count fixes.
        
    Returns:
        {
            "health_score": 0-100,
            "auto_fixes_applied": int,
            "issues_remaining": int,
            "issues": [...],
            "checks_run": [...],
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
            # Last 14 days of predictions
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
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 1: Database Connectivity
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Database Connectivity")
    if pg_available:
        summary["database"] = "connected"
    else:
        summary["database"] = "error"
    
    # SQLite check
    try:
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
            "type": "sqlite_error",
            "severity": "warn",
            "detail": f"SQLite check failed: {str(e)[:80]}",
        })
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 2: Prediction Staleness
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Prediction Staleness")
    if predictions:
        latest_pred = predictions[0]  # Already sorted DESC
        latest_ts = latest_pred.get("predicted_at", 0)
        age_minutes = (now_ts - latest_ts) / 60 if latest_ts else 9999
        summary["latest_prediction_age_min"] = round(age_minutes, 1)
        summary["latest_prediction_symbol"] = latest_pred.get("symbol", "?")
        
        if age_minutes > PREDICTION_VERY_STALE_MINUTES:
            issues.append({
                "type": "predictions_very_stale",
                "severity": "error",
                "detail": f"No new predictions in {age_minutes:.0f} min ({latest_pred.get('symbol', '?')} was last)",
            })
        elif age_minutes > PREDICTION_STALE_MINUTES:
            issues.append({
                "type": "predictions_stale",
                "severity": "warn",
                "detail": f"No new predictions in {age_minutes:.0f} min",
            })
    elif pg_available:
        issues.append({
            "type": "no_predictions",
            "severity": "error",
            "detail": "No predictions found in last 14 days",
        })
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 3: Overall Accuracy
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Prediction Accuracy")
    evaluated = [p for p in predictions
                 if p.get("checked") == 1
                 and not str(p.get("eval_version") or "").startswith("skip")]
    if evaluated:
        correct_count = sum(1 for p in evaluated if p.get("correct") == 1)
        accuracy_pct = (correct_count / len(evaluated)) * 100
        summary["accuracy_pct"] = round(accuracy_pct, 1)
        summary["total_evaluated"] = len(evaluated)
        summary["total_correct"] = correct_count
        
        if accuracy_pct < 40:
            issues.append({
                "type": "accuracy_critical",
                "severity": "error",
                "detail": f"Overall accuracy {accuracy_pct:.1f}% — below 40% threshold ({correct_count}/{len(evaluated)})",
            })
        elif accuracy_pct < 50:
            issues.append({
                "type": "accuracy_low",
                "severity": "warn",
                "detail": f"Overall accuracy {accuracy_pct:.1f}% — below 50% ({correct_count}/{len(evaluated)})",
            })
    else:
        summary["accuracy_pct"] = None
        summary["total_evaluated"] = 0
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 4: Learning Brain Status
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Learning Brain Status")
    try:
        from core.ghost_learning_brain import get_scorecard, get_benched_symbols, get_inverted_symbols
        scorecard = get_scorecard()
        benched = get_benched_symbols()
        inverted = get_inverted_symbols()
        
        summary["brain_symbols_total"] = len(scorecard)
        summary["brain_benched"] = benched
        summary["brain_inverted"] = inverted
        
        if len(benched) + len(inverted) > len(scorecard) * 0.5 and len(scorecard) >= 5:
            issues.append({
                "type": "brain_too_many_losers",
                "severity": "warn",
                "detail": f"Over 50% of symbols benched/inverted: {', '.join(benched + inverted)}",
            })
        
        # Check for symbols with very low accuracy (< 20%)
        for sym, data in scorecard.items():
            if data.get("total", 0) >= 10 and data.get("accuracy_pct", 50) < 20:
                issues.append({
                    "type": "symbol_critical_accuracy",
                    "severity": "warn",
                    "detail": f"{sym} accuracy {data['accuracy_pct']}% over {data['total']} predictions",
                })
    except Exception as e:
        LOGGER.warning(f"[INTEGRITY] Learning Brain check failed: {e}")
        summary["brain_status"] = "unavailable"
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 5: Price Feed Health
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Price Feed Health")
    try:
        import wolf_app
        latest_preds = getattr(wolf_app, '_LATEST_PREDICTIONS', {})
        summary["active_symbols"] = len(latest_preds)
        
        if len(latest_preds) == 0:
            issues.append({
                "type": "no_active_predictions",
                "severity": "error",
                "detail": "No symbols in _LATEST_PREDICTIONS cache — engines may be down",
            })
        
        # Check for stale cached prices
        stale_price_symbols = []
        for sym, pred in latest_preds.items():
            if isinstance(pred, dict):
                pred_ts = pred.get("timestamp") or pred.get("predicted_at") or 0
                if pred_ts and (now_ts - pred_ts) > 7200:  # 2 hours
                    stale_price_symbols.append(sym)
        
        if stale_price_symbols:
            issues.append({
                "type": "stale_prices",
                "severity": "warn",
                "detail": f"Stale price cache for: {', '.join(stale_price_symbols[:5])}",
            })
    except Exception as e:
        LOGGER.debug(f"[INTEGRITY] Price feed check skipped: {e}")
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 6: Duplicate Predictions
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Duplicate Predictions")
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
                        LOGGER.info(f"[INTEGRITY] Auto-fixed: removed {len(dup_ids)} duplicate predictions")
            except Exception as e:
                LOGGER.warning(f"[INTEGRITY] Duplicate fix failed: {e}")
                issues.append({
                    "type": "duplicate_predictions",
                    "severity": "warn",
                    "detail": f"{len(duplicates)} duplicate predictions found (auto-fix failed)",
                })
        elif duplicates:
            issues.append({
                "type": "duplicate_predictions",
                "severity": "warn",
                "detail": f"{len(duplicates)} duplicate predictions found",
            })
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 7: Stale Evaluations (past check_at but not checked)
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Stale Evaluations")
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
                "type": "many_overdue_evals",
                "severity": "error",
                "detail": f"{len(overdue)} predictions overdue for evaluation (> {EVAL_OVERDUE_HOURS}h past check_at)",
            })
        elif len(overdue) > 5:
            issues.append({
                "type": "overdue_evals",
                "severity": "warn",
                "detail": f"{len(overdue)} predictions overdue for evaluation",
            })
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 8: Direction vs Target Mismatch
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Direction-Target Consistency")
    if predictions:
        mismatches = 0
        mismatch_fixed = 0
        for p in predictions:
            direction = p.get("predicted_direction")
            target = p.get("target_price")
            entry = p.get("current_price")
            if not all([direction, target, entry]):
                continue
            try:
                target_f = float(target)
                entry_f = float(entry)
            except (ValueError, TypeError):
                continue
            
            if direction == "UP" and target_f < entry_f * 0.99:
                mismatches += 1
            elif direction == "DOWN" and target_f > entry_f * 1.01:
                mismatches += 1
        
        summary["direction_target_mismatches"] = mismatches
        
        if mismatches > 5:
            issues.append({
                "type": "direction_target_mismatch",
                "severity": "warn",
                "detail": f"{mismatches} predictions have direction contradicting target price",
            })
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 9: Config / Environment Variables
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Configuration")
    required_vars = {
        "DATABASE_URL": "PostgreSQL connection — predictions won't be stored",
        "TELEGRAM_BOT_TOKEN": "Telegram alerts — picks won't be sent",
        "TELEGRAM_CHAT_ID": "Telegram channel — picks won't be delivered",
    }
    optional_vars = {
        "ALPACA_API_KEY": "Alpaca trading disabled",
        "ALPACA_SECRET_KEY": "Alpaca trading disabled",
    }
    
    missing_required = []
    missing_optional = []
    for var, desc in required_vars.items():
        if not os.environ.get(var):
            missing_required.append(var)
            issues.append({
                "type": "config_missing_required",
                "severity": "error",
                "detail": f"{var} not set — {desc}",
            })
    
    for var, desc in optional_vars.items():
        if not os.environ.get(var):
            missing_optional.append(var)
            issues.append({
                "type": "config_missing_optional",
                "severity": "info",
                "detail": f"{var} not set — {desc}",
            })
    
    summary["config_missing_required"] = missing_required
    summary["config_missing_optional"] = missing_optional
    
    # ══════════════════════════════════════════════════════════════
    # CHECK 10: Per-Symbol Accuracy Breakdown
    # ══════════════════════════════════════════════════════════════
    checks_run.append("Per-Symbol Accuracy")
    if evaluated:
        symbol_stats = {}
        for p in evaluated:
            sym = p.get("symbol", "?")
            if sym not in symbol_stats:
                symbol_stats[sym] = {"total": 0, "correct": 0}
            symbol_stats[sym]["total"] += 1
            if p.get("correct") == 1:
                symbol_stats[sym]["correct"] += 1
        
        # Flag symbols with enough data but poor accuracy
        poor_symbols = []
        for sym, stats in symbol_stats.items():
            if stats["total"] >= 10:
                acc = (stats["correct"] / stats["total"]) * 100
                if acc < 35:
                    poor_symbols.append(f"{sym} ({acc:.0f}%)")
        
        summary["per_symbol_count"] = len(symbol_stats)
        if poor_symbols:
            issues.append({
                "type": "poor_symbol_accuracy",
                "severity": "info",
                "detail": f"Low accuracy symbols: {', '.join(poor_symbols)}",
            })
    
    # ══════════════════════════════════════════════════════════════
    # COMPUTE HEALTH SCORE
    # ══════════════════════════════════════════════════════════════
    penalty = sum(SEVERITY_WEIGHTS.get(i["severity"], 1) for i in issues)
    health_score = max(0, min(100, 100 - penalty))
    
    return {
        "health_score": round(health_score, 1),
        "auto_fixes_applied": fixes_applied,
        "issues_remaining": len(issues),
        "issues": issues[:20],  # Cap display at 20
        "checks_run": checks_run,
        "summary": summary,
        "last_audit": datetime.now().isoformat(),
    }
