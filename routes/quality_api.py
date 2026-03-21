"""
System Quality Monitoring API Endpoints (Phases 4.2, 4.3, 5.6)

Exposes prediction diversity, duplicates, and scheduling health checks.

Ghost Protocol v5 — Session 6
"""

from fastapi import APIRouter
import logging

router = APIRouter()
LOGGER = logging.getLogger("ghost.quality")


@router.get("/api/quality/diversity")
async def api_quality_diversity():
    """
    Check prediction diversity (Phase 4.3).
    
    Returns balance between UP and DOWN predictions over last 24 hours.
    """
    try:
        from core.prediction_diversity import check_prediction_diversity
        from core.db_pool import get_sync_connection
        import time
        
        # Fetch recent predictions from database
        cutoff_ts = int(time.time()) - (24 * 3600)  # Last 24 hours
        
        with get_sync_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT symbol, direction, EXTRACT(EPOCH FROM predicted_at)::int as predicted_at
                FROM ghost_predictions
                WHERE predicted_at > to_timestamp(%s)
                ORDER BY predicted_at DESC
                LIMIT 1000
            """, (cutoff_ts,))
            rows = cur.fetchall()
        
        predictions = [
            {"symbol": row[0], "direction": row[1], "predicted_at": row[2]}
            for row in rows
        ]
        
        result = check_prediction_diversity(predictions)
        return result
        
    except Exception as e:
        LOGGER.error(f"Diversity check failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "diversity_score": 0.0
        }


@router.get("/api/quality/duplicates")
async def api_quality_duplicates():
    """
    Check for duplicate predictions (Phase 5.6).
    
    Scans recent predictions for duplicates within 60-second window.
    """
    try:
        from core.duplicate_checker import check_for_duplicates
        from core.db_pool import get_sync_connection
        import time
        
        # Fetch recent predictions (last 7 days)
        cutoff_ts = int(time.time()) - (7 * 24 * 3600)
        
        with get_sync_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT id, symbol, direction, EXTRACT(EPOCH FROM predicted_at)::int as predicted_at
                FROM ghost_predictions
                WHERE predicted_at > to_timestamp(%s)
                ORDER BY predicted_at DESC
                LIMIT 5000
            """, (cutoff_ts,))
            rows = cur.fetchall()
        
        predictions = [
            {
                "id": row[0],
                "symbol": row[1],
                "direction": row[2],
                "predicted_at": row[3]
            }
            for row in rows
        ]
        
        result = check_for_duplicates(predictions)
        return result
        
    except Exception as e:
        LOGGER.error(f"Duplicate check failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "duplicate_count": 0
        }


@router.get("/api/quality/scheduling")
async def api_quality_scheduling():
    """
    Check prediction scheduling consistency (Phase 4.2).
    
    Returns scheduling health metrics and drift analysis.
    """
    try:
        from core.prediction_scheduler import get_scheduling_status
        
        result = get_scheduling_status()
        return result
        
    except Exception as e:
        LOGGER.error(f"Scheduling check failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "consistency_score": 0.0
        }


@router.get("/api/quality/summary")
async def api_quality_summary():
    """
    Combined quality metrics dashboard.
    
    Returns diversity, duplicates, and scheduling in one call.
    """
    diversity = await api_quality_diversity()
    duplicates = await api_quality_duplicates()
    scheduling = await api_quality_scheduling()
    
    # Calculate overall quality score
    scores = []
    if diversity.get("diversity_score") is not None:
        scores.append(diversity["diversity_score"])
    if scheduling.get("consistency_score") is not None:
        scores.append(scheduling["consistency_score"])
    if duplicates.get("ok"):
        scores.append(100.0)  # No duplicates = perfect score
    else:
        # Deduct points for duplicates
        dup_pct = duplicates.get("duplicate_pct", 0)
        scores.append(max(0, 100 - dup_pct * 10))
    
    overall_score = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "ok": diversity.get("ok", True) and duplicates.get("ok", True) and scheduling.get("ok", True),
        "overall_quality_score": round(overall_score, 1),
        "diversity": diversity,
        "duplicates": duplicates,
        "scheduling": scheduling
    }
