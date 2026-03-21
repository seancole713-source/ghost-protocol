"""
Phase 6.7: Health Score Regression Tests

Ensures health score never drops below 90 after a deploy.
Runs as part of CI/CD pipeline and post-deployment validation.
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.db_pool import get_pool
from core.logger import get_logger

LOGGER = get_logger(__name__)

# Thresholds
MIN_HEALTH_SCORE = 90.0
MIN_ACCURACY = 40.0  # Live accuracy should be >40%
MAX_STALE_MINUTES = 120  # Predictions shouldn't be >2 hours old


async def test_health_score():
    """
    Test that overall health score is >= 90.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Get latest integrity audit results
        audit = await conn.fetchrow("""
            SELECT 
                health_score,
                created_at
            FROM ghost_integrity_audit
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        if not audit:
            LOGGER.error("❌ No integrity audit found")
            return False
        
        health_score = audit["health_score"]
        audit_age_minutes = (datetime.now() - audit["created_at"]).total_seconds() / 60
        
        if health_score < MIN_HEALTH_SCORE:
            LOGGER.error(f"❌ Health score {health_score:.1f} below threshold {MIN_HEALTH_SCORE}")
            return False
        
        if audit_age_minutes > 30:
            LOGGER.warning(f"⚠️  Latest audit is {audit_age_minutes:.1f} minutes old")
        
        LOGGER.info(f"✅ Health score: {health_score:.1f}/100 (>= {MIN_HEALTH_SCORE})")
        return True


async def test_prediction_freshness():
    """
    Test that predictions are not stale (< 2 hours old).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        latest_prediction = await conn.fetchrow("""
            SELECT 
                predicted_at,
                EXTRACT(EPOCH FROM (NOW() - predicted_at)) / 60 as age_minutes
            FROM ghost_predictions
            ORDER BY predicted_at DESC
            LIMIT 1
        """)
        
        if not latest_prediction:
            LOGGER.error("❌ No predictions found")
            return False
        
        age_minutes = latest_prediction["age_minutes"]
        
        if age_minutes > MAX_STALE_MINUTES:
            LOGGER.error(f"❌ Predictions stale: {age_minutes:.1f} minutes old (max {MAX_STALE_MINUTES})")
            return False
        
        LOGGER.info(f"✅ Predictions fresh: {age_minutes:.1f} minutes old")
        return True


async def test_live_accuracy():
    """
    Test that live accuracy is >= 40%.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        accuracy_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE correct = true) as correct,
                ROUND(AVG(CASE WHEN correct THEN 1 ELSE 0 END) * 100, 2) as accuracy
            FROM ghost_predictions
            WHERE reconciled = true
            AND predicted_at > NOW() - INTERVAL '30 days'
        """)
        
        if not accuracy_stats or accuracy_stats["total"] == 0:
            LOGGER.warning("⚠️  No reconciled predictions in last 30 days")
            return True  # Don't fail on fresh deploy
        
        accuracy = float(accuracy_stats["accuracy"]) if accuracy_stats["accuracy"] else 0
        
        if accuracy < MIN_ACCURACY:
            LOGGER.error(f"❌ Accuracy {accuracy:.1f}% below threshold {MIN_ACCURACY}%")
            return False
        
        LOGGER.info(f"✅ Live accuracy: {accuracy:.1f}% (>= {MIN_ACCURACY}%)")
        return True


async def test_no_critical_errors():
    """
    Test that no CRITICAL level errors exist in recent logs.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Check for recent critical errors in audit
        critical_errors = await conn.fetchval("""
            SELECT COUNT(*)
            FROM ghost_integrity_audit
            WHERE created_at > NOW() - INTERVAL '1 hour'
            AND summary::text ILIKE '%critical%'
        """)
        
        if critical_errors and critical_errors > 0:
            LOGGER.error(f"❌ Found {critical_errors} critical errors in last hour")
            return False
        
        LOGGER.info("✅ No critical errors detected")
        return True


async def test_subsystems_operational():
    """
    Test that at least 7/9 intelligence subsystems are operational.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        latest_audit = await conn.fetchrow("""
            SELECT summary
            FROM ghost_integrity_audit
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        if not latest_audit:
            LOGGER.warning("⚠️  No audit data available")
            return True  # Don't fail on missing data
        
        summary = latest_audit["summary"]
        subsystems = summary.get("subsystems", {})
        
        active_count = sum(1 for status in subsystems.values() if status == "active")
        total_count = len(subsystems)
        
        if active_count < 7:
            LOGGER.error(f"❌ Only {active_count}/{total_count} subsystems active (need >= 7)")
            return False
        
        LOGGER.info(f"✅ Subsystems: {active_count}/{total_count} active")
        return True


async def run_health_regression_tests():
    """
    Run all health regression tests.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    print("\n" + "="*70)
    print("🏥 HEALTH REGRESSION TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        ("Health Score >= 90", test_health_score),
        ("Predictions Fresh (<2h)", test_prediction_freshness),
        ("Live Accuracy >= 40%", test_live_accuracy),
        ("No Critical Errors", test_no_critical_errors),
        ("Subsystems Operational (>=7/9)", test_subsystems_operational),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            LOGGER.error(f"Test '{test_name}' failed with exception: {e}", exc_info=True)
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {test_name}")
    print("="*70 + "\n")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    if passed_count == total_count:
        print(f"✅ ALL TESTS PASSED ({passed_count}/{total_count})")
        print("="*70 + "\n")
        return True
    else:
        print(f"❌ TESTS FAILED ({passed_count}/{total_count} passed)")
        print("="*70 + "\n")
        return False


async def main():
    try:
        all_passed = await run_health_regression_tests()
        sys.exit(0 if all_passed else 1)
    except Exception as e:
        LOGGER.error(f"Health regression tests failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
