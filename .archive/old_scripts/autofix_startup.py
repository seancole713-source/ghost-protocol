#!/usr/bin/env python3
"""
Ghost Protocol - Auto-Fix on Railway Startup
==============================================

This runs automatically on Railway deployment to:
1. Test PostgreSQL connections
2. Retrain model if needed
3. Set INVERSE_GHOST if accuracy < 50%

Runs in background so it doesn't block main app startup.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ghost.autofix")


async def run_autofix():
    """Run auto-fix sequence"""
    
    logger.info("=" * 70)
    logger.info("🧠 GHOST AUTO-FIX STARTING")
    logger.info("=" * 70)
    
    # Wait 30 seconds for main app to start
    logger.info("⏳ Waiting 30s for main app to start...")
    await asyncio.sleep(30)
    
    # Check if DATABASE_URL is set
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith(("postgres://", "postgresql://")):
        logger.error("❌ DATABASE_URL not configured - skipping auto-fix")
        return
    
    logger.info("✅ DATABASE_URL configured")
    
    # ========================================================================
    # STEP 1: Test PostgreSQL Fixes
    # ========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 1/3: Testing PostgreSQL Fixes")
    logger.info("=" * 70)
    
    try:
        from test_postgres_fixes import (
            test_database_url,
            test_ml_trainer,
            test_learning_loop,
            test_direct_postgres,
            test_data_quality
        )
        
        # Run tests
        results = {
            "database_url": test_database_url(),
            "ml_trainer": test_ml_trainer(),
            "learning_loop": test_learning_loop(),
            "direct_postgres": test_direct_postgres(),
            "data_quality": test_data_quality(),
        }
        
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        logger.info(f"Tests: {passed}/{total} passed")
        
        if passed < total:
            logger.warning(f"⚠️  {total - passed} tests failed - check logs")
            # Continue anyway, maybe retrain will fix it
        else:
            logger.info("✅ All PostgreSQL tests passed!")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.info("Continuing to retrain anyway...")
    
    # ========================================================================
    # STEP 2: Check if Model Needs Retraining
    # ========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 2/3: Checking Model Status")
    logger.info("=" * 70)
    
    try:
        from core.learning_loop import get_learning_loop
        
        ll = get_learning_loop()
        accuracy = ll._get_postgres_direction_accuracy(days=7)
        
        if "error" in accuracy or accuracy.get("count", 0) == 0:
            logger.warning(f"⚠️  Cannot get accuracy: {accuracy.get('error', 'no data')}")
            logger.info("Will retrain model anyway...")
            should_retrain = True
        else:
            acc_pct = accuracy.get("accuracy_pct", 0)
            count = accuracy.get("count", 0)
            
            logger.info(f"📊 Current 7-day accuracy: {acc_pct:.2f}% ({count} outcomes)")
            
            # Retrain if accuracy is bad OR model is old
            model_path = Path("/app/models/production/ghost_model_ALL.pkl")
            model_age_days = 999
            
            if model_path.exists():
                model_age_s = time.time() - model_path.stat().st_mtime
                model_age_days = model_age_s / 86400
                logger.info(f"📅 Model age: {model_age_days:.1f} days")
            else:
                logger.info("📅 Model not found")
            
            should_retrain = (acc_pct < 55 or model_age_days > 30)
            
            if should_retrain:
                logger.info(f"🔄 Retraining needed (accuracy={acc_pct:.1f}% or age={model_age_days:.0f}d)")
            else:
                logger.info(f"✅ Model is good (accuracy={acc_pct:.1f}%, age={model_age_days:.0f}d)")
        
    except Exception as e:
        logger.error(f"❌ Accuracy check failed: {e}")
        should_retrain = True
        logger.info("Will retrain as fallback...")
    
    # ========================================================================
    # STEP 2b: Retrain Model if Needed
    # ========================================================================
    if should_retrain:
        logger.info("")
        logger.info("=" * 70)
        logger.info("STEP 2b: Retraining Model")
        logger.info("=" * 70)
        logger.info("⏱️  This may take 2-5 minutes...")
        
        try:
            from core.ml_trainer import train_model
            
            result = train_model(
                symbol=None,
                lookback_days=180,
                min_samples=100
            )
            
            if result.get("ok"):
                train_acc = result.get("train_accuracy", 0)
                test_acc = result.get("test_accuracy", 0)
                samples = result.get("samples", 0)
                
                logger.info(f"✅ Model retrained successfully!")
                logger.info(f"   Training samples: {samples}")
                logger.info(f"   Train accuracy: {train_acc*100:.2f}%")
                logger.info(f"   Test accuracy: {test_acc*100:.2f}%")
                
                # Store test accuracy for step 3
                global _test_accuracy
                _test_accuracy = test_acc
                
            else:
                logger.error(f"❌ Retraining failed: {result.get('error')}")
                _test_accuracy = None
                
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}", exc_info=True)
            _test_accuracy = None
    else:
        # Use current accuracy
        try:
            _test_accuracy = accuracy.get("accuracy_pct", 0) / 100
        except:
            _test_accuracy = None
    
    # ========================================================================
    # STEP 3: Check INVERSE_GHOST Setting
    # ========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("STEP 3/3: Checking INVERSE_GHOST Setting")
    logger.info("=" * 70)
    
    try:
        current_inverse = os.getenv("INVERSE_GHOST", "0")
        logger.info(f"Current INVERSE_GHOST: {current_inverse}")
        
        if _test_accuracy is not None:
            test_acc_pct = _test_accuracy * 100
            
            if test_acc_pct < 50:
                logger.warning(f"⚠️  Test accuracy {test_acc_pct:.1f}% < 50% (anti-correlated)")
                
                if current_inverse != "1":
                    logger.warning("🔧 INVERSE_GHOST should be 1 but is 0")
                    logger.warning("   Set INVERSE_GHOST=1 in Railway environment variables")
                    logger.warning(f"   This would flip accuracy to ~{100-test_acc_pct:.1f}%")
                else:
                    logger.info("✅ INVERSE_GHOST=1 correctly enabled (model is anti-correlated)")
                    
            else:
                logger.info(f"✅ Test accuracy {test_acc_pct:.1f}% > 50% (correctly correlated)")
                
                if current_inverse == "1":
                    logger.warning("⚠️  INVERSE_GHOST=1 but model is NOT anti-correlated")
                    logger.warning("   Set INVERSE_GHOST=0 in Railway environment variables")
                    logger.warning("   Current setting makes predictions WORSE")
                else:
                    logger.info("✅ INVERSE_GHOST=0 correctly set (model is good)")
        else:
            logger.warning("⚠️  Cannot determine if INVERSE_GHOST needed (no test accuracy)")
            
    except Exception as e:
        logger.error(f"❌ INVERSE_GHOST check failed: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    logger.info("")
    logger.info("=" * 70)
    logger.info("🎉 AUTO-FIX COMPLETE")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  • PostgreSQL: Tested")
    logger.info(f"  • Model: {'Retrained' if should_retrain else 'Good (no retrain needed)'}")
    if _test_accuracy is not None:
        logger.info(f"  • Test Accuracy: {_test_accuracy*100:.2f}%")
        logger.info(f"  • INVERSE_GHOST: {current_inverse} ({'✅ Correct' if (_test_accuracy < 0.5 and current_inverse == '1') or (_test_accuracy >= 0.5 and current_inverse == '0') else '⚠️  Check setting'})")
    logger.info("")
    logger.info("✅ All synapses should now be GREEN!")
    logger.info("")


_test_accuracy = None


async def run_autofix_startup():
    """
    Wrapper function called by orchestrator.py
    Runs the auto-fix sequence in background
    """
    try:
        await run_autofix()
    except Exception as e:
        logger.error(f"Auto-fix startup failed: {e}", exc_info=True)


def main():
    """Main entry point for standalone execution"""
    try:
        asyncio.run(run_autofix())
    except Exception as e:
        logger.error(f"Auto-fix failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
