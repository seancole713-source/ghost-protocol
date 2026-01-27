#!/usr/bin/env python3
"""
Retrain Ghost XGBoost Model with PostgreSQL Data
=================================================

This script retrains the Ghost prediction model using REAL outcomes
from PostgreSQL instead of empty SQLite databases.

Usage:
    python3 retrain_model.py

What it does:
1. Fetches training data from PostgreSQL ghost_prediction_outcomes
2. Trains XGBoost v3 classifier on 25,691+ outcomes
3. Evaluates train/test accuracy
4. Saves to models/production/ghost_model_ALL.pkl

Expected Results:
- Training samples: 1,000+ (not 0!)
- Test accuracy: >50% (better than random)
- Model learns actual patterns from data

Author: Ghost AI
Date: January 7, 2026
"""

import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Retrain Ghost model with PostgreSQL data"""
    
    print("=" * 70)
    print("GHOST XGBOOST MODEL RETRAINING")
    print("Using PostgreSQL outcomes (25,691+ predictions)")
    print("=" * 70)
    
    # Import training function
    try:
        from core.ml_trainer import train_model
    except ImportError as e:
        logger.error(f"Failed to import ml_trainer: {e}")
        return 1
    
    # Check DATABASE_URL
    import os
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith(("postgres://", "postgresql://")):
        logger.error("DATABASE_URL not configured for PostgreSQL")
        logger.error("Set DATABASE_URL environment variable")
        return 1
    
    logger.info("✅ DATABASE_URL configured")
    
    # Train model
    logger.info("Starting training...")
    logger.info("  Symbol: ALL (multi-symbol model)")
    logger.info("  Lookback: 180 days")
    logger.info("  Min samples: 100")
    
    try:
        result = train_model(
            symbol=None,  # Train on all symbols
            lookback_days=180,  # 6 months of data
            min_samples=100  # Require at least 100 predictions
        )
        
        print("\n" + "=" * 70)
        print("TRAINING RESULTS")
        print("=" * 70)
        
        if not result.get("ok"):
            logger.error(f"❌ Training failed: {result.get('error')}")
            return 1
        
        # Display results
        print(f"✅ Model trained successfully!")
        print(f"\n📊 Metrics:")
        print(f"   Training samples: {result['samples']}")
        print(f"   Train accuracy: {result['train_accuracy']*100:.2f}%")
        print(f"   Test accuracy: {result['test_accuracy']*100:.2f}%")
        print(f"   Features: {len(result['features'])} indicators")
        print(f"\n💾 Model saved:")
        print(f"   Path: {result['model_path']}")
        print(f"   Symbol: {result['symbol']}")
        
        # Evaluate results
        test_acc = result['test_accuracy']
        
        print(f"\n📈 Evaluation:")
        if test_acc < 0.45:
            print(f"   ⚠️  WARNING: Test accuracy {test_acc*100:.1f}% is LOW")
            print(f"   This is worse than random (50%)")
            print(f"   Consider enabling INVERSE_GHOST=1")
        elif test_acc > 0.55:
            print(f"   ✅ EXCELLENT: Test accuracy {test_acc*100:.1f}% beats random!")
            print(f"   Model is learning real patterns")
        else:
            print(f"   ⚠️  MARGINAL: Test accuracy {test_acc*100:.1f}% near random")
            print(f"   Model needs more data or feature engineering")
        
        # Show top features
        if result['features']:
            print(f"\n🔍 Top 10 Features:")
            for i, feature in enumerate(result['features'][:10], 1):
                print(f"   {i}. {feature}")
            if len(result['features']) > 10:
                print(f"   ... and {len(result['features']) - 10} more")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("1. Deploy this model to production")
        print("2. Monitor accuracy on new predictions")
        print("3. If accuracy is still inverted (<50%), set INVERSE_GHOST=1")
        print("4. Retrain monthly as more outcomes accumulate")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
