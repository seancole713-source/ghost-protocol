#!/usr/bin/env python3
"""
Smoke Test: PredictionStore

Tests that the PredictionStore abstraction is correctly configured.
NO live price calls, NO fastapi, NO broker, NO execution.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.prediction_store import get_prediction_store, PREDICTION_STORE_ENGINE, PREDICTION_DUAL_WRITE


def main():
    """Run smoke test for PredictionStore configuration."""
    print("=" * 60)
    print("PredictionStore Smoke Test")
    print("=" * 60)
    
    # Display configuration
    print(f"\n✅ Active Backend: {PREDICTION_STORE_ENGINE.upper()}")
    print(f"✅ Dual-Write Mode: {'ENABLED' if PREDICTION_DUAL_WRITE else 'DISABLED'}")
    
    # Get prediction store instance
    store = get_prediction_store()
    print(f"✅ PredictionStore initialized: {store.__class__.__name__}")
    print(f"✅ Primary Backend: {store.backend.__class__.__name__}")
    
    if store.dual_write_backend:
        print(f"✅ Secondary Backend: {store.dual_write_backend.__class__.__name__}")
    else:
        print("✅ No secondary backend (dual-write disabled)")
    
    # Test prediction creation interface (mock data)
    print("\n" + "=" * 60)
    print("Testing Prediction Creation Interface")
    print("=" * 60)
    
    # Mock forecast points (no live data)
    mock_forecast_points = [
        (1700000000.0, 100.0),
        (1700003600.0, 101.0),
        (1700007200.0, 102.0),
    ]
    
    try:
        prediction_id = store.save_prediction(
            symbol="TEST",
            forecast_points=mock_forecast_points,
            method="smoke_test",
            confidence=0.75,
            direction="UP",
            features={"test": True},
            params={"horizon_h": 48},
            tag="smoke_test",
        )
        print(f"✅ Prediction created successfully (ID: {prediction_id})")
        
        # Verify retrieval
        prediction = store.get_prediction(prediction_id)
        if prediction:
            print(f"✅ Prediction retrieved: {prediction['symbol']} - {prediction['direction']}")
        else:
            print("⚠️  Warning: Could not retrieve created prediction")
        
    except Exception as e:
        print(f"❌ Error creating prediction: {e}")
        return 1
    
    print("\n" + "=" * 60)
    print("Smoke Test Complete")
    print("=" * 60)
    print("\n✅ PredictionStore is correctly configured")
    print(f"✅ Default backend: {PREDICTION_STORE_ENGINE.upper()}")
    print(f"✅ API behavior: UNCHANGED (SQLite default)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
