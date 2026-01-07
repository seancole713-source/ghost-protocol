#!/usr/bin/env python3
"""
GHOST PROTOCOL - MODEL RETRAINING WITH BIAS FIX
================================================
Fixes the 70% DOWN bias by balancing training data with scale_pos_weight

Root Cause:
- Model trained on imbalanced data (70% DOWN, 30% UP)
- Predicts DOWN 96% of time in bullish market
- SELL predictions: 33% accuracy
- BUY predictions: 87% accuracy

The Fix:
- Use XGBoost's scale_pos_weight to balance classes
- Calculate proper weight: (negative_samples / positive_samples)
- Expected result: 50% UP, 50% DOWN predictions
- Target accuracy: 65-70%

Usage:
    python3 retrain_model.py
    # OR on Railway:
    railway run python3 retrain_model.py
"""

import os
import sys
sys.path.insert(0, '/workspaces/ghost-protocol')
sys.path.insert(0, '.')

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import logging

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False
    print("❌ psycopg2 not installed")
    sys.exit(1)

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("❌ ML libraries not installed (xgboost, sklearn, numpy)")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("🔧 GHOST PROTOCOL - MODEL RETRAINING (BIAS FIX)")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL not set")
    sys.exit(1)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

def get_training_data(days: int = 90):
    """Fetch training data from PostgreSQL outcomes"""
    print(f"\n📊 Fetching training data (last {days} days)...")
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get outcomes with features
    cur.execute("""
        SELECT 
            o.prediction_id,
            o.symbol,
            p.predicted_direction,
            p.confidence,
            o.hit_direction,
            o.entry_price,
            o.exit_price,
            p.features_json,
            o.created_at
        FROM ghost_prediction_outcomes o
        JOIN ghost_predictions p ON o.prediction_id = p.id
        WHERE o.status = 'closed'
          AND o.created_at > NOW() - INTERVAL '%s days'
          AND p.features_json IS NOT NULL
        ORDER BY o.created_at ASC
    """, (days,))
    
    rows = cur.fetchall()
    conn.close()
    
    print(f"  Found {len(rows)} outcomes with features")
    return rows


def extract_features_and_labels(data):
    """Extract feature vectors and labels from outcomes"""
    print("\n🔍 Extracting features and labels...")
    
    X = []
    y = []
    feature_names = None
    
    for row in data:
        # Extract features
        features_json = row.get('features_json', {})
        if isinstance(features_json, str):
            features_json = json.loads(features_json)
        
        if not features_json:
            continue
        
        # Get feature names from first sample
        if feature_names is None:
            feature_names = sorted(features_json.keys())
            print(f"  Using {len(feature_names)} features")
        
        # Build feature vector
        feature_vector = [features_json.get(name, 0) for name in feature_names]
        X.append(feature_vector)
        
        # Extract label (UP=1, DOWN=0)
        direction = row['hit_direction']
        if direction:
            label = 1 if direction.upper() == 'UP' else 0
        else:
            # Fallback: compare exit vs entry price
            entry = float(row.get('entry_price', 0))
            exit_price = float(row.get('exit_price', 0))
            label = 1 if exit_price > entry else 0
        
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Calculate class distribution
    up_count = np.sum(y == 1)
    down_count = np.sum(y == 0)
    up_pct = (up_count / len(y)) * 100 if len(y) > 0 else 0
    down_pct = (down_count / len(y)) * 100 if len(y) > 0 else 0
    
    print(f"\n📊 Training Data Distribution:")
    print(f"  UP (1):   {up_count:>6} samples ({up_pct:>5.1f}%)")
    print(f"  DOWN (0): {down_count:>6} samples ({down_pct:>5.1f}%)")
    print(f"  Total:    {len(y):>6} samples")
    
    # Calculate scale_pos_weight for balancing
    if up_count > 0:
        scale_pos_weight = down_count / up_count
        print(f"\n⚖️  scale_pos_weight = {scale_pos_weight:.2f}")
        print(f"   (This will balance the classes during training)")
    else:
        scale_pos_weight = 1.0
        print(f"\n⚠️  No UP samples found, using scale_pos_weight = 1.0")
    
    return X, y, feature_names, scale_pos_weight


def train_model(X, y, feature_names, scale_pos_weight):
    """Train XGBoost model with time-series split and class balancing"""
    print("\n🤖 Training XGBoost model...")
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(X)}")
    print(f"  scale_pos_weight: {scale_pos_weight:.2f} (balances UP/DOWN)")
    
    # Time series split (no look-ahead bias)
    tscv = TimeSeriesSplit(n_splits=5)
    
    # XGBoost parameters with class balancing
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'scale_pos_weight': scale_pos_weight,  # KEY: Balances classes
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    
    # Evaluate with time-series CV
    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train, verbose=False)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        # Check prediction distribution
        y_pred = model.predict(X_test)
        up_pred = np.sum(y_pred == 1)
        down_pred = np.sum(y_pred == 0)
        up_pred_pct = (up_pred / len(y_pred)) * 100 if len(y_pred) > 0 else 0
        
        fold_scores.append({
            'fold': fold,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'up_pred_pct': up_pred_pct
        })
        
        print(f"  Fold {fold}: Train={train_acc:.3f}, Test={test_acc:.3f}, UP predictions={up_pred_pct:.1f}%")
    
    # Train final model on all data
    print("\n  Training final model on all data...")
    model.fit(X, y, verbose=False)
    
    final_acc = model.score(X, y)
    print(f"  Final accuracy: {final_acc:.3f}")
    
    # Test prediction distribution on all data
    y_pred_all = model.predict(X)
    up_pred_all = np.sum(y_pred_all == 1)
    down_pred_all = np.sum(y_pred_all == 0)
    up_pred_pct_all = (up_pred_all / len(y_pred_all)) * 100
    
    print(f"\n📊 Final Model Prediction Distribution:")
    print(f"  UP predictions:   {up_pred_pct_all:.1f}%")
    print(f"  DOWN predictions: {100-up_pred_pct_all:.1f}%")
    
    if 40 <= up_pred_pct_all <= 60:
        print("  ✅ BALANCED (target: 50/50)")
    else:
        print(f"  ⚠️  Still biased (target: 50%, got {up_pred_pct_all:.1f}%)")
    
    # Feature importance
    print("\n📈 Top 10 Feature Importances:")
    importance = model.feature_importances_
    top_features = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)[:10]
    for name, imp in top_features:
        print(f"  {name:30}: {imp:.4f}")
    
    return model, fold_scores


def save_model(model, feature_names, metadata):
    """Save model and metadata"""
    print("\n💾 Saving model...")
    
    # Save model
    model_path = MODEL_DIR / "ensemble_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Model saved: {model_path}")
    
    # Save feature names
    features_path = MODEL_DIR / "feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(feature_names, f, indent=2)
    print(f"  Features saved: {features_path}")
    
    # Save metadata
    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: {metadata_path}")
    
    return model_path


def main():
    """Main retraining pipeline"""
    
    # 1. Fetch training data
    data = get_training_data(days=90)
    
    if len(data) < 100:
        print(f"\n❌ Insufficient data: {len(data)} samples (need at least 100)")
        print("   Wait for more predictions to resolve, or increase days parameter")
        return
    
    # 2. Extract features and labels
    X, y, feature_names, scale_pos_weight = extract_features_and_labels(data)
    
    if len(X) < 100:
        print(f"\n❌ Insufficient valid samples: {len(X)} (need at least 100)")
        return
    
    # 3. Train model with class balancing
    model, fold_scores = train_model(X, y, feature_names, scale_pos_weight)
    
    # 4. Save model
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'samples': len(X),
        'features': len(feature_names),
        'scale_pos_weight': float(scale_pos_weight),
        'cv_scores': fold_scores,
        'note': 'Retrained with scale_pos_weight to fix DOWN bias'
    }
    
    model_path = save_model(model, feature_names, metadata)
    
    # 5. Summary
    avg_test_acc = np.mean([s['test_acc'] for s in fold_scores])
    avg_up_pred = np.mean([s['up_pred_pct'] for s in fold_scores])
    
    print("\n" + "=" * 80)
    print("✅ RETRAINING COMPLETE")
    print("=" * 80)
    print(f"""
📊 Results:
   Training samples: {len(X)}
   Features: {len(feature_names)}
   Scale weight: {scale_pos_weight:.2f}
   
   Average Test Accuracy: {avg_test_acc:.1%}
   Average UP predictions: {avg_up_pred:.1f}%
   
   UP/DOWN Balance: {'✅ FIXED' if 40 <= avg_up_pred <= 60 else '⚠️ Still biased'}
   
🎯 Expected Impact:
   Before: 70% DOWN predictions, 55% accuracy
   After:  ~50% UP/DOWN, 65-70% accuracy
   
📁 Files Updated:
   {model_path}
   {MODEL_DIR / 'feature_names.json'}
   {MODEL_DIR / 'model_metadata.json'}
   
🚀 Next Steps:
   1. Commit and push changes
   2. Railway will auto-deploy new model
   3. Wait 48h for new predictions to resolve
   4. Check accuracy improvement
   
💬 Verification Command:
   curl "https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC"
   # Should see more balanced UP/DOWN predictions, not 96% DOWN
""")
    
    return metadata


if __name__ == "__main__":
    try:
        metadata = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
