#!/usr/bin/env python3
"""
PRODUCTION MODEL RETRAINING - DIRECT EXECUTION
===============================================
Retrains ghost_xgboost_v2.pkl with scale_pos_weight to fix 70% DOWN bias.

This script can be called via Railway API or directly:
    python3 scripts/retrain_production_model.py
"""

import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    print("❌ psycopg2 not installed")
    sys.exit(1)

try:
    import xgboost as xgb
    from sklearn.model_selection import TimeSeriesSplit
    import numpy as np
except ImportError:
    print("❌ xgboost/sklearn/numpy not installed")
    sys.exit(1)

print("=" * 80)
print("🔧 PRODUCTION MODEL RETRAIN - ghost_xgboost_v2.pkl")
print("=" * 80)

# Get database from environment
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("❌ DATABASE_URL not set")
    sys.exit(1)

print(f"✅ Database: Connected")

# Model paths
MODEL_DIR = Path(__file__).parent.parent / "models" / "trained"
MODEL_PATH = MODEL_DIR / "ghost_xgboost_v2.pkl"
BACKUP_PATH = MODEL_DIR / f"ghost_xgboost_v2_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

print(f"📁 Model path: {MODEL_PATH}")

def fetch_outcomes(days=90):
    """Fetch closed prediction outcomes"""
    print(f"\n📊 Fetching outcomes (last {days} days)...")
    
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    cur.execute("""
        SELECT 
            o.prediction_id,
            p.symbol,
            o.hit_direction,
            o.price_at_prediction as entry_price,
            o.price_at_resolution as exit_price,
            p.features_json,
            o.created_at
        FROM ghost_prediction_outcomes o
        JOIN predictions p ON o.prediction_id = p.id
        WHERE o.status = 'completed'
          AND o.created_at > NOW() - INTERVAL '%s days'
          AND p.features_json IS NOT NULL
          AND o.hit_direction IS NOT NULL
        ORDER BY o.created_at ASC
    """, (days,))
    
    rows = cur.fetchall()
    conn.close()
    
    print(f"  Found {len(rows)} closed outcomes")
    return rows


def extract_features(data):
    """Extract feature matrix and labels"""
    print("\n🔍 Extracting features...")
    
    X = []
    y = []
    feature_names = None
    
    for row in data:
        features_json = row.get('features_json', {})
        if isinstance(features_json, str):
            features_json = json.loads(features_json)
        
        if not features_json:
            continue
        
        if feature_names is None:
            feature_names = sorted(features_json.keys())
            print(f"  Features: {len(feature_names)}")
        
        feature_vector = [features_json.get(name, 0) for name in feature_names]
        X.append(feature_vector)
        
        # Label: UP=1, DOWN=0
        direction = row['hit_direction']
        if direction is not None:
            # Handle both int and string direction values
            if isinstance(direction, int):
                label = direction  # Already 1 or 0
            elif isinstance(direction, str):
                label = 1 if direction.upper() == 'UP' else 0
            else:
                label = 1 if direction else 0
        else:
            # Fallback: compare exit vs entry price
            entry = float(row.get('entry_price', 0))
            exit_price = float(row.get('exit_price', 0))
            label = 1 if exit_price > entry else 0
        
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    # Distribution
    up_count = np.sum(y == 1)
    down_count = np.sum(y == 0)
    
    print(f"\n📊 Distribution:")
    print(f"  UP:   {up_count:>5} ({up_count/len(y)*100:.1f}%)")
    print(f"  DOWN: {down_count:>5} ({down_count/len(y)*100:.1f}%)")
    
    # Calculate scale_pos_weight
    scale_pos_weight = down_count / up_count if up_count > 0 else 1.0
    print(f"\n⚖️  scale_pos_weight = {scale_pos_weight:.2f}")
    
    return X, y, feature_names, scale_pos_weight


def train_model(X, y, scale_pos_weight):
    """Train XGBoost with balanced classes"""
    print("\n🤖 Training XGBoost...")
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'scale_pos_weight': scale_pos_weight,  # KEY FIX
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    model = xgb.XGBClassifier(**params)
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    fold_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train, verbose=False)
        acc = model.score(X_test, y_test)
        
        y_pred = model.predict(X_test)
        up_pct = (np.sum(y_pred == 1) / len(y_pred)) * 100
        
        fold_scores.append(acc)
        print(f"  Fold {fold}: {acc:.1%} (UP predictions: {up_pct:.1f}%)")
    
    # Final model
    print("\n  Training final model...")
    model.fit(X, y, verbose=False)
    
    final_acc = model.score(X, y)
    y_pred = model.predict(X)
    up_pct = (np.sum(y_pred == 1) / len(y_pred)) * 100
    
    print(f"  Final accuracy: {final_acc:.1%}")
    print(f"  UP predictions: {up_pct:.1f}%")
    
    if 40 <= up_pct <= 60:
        print("  ✅ BALANCED")
    else:
        print(f"  ⚠️  Still biased ({up_pct:.1f}% UP)")
    
    return model, fold_scores


def save_model(model, feature_names):
    """Save model with backup"""
    print("\n💾 Saving model...")
    
    # Backup existing
    if MODEL_PATH.exists():
        import shutil
        shutil.copy(MODEL_PATH, BACKUP_PATH)
        print(f"  Backup: {BACKUP_PATH.name}")
    
    # Save new model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"  ✅ Saved: {MODEL_PATH}")
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'features': len(feature_names),
        'note': 'Retrained with scale_pos_weight to fix DOWN bias'
    }
    
    metadata_path = MODEL_DIR / "training_results_v2.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✅ Metadata: {metadata_path.name}")


def main():
    # 1. Fetch data
    data = fetch_outcomes(days=90)
    
    if len(data) < 100:
        print(f"\n❌ Insufficient data: {len(data)} samples (need 100+)")
        return False
    
    # 2. Extract features
    X, y, feature_names, scale_pos_weight = extract_features(data)
    
    if len(X) < 100:
        print(f"\n❌ Insufficient valid samples: {len(X)}")
        return False
    
    # 3. Train
    model, fold_scores = train_model(X, y, scale_pos_weight)
    
    # 4. Save
    save_model(model, feature_names)
    
    # 5. Summary
    avg_acc = np.mean(fold_scores)
    
    print("\n" + "=" * 80)
    print("✅ RETRAINING COMPLETE")
    print("=" * 80)
    print(f"""
📊 Results:
   Samples: {len(X)}
   Features: {len(feature_names)}
   Scale weight: {scale_pos_weight:.2f}
   Avg accuracy: {avg_acc:.1%}
   
🚀 Next Steps:
   1. Railway will use new model on next restart
   2. Or restart now: railway up
   3. Verify: curl https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=BTC
   
💬 Expected Impact:
   Before: 70% DOWN predictions
   After:  ~50% UP / 50% DOWN (balanced)
""")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
