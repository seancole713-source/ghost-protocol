#!/usr/bin/env python3
"""
XGBoost Model Retraining Script for Ghost Protocol

This script:
1. Fetches recent prediction outcomes from paper_trades
2. Balances the dataset (equal UP/DOWN samples)
3. Retrains XGBoost with fresh data
4. Saves new model weights
5. Backs up old model

Usage:
    python scripts/retrain_xgboost.py --min-samples 500 --test-split 0.2
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psycopg2
    import psycopg2.extras
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("WARNING: psycopg2 not available")

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ERROR: Required packages not installed. Run:")
    print("  pip install xgboost scikit-learn pandas")


def fetch_training_data(db_url: str, min_samples: int = 500) -> pd.DataFrame:
    """
    Fetch resolved paper trades with their features for training.
    Only includes trades where we know the outcome (WIN/LOSS).
    """
    print(f"📊 Fetching training data (min {min_samples} samples)...")
    
    if not POSTGRES_AVAILABLE:
        print("ERROR: psycopg2 not available")
        return None
    
    conn = psycopg2.connect(db_url)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    # Get resolved trades with outcomes
    cur.execute("""
        SELECT 
            pt.symbol,
            pt.signal_direction,
            pt.signal_confidence,
            pt.entry_price,
            pt.outcome,
            pt.profit_loss_pct,
            pt.entry_time
        FROM paper_trades pt
        WHERE pt.outcome IN ('WIN', 'LOSS', 'STOPPED')
        ORDER BY pt.entry_time DESC
        LIMIT 10000
    """)
    
    rows = cur.fetchall()
    conn.close()
    
    if len(rows) < min_samples:
        print(f"⚠️  Only {len(rows)} resolved trades found. Need at least {min_samples}.")
        print("   Wait for more paper trades to resolve before retraining.")
        return None
    
    print(f"✅ Found {len(rows)} resolved trades")
    return pd.DataFrame([dict(r) for r in rows])


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Extract features and labels from the training data.
    Returns (X, y) where X is feature matrix and y is labels.
    """
    print("🔧 Preparing features...")
    
    # Create basic features from available data
    feature_rows = []
    labels = []
    
    for _, row in df.iterrows():
        try:
            # Basic features we can extract
            features = {
                'confidence': float(row['signal_confidence']) if row['signal_confidence'] else 0.5,
                'is_buy': 1 if row['signal_direction'] == 'UP' else 0,
                'entry_price': float(row['entry_price']) if row['entry_price'] else 0,
            }
            feature_rows.append(features)
            
            # Label: 1 if prediction was correct, 0 if wrong
            if row['outcome'] == 'WIN':
                labels.append(1)
            else:
                labels.append(0)
        except Exception as e:
            continue
    
    X = pd.DataFrame(feature_rows)
    y = np.array(labels)
    
    # Fill NaN with 0
    X = X.fillna(0)
    
    print(f"✅ Prepared {len(X)} samples with {len(X.columns)} features")
    print(f"   Class distribution: {sum(y)} correct / {len(y) - sum(y)} incorrect")
    
    return X, y


def balance_dataset(X: pd.DataFrame, y: np.ndarray) -> tuple:
    """
    Balance the dataset to have equal positive/negative samples.
    This prevents the model from being biased toward one direction.
    """
    print("⚖️  Balancing dataset...")
    
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    
    min_samples = min(len(pos_idx), len(neg_idx))
    
    if min_samples == 0:
        print("⚠️  Cannot balance - one class has no samples")
        return X, y
    
    # Randomly sample equal amounts
    np.random.seed(42)
    pos_sample = np.random.choice(pos_idx, min_samples, replace=False)
    neg_sample = np.random.choice(neg_idx, min_samples, replace=False)
    
    balanced_idx = np.concatenate([pos_sample, neg_sample])
    np.random.shuffle(balanced_idx)
    
    X_balanced = X.iloc[balanced_idx]
    y_balanced = y[balanced_idx]
    
    print(f"✅ Balanced to {len(X_balanced)} samples ({min_samples} each class)")
    
    return X_balanced, y_balanced


def train_model(X: pd.DataFrame, y: np.ndarray, test_split: float = 0.2) -> tuple:
    """
    Train XGBoost model on the prepared data.
    Returns (model, accuracy, report).
    """
    if not ML_AVAILABLE:
        print("ERROR: ML packages not available")
        return None, 0, ""
    
    print("🚀 Training XGBoost model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n✅ Model trained!")
    print(f"   Test Accuracy: {accuracy:.1%}")
    print(f"\nClassification Report:\n{report}")
    
    return model, accuracy, report


def save_model(model, accuracy: float, model_dir: str = "models"):
    """
    Save the trained model with backup of old model.
    """
    model_path = Path(model_dir)
    model_path.mkdir(exist_ok=True)
    
    # Backup old model
    old_model = model_path / "xgboost_v2.pkl"
    if old_model.exists():
        backup_name = f"xgboost_v2_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        old_model.rename(model_path / backup_name)
        print(f"📦 Backed up old model to {backup_name}")
    
    # Save new model
    new_model_path = model_path / "xgboost_v2.pkl"
    with open(new_model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "accuracy": accuracy,
        "version": "v2_retrained"
    }
    with open(model_path / "xgboost_v2_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved new model to {new_model_path}")
    print(f"   Accuracy: {accuracy:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Retrain XGBoost model for Ghost Protocol")
    parser.add_argument("--min-samples", type=int, default=500, help="Minimum samples required")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--dry-run", action="store_true", help="Don't save model, just evaluate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  GHOST PROTOCOL - XGBOOST RETRAINING")
    print("=" * 60)
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)
    
    # Fetch data
    df = fetch_training_data(db_url, args.min_samples)
    if df is None:
        sys.exit(1)
    
    # Prepare features
    X, y = prepare_features(df)
    
    if len(X) < args.min_samples:
        print(f"ERROR: Not enough samples after feature preparation")
        sys.exit(1)
    
    # Balance dataset
    X_balanced, y_balanced = balance_dataset(X, y)
    
    # Train model
    model, accuracy, report = train_model(X_balanced, y_balanced, args.test_split)
    
    if model is None:
        print("ERROR: Training failed")
        sys.exit(1)
    
    # Save model
    if not args.dry_run:
        save_model(model, accuracy)
    else:
        print("\n⚠️  DRY RUN - Model not saved")
    
    print("\n" + "=" * 60)
    print("  RETRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
