"""
Phase 1.2: Walk-Forward Validation

Implements time-series cross-validation to prevent overfitting.
Tests model on unseen future data using rolling training windows.

Validates that model performance doesn't degrade when deployed
on new data.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import get_logger
from core.db_pool import get_pool

LOGGER = get_logger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for time-series prediction models.
    
    Trains on historical data, tests on future data, then rolls forward.
    This simulates real-world deployment where model only sees past data.
    """
    
    def __init__(
        self,
        train_days: int = 60,
        test_days: int = 30,
        step_days: int = 15
    ):
        """
        Initialize validator.
        
        Args:
            train_days: Days of data to train on
            test_days: Days of data to test on
            step_days: Days to step forward between folds
        """
        self.train_days = train_days
        self.test_days = test_days
        self.step_days = step_days
        self.results = []
    
    async def load_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Load prediction data with outcomes."""
        pool = await get_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT 
                    symbol,
                    predicted_at,
                    direction,
                    confidence,
                    correct,
                    actual_move_pct,
                    features
                FROM ghost_predictions
                WHERE reconciled = true
                AND predicted_at BETWEEN $1 AND $2
                ORDER BY predicted_at ASC
            """
            
            rows = await conn.fetch(query, start_date, end_date)
            
            if not rows:
                LOGGER.error("No data found")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for row in rows:
                features = json.loads(row["features"]) if row["features"] else {}
                data.append({
                    "timestamp": row["predicted_at"],
                    "symbol": row["symbol"],
                    "correct": row["correct"],
                    **features
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values("timestamp")
            LOGGER.info(f"Loaded {len(df)} samples")
            return df
    
    def prepare_fold_data(
        self,
        df: pd.DataFrame,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare train/test data for one fold.
        
        Args:
            df: Full dataset
            train_start: Start of training period
            train_end: End of training period
            test_start: Start of test period
            test_end: End of test period
        
        Returns:
            (X_train, y_train, X_test, y_test)
        """
        # Filter data for training period
        train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
        train_df = df[train_mask]
        
        # Filter data for test period
        test_mask = (df["timestamp"] >= test_start) & (df["timestamp"] < test_end)
        test_df = df[test_mask]
        
        # Prepare features
        exclude_cols = {"timestamp", "symbol", "correct"}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X_train = train_df[feature_cols].fillna(0).values
        y_train = train_df["correct"].astype(int).values
        
        X_test = test_df[feature_cols].fillna(0).values
        y_test = test_df["correct"].astype(int).values
        
        return X_train, y_train, X_test, y_test, feature_cols
    
    def train_and_test_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        fold_num: int
    ) -> Dict[str, Any]:
        """
        Train model on training fold and test on test fold.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            feature_names: Feature names
            fold_num: Fold number
        
        Returns:
            Fold results
        """
        try:
            import xgboost as xgb
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
            dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
            
            # Training parameters - same as retrain_model.py
            params = {
                'objective': 'binary:logistic',
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'eval_metric': 'logloss',
                'seed': 42
            }
            
            # Train model
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=200,
                verbose_eval=False
            )
            
            # Predict on test set
            y_pred_proba = model.predict(dtest)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            train_accuracy = accuracy_score(y_train, (model.predict(dtrain) > 0.5).astype(int))
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, zero_division=0)
            test_recall = recall_score(y_test, y_pred, zero_division=0)
            
            # Overfitting indicator
            overfitting_gap = train_accuracy - test_accuracy
            
            results = {
                "fold": fold_num,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "train_accuracy": float(train_accuracy),
                "test_accuracy": float(test_accuracy),
                "test_precision": float(test_precision),
                "test_recall": float(test_recall),
                "overfitting_gap": float(overfitting_gap)
            }
            
            LOGGER.info(
                f"Fold {fold_num}: Train={train_accuracy:.1%}, Test={test_accuracy:.1%}, "
                f"Gap={overfitting_gap:.1%}"
            )
            
            return results
            
        except Exception as e:
            LOGGER.error(f"Fold {fold_num} failed: {e}")
            return {}
    
    async def run_validation(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Run walk-forward validation.
        
        Args:
            start_date: Start of validation period
            end_date: End of validation period
        
        Returns:
            List of fold results
        """
        print(f"\nLoading data from {start_date.date()} to {end_date.date()}...")
        df = await self.load_data(start_date, end_date)
        
        if df.empty:
            print("❌ No data available")
            return []
        
        # Generate folds
        folds = []
        current_start = start_date
        fold_num = 1
        
        while True:
            train_start = current_start
            train_end = train_start + timedelta(days=self.train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_days)
            
            # Stop if test period exceeds end date
            if test_end > end_date:
                break
            
            folds.append({
                "fold": fold_num,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end
            })
            
            # Step forward
            current_start += timedelta(days=self.step_days)
            fold_num += 1
        
        print(f"Generated {len(folds)} folds")
        print(f"Configuration: {self.train_days} days train, {self.test_days} days test, {self.step_days} days step\n")
        
        # Run each fold
        results = []
        for fold_info in folds:
            print(f"Processing Fold {fold_info['fold']}...")
            print(f"  Train: {fold_info['train_start'].date()} to {fold_info['train_end'].date()}")
            print(f"  Test:  {fold_info['test_start'].date()} to {fold_info['test_end'].date()}")
            
            # Prepare data
            X_train, y_train, X_test, y_test, feature_names = self.prepare_fold_data(
                df,
                fold_info["train_start"],
                fold_info["train_end"],
                fold_info["test_start"],
                fold_info["test_end"]
            )
            
            if len(X_train) < 50 or len(X_test) < 10:
                print(f"  ⚠️  Skipping fold (insufficient data)")
                continue
            
            # Train and test
            fold_results = self.train_and_test_fold(
                X_train, y_train, X_test, y_test, feature_names, fold_info["fold"]
            )
            
            if fold_results:
                fold_results.update({
                    "train_start": fold_info["train_start"].isoformat(),
                    "train_end": fold_info["train_end"].isoformat(),
                    "test_start": fold_info["test_start"].isoformat(),
                    "test_end": fold_info["test_end"].isoformat()
                })
                results.append(fold_results)
        
        self.results = results
        return results
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze validation results."""
        if not self.results:
            return {}
        
        test_accuracies = [r["test_accuracy"] for r in self.results]
        overfitting_gaps = [r["overfitting_gap"] for r in self.results]
        
        analysis = {
            "num_folds": len(self.results),
            "mean_test_accuracy": float(np.mean(test_accuracies)),
            "std_test_accuracy": float(np.std(test_accuracies)),
            "min_test_accuracy": float(np.min(test_accuracies)),
            "max_test_accuracy": float(np.max(test_accuracies)),
            "mean_overfitting_gap": float(np.mean(overfitting_gaps)),
            "max_overfitting_gap": float(np.max(overfitting_gaps)),
            "accuracy_trend": self._calculate_trend(test_accuracies)
        }
        
        # Warnings
        warnings = []
        if analysis["mean_overfitting_gap"] > 0.15:
            warnings.append("High overfitting detected (gap > 15%)")
        if analysis["std_test_accuracy"] > 0.10:
            warnings.append("High variance in test accuracy (std > 10%)")
        if analysis["accuracy_trend"] < -0.05:
            warnings.append("Degrading performance over time")
        
        analysis["warnings"] = warnings
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) of values over time."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)
    
    def plot_results(self, output_path: str = "validation_results.png"):
        """Plot validation results."""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        folds = [r["fold"] for r in self.results]
        train_acc = [r["train_accuracy"] for r in self.results]
        test_acc = [r["test_accuracy"] for r in self.results]
        overfitting = [r["overfitting_gap"] for r in self.results]
        
        # Plot 1: Train vs Test Accuracy
        axes[0].plot(folds, train_acc, 'o-', label='Train Accuracy', color='blue')
        axes[0].plot(folds, test_acc, 'o-', label='Test Accuracy', color='green')
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Walk-Forward Validation: Accuracy per Fold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Overfitting Gap
        axes[1].bar(folds, overfitting, color='orange', alpha=0.7)
        axes[1].axhline(y=0.15, color='red', linestyle='--', alpha=0.5, label='15% Warning Threshold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Overfitting Gap (Train - Test)')
        axes[1].set_title('Overfitting Analysis')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"\n📊 Results plot saved to: {output_path}")
    
    def save_results(self, output_path: str = "validation_results.json"):
        """Save results to JSON."""
        if not self.results:
            return
        
        analysis = self.analyze_results()
        
        output = {
            "configuration": {
                "train_days": self.train_days,
                "test_days": self.test_days,
                "step_days": self.step_days
            },
            "analysis": analysis,
            "folds": self.results,
            "validated_at": datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")


async def main():
    """Run walk-forward validation."""
    print("\n" + "="*70)
    print("🔄 WALK-FORWARD VALIDATION")
    print("="*70 + "\n")
    
    # Initialize validator
    # 60 days train, 30 days test, 15 days step = ~4 folds in 120 days
    validator = WalkForwardValidator(
        train_days=60,
        test_days=30,
        step_days=15
    )
    
    # Run validation on last 120 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    
    results = await validator.run_validation(start_date, end_date)
    
    if not results:
        print("❌ Validation failed (no results)")
        sys.exit(1)
    
    # Analyze results
    print("\n" + "="*70)
    print("📊 VALIDATION RESULTS")
    print("="*70)
    
    analysis = validator.analyze_results()
    
    print(f"\nFolds Completed: {analysis['num_folds']}")
    print(f"Mean Test Accuracy: {analysis['mean_test_accuracy']:.1%}")
    print(f"Std Test Accuracy: {analysis['std_test_accuracy']:.1%}")
    print(f"Min/Max Accuracy: {analysis['min_test_accuracy']:.1%} / {analysis['max_test_accuracy']:.1%}")
    print(f"Mean Overfitting Gap: {analysis['mean_overfitting_gap']:.1%}")
    print(f"Max Overfitting Gap: {analysis['max_overfitting_gap']:.1%}")
    print(f"Accuracy Trend: {analysis['accuracy_trend']:+.1%} per fold")
    
    if analysis["warnings"]:
        print("\n⚠️  WARNINGS:")
        for warning in analysis["warnings"]:
            print(f"  • {warning}")
    else:
        print("\n✅ No warnings detected")
    
    # Save results
    validator.save_results()
    validator.plot_results()
    
    print("\n" + "="*70)
    print("✅ VALIDATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
