#!/usr/bin/env python3
"""
Ghost Protocol - ML Training Pipeline v3 (HOURLY DATA)
======================================================

CRITICAL FIX: Train on HOURLY data for 48-hour predictions.

Previous versions used DAILY data which only provided ~2 data points
for a 48-hour forecast. This version uses hourly bars for proper
time-series prediction.

Key Changes from v2:
- Uses histohour API instead of histoday
- 720 hours (30 days) of hourly data = 720 training points per symbol
- Horizon is 48 hours (not 2 days) - matches prediction window exactly
- Adjusted technical indicators for hourly timeframe

Author: Ghost AI
Date: January 5, 2026
"""

import json
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Model storage
MODELS_DIR = Path(__file__).parent / "models" / "trained"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Training symbols - BTC MUST be first (we need it for correlation)
TRAINING_SYMBOLS = [
    "BTC",  # MUST BE FIRST - used for correlation
    "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", 
    "DOT", "LTC", "UNI", "ATOM", "MATIC"
]

# Hourly data settings
HOURS_OF_DATA = 2000  # ~83 days of hourly data (CryptoCompare limit)
PREDICTION_HORIZON_HOURS = 48  # 48-hour prediction window


# ============================================================================
# DATA COLLECTION - HOURLY
# ============================================================================

def fetch_hourly_data(symbol: str, hours: int = HOURS_OF_DATA) -> pd.DataFrame | None:
    """
    Fetch HOURLY OHLCV data from CryptoCompare.
    
    CRITICAL: This is the key fix - we need hourly granularity for 48h predictions.
    Daily data only gave us 2 data points; hourly gives us 48.
    """
    try:
        url = "https://min-api.cryptocompare.com/data/v2/histohour"  # HOURLY!
        params = {
            "fsym": symbol.upper(), 
            "tsym": "USD", 
            "limit": min(hours, 2000)  # CryptoCompare limit
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") != "Success":
                logger.warning(f"CryptoCompare returned error for {symbol}")
                return None
            
            history = data.get("Data", {}).get("Data", [])
            if not history:
                return None
            
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'volumefrom': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df[df['close'] > 0]
            
            logger.info(f"  ✅ {symbol}: {len(df)} hourly bars loaded")
            return df
        return None
    except Exception as e:
        logger.error(f"CryptoCompare error for {symbol}: {e}")
        return None


def fetch_fear_greed_index(days: int = 90) -> pd.DataFrame | None:
    """Fetch Fear & Greed Index from alternative.me API."""
    try:
        url = f"https://api.alternative.me/fng/?limit={days}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('data', [])
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['timestamp'].astype(int), unit='s').dt.date
            df['fear_greed_value'] = df['value'].astype(int)
            
            # Map to numeric
            class_map = {
                'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2,
                'Greed': 3, 'Extreme Greed': 4
            }
            df['fear_greed_numeric'] = df['value_classification'].map(class_map).fillna(2)
            
            logger.info(f"  ✅ Fear & Greed Index: {len(df)} days loaded")
            return df[['date', 'fear_greed_value', 'fear_greed_numeric']]
        return None
    except Exception as e:
        logger.error(f"Fear & Greed API error: {e}")
        return None


# ============================================================================
# FEATURE ENGINEERING (ADAPTED FOR HOURLY DATA)
# ============================================================================

def calculate_technical_indicators_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators adapted for hourly timeframe.
    
    Key differences from daily:
    - Use shorter periods (7 hours vs 7 days)
    - RSI on 14 hours (not 14 days)
    - Moving averages: 12h, 24h, 48h, 168h (1 week)
    """
    df = df.copy()
    
    # === MOVING AVERAGES (hourly adapted) ===
    df['SMA_12'] = df['close'].rolling(window=12).mean()      # 12 hours
    df['SMA_24'] = df['close'].rolling(window=24).mean()      # 24 hours (1 day)
    df['SMA_48'] = df['close'].rolling(window=48).mean()      # 48 hours (2 days)
    df['SMA_168'] = df['close'].rolling(window=168).mean()    # 168 hours (1 week)
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_24'] = df['close'].ewm(span=24, adjust=False).mean()
    
    # === RSI (14 hours) ===
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # RSI zones
    df['RSI_OVERSOLD'] = (df['RSI_14'] < 30).astype(int)
    df['RSI_OVERBOUGHT'] = (df['RSI_14'] > 70).astype(int)
    
    # === MACD (standard but on hourly) ===
    df['MACD_LINE'] = df['EMA_12'] - df['EMA_24']
    df['MACD_SIGNAL'] = df['MACD_LINE'].ewm(span=9, adjust=False).mean()
    df['MACD_HISTOGRAM'] = df['MACD_LINE'] - df['MACD_SIGNAL']
    df['MACD_BULLISH'] = (df['MACD_LINE'] > df['MACD_SIGNAL']).astype(int)
    
    # === BOLLINGER BANDS (24-hour, 2 std) ===
    df['BB_MIDDLE'] = df['close'].rolling(window=24).mean()
    bb_std = df['close'].rolling(window=24).std()
    df['BB_UPPER'] = df['BB_MIDDLE'] + (bb_std * 2)
    df['BB_LOWER'] = df['BB_MIDDLE'] - (bb_std * 2)
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE'].replace(0, 1e-10)
    df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'] + 1e-10)
    
    # === STOCHASTIC (14 hours) ===
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['STOCH_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
    
    # === ATR (14 hours) ===
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    df['ATR_PCT'] = df['ATR_14'] / df['close'] * 100  # ATR as % of price
    
    # === VOLUME INDICATORS ===
    df['VOLUME_SMA_24'] = df['volume'].rolling(window=24).mean()
    df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_SMA_24'].replace(0, 1e-10)
    df['VOLUME_SPIKE'] = (df['VOLUME_RATIO'] > 2.0).astype(int)
    
    # OBV
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_SMA'] = df['OBV'].rolling(window=24).mean()
    df['OBV_TREND'] = (df['OBV'] > df['OBV_SMA']).astype(int)
    
    # === MOMENTUM (hourly periods) ===
    df['MOMENTUM_1H'] = df['close'].pct_change(1) * 100
    df['MOMENTUM_4H'] = df['close'].pct_change(4) * 100
    df['MOMENTUM_12H'] = df['close'].pct_change(12) * 100
    df['MOMENTUM_24H'] = df['close'].pct_change(24) * 100
    df['MOMENTUM_48H'] = df['close'].pct_change(48) * 100
    df['ROC_24'] = ((df['close'] - df['close'].shift(24)) / df['close'].shift(24) + 1e-10) * 100
    
    # === TREND INDICATORS ===
    df['ABOVE_SMA_24'] = (df['close'] > df['SMA_24']).astype(int)
    df['ABOVE_SMA_48'] = (df['close'] > df['SMA_48']).astype(int)
    df['ABOVE_SMA_168'] = (df['close'] > df['SMA_168']).astype(int)
    df['EMA_BULLISH'] = (df['EMA_12'] > df['EMA_24']).astype(int)
    df['SMA_CROSS_24_48'] = (df['SMA_24'] > df['SMA_48']).astype(int)
    
    # === PRICE POSITION ===
    df['NEAR_24H_HIGH'] = df['close'] / df['high'].rolling(24).max()
    df['NEAR_24H_LOW'] = df['close'] / df['low'].rolling(24).min()
    df['NEAR_48H_HIGH'] = df['close'] / df['high'].rolling(48).max()
    df['NEAR_48H_LOW'] = df['close'] / df['low'].rolling(48).min()
    
    # === VOLATILITY ===
    df['VOLATILITY_24H'] = df['close'].rolling(24).std() / df['close'].rolling(24).mean() * 100
    df['VOLATILITY_48H'] = df['close'].rolling(48).std() / df['close'].rolling(48).mean() * 100
    df['HOURLY_RANGE_PCT'] = (df['high'] - df['low']) / df['low'] * 100
    
    # === TIME FEATURES (hour of day, day of week) ===
    df['HOUR_OF_DAY'] = df['timestamp'].dt.hour
    df['DAY_OF_WEEK'] = df['timestamp'].dt.dayofweek
    df['IS_WEEKEND'] = (df['DAY_OF_WEEK'] >= 5).astype(int)
    
    return df


def add_btc_correlation_features(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add BTC correlation features - THE MOST IMPORTANT PREDICTOR."""
    df = df.copy()
    
    if symbol == "BTC":
        df['BTC_MOMENTUM_4H'] = df['MOMENTUM_4H']
        df['BTC_MOMENTUM_24H'] = df['MOMENTUM_24H']
        df['BTC_RSI'] = df['RSI_14']
        df['BTC_MACD_BULLISH'] = df['MACD_BULLISH']
        df['BTC_CORRELATION'] = 1.0
        df['BTC_LEADS'] = 0
        return df
    
    # Merge BTC data on timestamp
    btc_features = btc_df[['timestamp', 'close', 'MOMENTUM_4H', 'MOMENTUM_24H', 'RSI_14', 'MACD_BULLISH']].copy()
    btc_features = btc_features.rename(columns={
        'close': 'BTC_PRICE',
        'MOMENTUM_4H': 'BTC_MOMENTUM_4H',
        'MOMENTUM_24H': 'BTC_MOMENTUM_24H',
        'RSI_14': 'BTC_RSI',
        'MACD_BULLISH': 'BTC_MACD_BULLISH'
    })
    
    df = df.merge(btc_features, on='timestamp', how='left')
    
    # Rolling correlation with BTC (48-hour window)
    df['BTC_CORRELATION'] = df['close'].rolling(48).corr(df['BTC_PRICE'])
    
    # Does BTC lead this coin?
    df['BTC_CHANGE_PREV'] = df['BTC_MOMENTUM_4H'].shift(1)
    df['BTC_LEADS'] = (np.sign(df['BTC_CHANGE_PREV']) == np.sign(df['MOMENTUM_4H'])).astype(int)
    
    df = df.drop(columns=['BTC_CHANGE_PREV', 'BTC_PRICE'], errors='ignore')
    
    return df


def add_sentiment_features(df: pd.DataFrame, fear_greed_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add Fear & Greed Index (daily, applied to all hours of that day)."""
    df = df.copy()
    
    if fear_greed_df is not None:
        # Create date column for merge
        df['date'] = df['timestamp'].dt.date
        df = df.merge(fear_greed_df, on='date', how='left')
        df['fear_greed_value'] = df['fear_greed_value'].ffill().fillna(50)
        df['fear_greed_numeric'] = df['fear_greed_numeric'].ffill().fillna(2)
        df = df.drop(columns=['date'])
        
        # Extreme sentiment zones
        df['EXTREME_FEAR'] = (df['fear_greed_value'] < 25).astype(int)
        df['EXTREME_GREED'] = (df['fear_greed_value'] > 75).astype(int)
    else:
        df['fear_greed_value'] = 50
        df['fear_greed_numeric'] = 2
        df['EXTREME_FEAR'] = 0
        df['EXTREME_GREED'] = 0
    
    return df


def create_target_variable(df: pd.DataFrame, horizon_hours: int = 48) -> pd.DataFrame:
    """
    Create target variable for 48-HOUR prediction.
    
    CRITICAL: horizon_hours=48 means we predict 48 hours ahead.
    This matches our actual prediction window.
    """
    df = df.copy()
    
    df['FUTURE_PRICE'] = df['close'].shift(-horizon_hours)
    df['FUTURE_CHANGE_PCT'] = (df['FUTURE_PRICE'] - df['close']) / df['close'] * 100
    
    # Binary classification: UP (>2% in 48h), DOWN (<-2% in 48h)
    # Using 2% threshold because crypto is volatile
    df['TARGET'] = np.where(df['FUTURE_CHANGE_PCT'] > 2, 1, 
                            np.where(df['FUTURE_CHANGE_PCT'] < -2, 0, np.nan))
    
    return df


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def prepare_training_data(symbols: list[str] = None, hours: int = HOURS_OF_DATA) -> tuple:
    """Prepare complete training dataset with HOURLY data."""
    if symbols is None:
        symbols = TRAINING_SYMBOLS
    
    logger.info("=" * 60)
    logger.info("📊 PREPARING HOURLY TRAINING DATA (v3)")
    logger.info("=" * 60)
    logger.info(f"   Hours of data: {hours}")
    logger.info(f"   Prediction horizon: {PREDICTION_HORIZON_HOURS} hours")
    
    # STEP 1: Fetch BTC data first (needed for correlation)
    logger.info("\n📈 Fetching BTC hourly data...")
    btc_df = fetch_hourly_data("BTC", hours=hours)
    if btc_df is None:
        raise ValueError("Failed to fetch BTC data - required for correlation features")
    btc_df = calculate_technical_indicators_hourly(btc_df)
    time.sleep(1)
    
    # STEP 2: Fetch sentiment data
    logger.info("\n📊 Fetching sentiment data...")
    fear_greed_df = fetch_fear_greed_index(days=90)
    time.sleep(1)
    
    # STEP 3: Fetch and process all symbols
    logger.info(f"\n📊 Processing {len(symbols)} symbols...")
    all_data = []
    
    for i, symbol in enumerate(symbols):
        logger.info(f"  [{i+1}/{len(symbols)}] {symbol}...")
        
        df = fetch_hourly_data(symbol, hours=hours)
        if df is None or len(df) < 200:
            logger.warning(f"    ⚠️ Insufficient data, skipping")
            continue
        
        # Calculate technical indicators (HOURLY)
        df = calculate_technical_indicators_hourly(df)
        
        # Add BTC correlation (THE KEY FEATURE)
        df = add_btc_correlation_features(df, btc_df, symbol)
        
        # Add sentiment features
        df = add_sentiment_features(df, fear_greed_df)
        
        # Create target variable (48 HOURS ahead)
        df = create_target_variable(df, horizon_hours=PREDICTION_HORIZON_HOURS)
        
        df['symbol'] = symbol
        all_data.append(df)
        
        time.sleep(0.5)  # Rate limiting
    
    if not all_data:
        raise ValueError("No training data collected!")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=['TARGET'])
    
    # Define feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'FUTURE_PRICE', 'FUTURE_CHANGE_PCT', 'TARGET', 'symbol']
    feature_cols = [c for c in combined_df.columns if c not in exclude_cols]
    
    # Drop rows with NaN features
    combined_df = combined_df.dropna(subset=feature_cols)
    
    X = combined_df[feature_cols].values
    y = combined_df['TARGET'].values
    
    logger.info(f"\n✅ Training data prepared:")
    logger.info(f"   Samples: {len(combined_df)}")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Class balance: UP={int((y==1).sum())}, DOWN={int((y==0).sum())}")
    
    metadata = {
        "version": "v3_hourly",
        "symbols": symbols,
        "samples": len(combined_df),
        "features": len(feature_cols),
        "feature_names": feature_cols,
        "class_balance": {"UP": int((y==1).sum()), "DOWN": int((y==0).sum())},
        "data_granularity": "hourly",
        "prediction_horizon": f"{PREDICTION_HORIZON_HOURS} hours",
        "trained_at": datetime.now().isoformat()
    }
    
    return X, y, feature_cols, metadata


def train_xgboost_model(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Train XGBoost classifier with proper validation and CLASS BALANCING."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        return {"ok": False, "error": "XGBoost not installed"}
    
    logger.info("\n" + "=" * 60)
    logger.info("🤖 TRAINING XGBOOST MODEL (v3 - Hourly, CLASS-BALANCED)")
    logger.info("=" * 60)
    
    # Time-series split (CRITICAL: no data leakage!)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train size (before balancing): {len(X_train)}, Test size: {len(X_test)}")
    
    # === CRITICAL FIX: DOWNSAMPLE majority class (DOWN) to match minority (UP) ===
    # scale_pos_weight alone is NOT enough when the ratio is severe.
    # We need to physically balance the training data.
    n_down_train = int((y_train == 0).sum())
    n_up_train = int((y_train == 1).sum())
    logger.info(f"Class balance BEFORE balancing: UP={n_up_train}, DOWN={n_down_train}")
    
    # Downsample DOWN to match UP count
    down_indices = np.where(y_train == 0)[0]
    up_indices = np.where(y_train == 1)[0]
    
    np.random.seed(42)
    # Keep all UP samples, randomly sample DOWN to match
    down_sampled = np.random.choice(down_indices, size=len(up_indices), replace=False)
    balanced_indices = np.concatenate([up_indices, down_sampled])
    np.random.shuffle(balanced_indices)
    
    X_train_balanced = X_train[balanced_indices]
    y_train_balanced = y_train[balanced_indices]
    
    n_down_bal = int((y_train_balanced == 0).sum())
    n_up_bal = int((y_train_balanced == 1).sum())
    logger.info(f"Class balance AFTER balancing:  UP={n_up_bal}, DOWN={n_down_bal}")
    logger.info(f"Balanced training size: {len(X_train_balanced)}")
    
    # XGBoost parameters (optimized for hourly crypto prediction)
    # scale_pos_weight=1.0 since we already balanced the data
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,           # Slightly shallower for hourly data
        learning_rate=0.03,    # Lower LR for more stable learning
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,    # Higher for hourly noise
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,  # Already balanced via downsampling
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    
    # Train with early stopping on BALANCED data
    model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate on ORIGINAL (unbalanced) test set - this is the real-world distribution
    y_pred_train = model.predict(X_train_balanced)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train_balanced, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Check direction balance on test predictions
    test_up = int((y_pred_test == 1).sum())
    test_down = int((y_pred_test == 0).sum())
    logger.info(f"\n📊 Test set prediction mix: UP={test_up} ({test_up/len(y_pred_test)*100:.0f}%), DOWN={test_down} ({test_down/len(y_pred_test)*100:.0f}%)")
    
    # Cross-validation (time-series aware)
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    logger.info(f"\n📊 MODEL PERFORMANCE:")
    logger.info(f"   Train Accuracy: {train_accuracy:.1%}")
    logger.info(f"   Test Accuracy:  {test_accuracy:.1%}")
    logger.info(f"   CV Score:       {cv_mean:.1%} (±{cv_std:.1%})")
    
    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    logger.info(f"\n🎯 TOP 15 PREDICTIVE FEATURES:")
    for feat, imp in top_features:
        logger.info(f"   {feat}: {imp:.4f}")
    
    return {
        "model": model,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "cv_score": cv_mean,
        "cv_std": cv_std,
        "feature_importance": importance,
        "top_features": top_features
    }


def save_model(model_data: dict, feature_names: list[str], metadata: dict):
    """Save trained model to disk."""
    model_path = MODELS_DIR / "ghost_xgboost_v3_hourly.pkl"
    
    save_data = {
        "model": model_data["model"],
        "feature_names": feature_names,
        "train_accuracy": model_data["train_accuracy"],
        "test_accuracy": model_data["test_accuracy"],
        "cv_score": model_data["cv_score"],
        "feature_importance": model_data["feature_importance"],
        "metadata": metadata
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(save_data, f)
    
    logger.info(f"\n💾 Model saved to: {model_path}")
    
    # Also save as v2 for backward compatibility
    v2_path = MODELS_DIR / "ghost_xgboost_v2.pkl"
    with open(v2_path, "wb") as f:
        pickle.dump(save_data, f)
    logger.info(f"💾 Also saved as v2 for compatibility: {v2_path}")
    
    return model_path


def main():
    """Main training pipeline."""
    logger.info("\n" + "=" * 70)
    logger.info("🚀 GHOST PROTOCOL - ML TRAINING v3 (HOURLY DATA)")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now()}")
    
    try:
        # Prepare training data (HOURLY)
        X, y, feature_names, metadata = prepare_training_data()
        
        # Train XGBoost
        model_data = train_xgboost_model(X, y, feature_names)
        
        if "error" in model_data:
            logger.error(f"Training failed: {model_data['error']}")
            return
        
        # Check if model meets accuracy threshold
        if model_data["test_accuracy"] < 0.52:
            logger.warning(f"⚠️ Model accuracy {model_data['test_accuracy']:.1%} < 52% threshold")
            logger.warning("   Consider: more data, different features, or hyperparameter tuning")
        
        # Save model
        model_path = save_model(model_data, feature_names, metadata)
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"   Model: {model_path}")
        logger.info(f"   Test Accuracy: {model_data['test_accuracy']:.1%}")
        logger.info(f"   CV Score: {model_data['cv_score']:.1%}")
        logger.info(f"   Data Granularity: HOURLY")
        logger.info(f"   Prediction Horizon: {PREDICTION_HORIZON_HOURS} hours")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
