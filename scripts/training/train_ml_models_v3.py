#!/usr/bin/env python3
"""
Ghost Protocol - ML Training Pipeline v3 (HOURLY DATA)
======================================================

CRITICAL FIX: Uses HOURLY data for 48-hour predictions.

Previous versions used DAILY data which only provided ~2 data points
for a 48-hour prediction horizon. This version uses hourly data to give
48 data points of price movement information.

Changes from v2:
- histoday → histohour (hourly bars)
- horizon_days=2 → horizon_hours=48
- Technical indicators adjusted for hourly timeframe
- More data points per symbol (~720 hours = 30 days)

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

# Training symbols
TRAINING_SYMBOLS = [
    "BTC",  # MUST BE FIRST - used for correlation
    "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", 
    "DOT", "LTC", "UNI", "ATOM", "MATIC", "NEAR", "FTM"
]

# Hourly data config
HOURS_OF_DATA = 2000  # ~83 days of hourly data (CryptoCompare limit)
PREDICTION_HORIZON_HOURS = 48  # 48-hour prediction


# ============================================================================
# DATA COLLECTION - HOURLY
# ============================================================================

def fetch_hourly_data(symbol: str, hours: int = HOURS_OF_DATA) -> pd.DataFrame | None:
    """
    Fetch HOURLY OHLCV data from CryptoCompare.
    
    This is the KEY FIX - using hourly instead of daily data.
    For 48h predictions, we need hourly granularity.
    """
    try:
        # CRITICAL: Use histohour instead of histoday
        url = "https://min-api.cryptocompare.com/data/v2/histohour"
        params = {
            "fsym": symbol.upper(), 
            "tsym": "USD", 
            "limit": min(hours, 2000)  # CryptoCompare limit
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") != "Success":
                logger.error(f"CryptoCompare error for {symbol}: {data.get('Message', 'Unknown')}")
                return None
            
            history = data.get("Data", {}).get("Data", [])
            if not history:
                return None
            
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={'volumefrom': 'volume'})
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df[df['close'] > 0]
            
            logger.info(f"  ✅ {symbol}: {len(df)} hourly bars fetched")
            return df
        return None
    except Exception as e:
        logger.error(f"CryptoCompare hourly error for {symbol}: {e}")
        return None


def fetch_fear_greed_index(days: int = 90) -> pd.DataFrame | None:
    """Fetch Fear & Greed Index - daily granularity, we'll forward-fill for hourly."""
    try:
        url = f"https://api.alternative.me/fng/?limit={days}"
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get('data', [])
            
            if not records:
                return None
            
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
            df['fear_greed_value'] = df['value'].astype(int)
            
            # Create numeric classification
            class_map = {
                'Extreme Fear': 0, 'Fear': 1, 'Neutral': 2, 'Greed': 3, 'Extreme Greed': 4
            }
            df['fear_greed_numeric'] = df['value_classification'].map(class_map).fillna(2)
            
            logger.info(f"  ✅ Fear & Greed Index: {len(df)} days loaded")
            return df[['timestamp', 'fear_greed_value', 'fear_greed_numeric']]
        return None
    except Exception as e:
        logger.error(f"Fear & Greed API error: {e}")
        return None


# ============================================================================
# FEATURE ENGINEERING - HOURLY ADJUSTED
# ============================================================================

def calculate_technical_indicators_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for HOURLY data.
    
    Window sizes adjusted for hourly timeframe:
    - Daily windows → multiply by 24 for equivalent hourly
    - Example: 7-day SMA → 168-hour SMA (but we use shorter for responsiveness)
    """
    df = df.copy()
    
    # === MOVING AVERAGES (hourly-adjusted) ===
    # Shorter windows for hourly data - more responsive
    df['SMA_12'] = df['close'].rolling(window=12).mean()   # 12 hours
    df['SMA_24'] = df['close'].rolling(window=24).mean()   # 1 day
    df['SMA_48'] = df['close'].rolling(window=48).mean()   # 2 days
    df['SMA_168'] = df['close'].rolling(window=168).mean() # 7 days
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # === RSI (14-period, standard) ===
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
    
    # === MACD ===
    df['MACD_LINE'] = df['EMA_12'] - df['EMA_26']
    df['MACD_SIGNAL'] = df['MACD_LINE'].ewm(span=9, adjust=False).mean()
    df['MACD_HISTOGRAM'] = df['MACD_LINE'] - df['MACD_SIGNAL']
    df['MACD_BULLISH'] = (df['MACD_LINE'] > df['MACD_SIGNAL']).astype(int)
    
    # === BOLLINGER BANDS (20-hour) ===
    df['BB_MIDDLE'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MIDDLE'] + (bb_std * 2)
    df['BB_LOWER'] = df['BB_MIDDLE'] - (bb_std * 2)
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
    df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'] + 1e-10)
    
    # === STOCHASTIC ===
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['STOCH_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
    
    # === ATR (VOLATILITY) ===
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    # === VOLUME INDICATORS ===
    df['VOLUME_SMA_24'] = df['volume'].rolling(window=24).mean()
    df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_SMA_24'].replace(0, 1e-10)
    df['VOLUME_SPIKE'] = (df['VOLUME_RATIO'] > 2.0).astype(int)
    
    # OBV
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_SMA'] = df['OBV'].rolling(window=24).mean()
    
    # === MOMENTUM (hourly-adjusted) ===
    df['MOMENTUM_1H'] = df['close'].pct_change(1) * 100      # 1 hour
    df['MOMENTUM_4H'] = df['close'].pct_change(4) * 100      # 4 hours
    df['MOMENTUM_24H'] = df['close'].pct_change(24) * 100    # 1 day
    df['MOMENTUM_48H'] = df['close'].pct_change(48) * 100    # 2 days
    df['ROC_12'] = ((df['close'] - df['close'].shift(12)) / df['close'].shift(12) + 1e-10) * 100
    
    # === TREND INDICATORS ===
    df['ABOVE_SMA_24'] = (df['close'] > df['SMA_24']).astype(int)
    df['ABOVE_SMA_48'] = (df['close'] > df['SMA_48']).astype(int)
    df['EMA_BULLISH'] = (df['EMA_12'] > df['EMA_26']).astype(int)
    df['SMA_CROSS_24_48'] = (df['SMA_24'] > df['SMA_48']).astype(int)
    
    # === SUPPORT/RESISTANCE (hourly windows) ===
    df['NEAR_24H_HIGH'] = df['close'] / df['high'].rolling(24).max()
    df['NEAR_24H_LOW'] = df['close'] / df['low'].rolling(24).min()
    df['NEAR_48H_HIGH'] = df['close'] / df['high'].rolling(48).max()
    df['NEAR_48H_LOW'] = df['close'] / df['low'].rolling(48).min()
    
    # === VOLATILITY (hourly-adjusted) ===
    df['VOLATILITY_24H'] = df['close'].rolling(24).std() / df['close'].rolling(24).mean() * 100
    df['VOLATILITY_48H'] = df['close'].rolling(48).std() / df['close'].rolling(48).mean() * 100
    df['HOURLY_RANGE_PCT'] = (df['high'] - df['low']) / df['low'] * 100
    
    return df


def add_btc_correlation_features_hourly(df: pd.DataFrame, btc_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add BTC correlation features - THE MOST IMPORTANT PREDICTOR."""
    df = df.copy()
    
    if symbol == "BTC":
        df['BTC_MOMENTUM_1H'] = df['MOMENTUM_1H']
        df['BTC_MOMENTUM_24H'] = df['MOMENTUM_24H']
        df['BTC_RSI'] = df['RSI_14']
        df['BTC_MACD_BULLISH'] = df['MACD_BULLISH']
        df['BTC_CORRELATION'] = 1.0
        df['BTC_LEADS'] = 0
        return df
    
    # Merge BTC data on timestamp
    btc_features = btc_df[['timestamp', 'close', 'MOMENTUM_1H', 'MOMENTUM_24H', 'RSI_14', 'MACD_BULLISH']].copy()
    btc_features = btc_features.rename(columns={
        'close': 'BTC_PRICE',
        'MOMENTUM_1H': 'BTC_MOMENTUM_1H',
        'MOMENTUM_24H': 'BTC_MOMENTUM_24H',
        'RSI_14': 'BTC_RSI',
        'MACD_BULLISH': 'BTC_MACD_BULLISH'
    })
    
    df = df.merge(btc_features, on='timestamp', how='left')
    
    # Rolling correlation with BTC (48-hour window)
    df['BTC_CORRELATION'] = df['close'].rolling(48).corr(df['BTC_PRICE'])
    
    # Does BTC lead this coin? (BTC moves first, then altcoin follows)
    df['BTC_CHANGE_PREV'] = df['BTC_MOMENTUM_1H'].shift(1)
    df['BTC_LEADS'] = (np.sign(df['BTC_CHANGE_PREV']) == np.sign(df['MOMENTUM_1H'])).astype(int)
    
    # Drop temporary columns
    df = df.drop(columns=['BTC_CHANGE_PREV', 'BTC_PRICE'], errors='ignore')
    
    return df


def add_sentiment_features_hourly(df: pd.DataFrame, fear_greed_df: pd.DataFrame | None) -> pd.DataFrame:
    """Add sentiment features with forward-fill for hourly data."""
    df = df.copy()
    
    if fear_greed_df is not None:
        # Fear & Greed is daily - need to expand to hourly
        fear_greed_df = fear_greed_df.copy()
        fear_greed_df['date'] = fear_greed_df['timestamp'].dt.date
        df['date'] = df['timestamp'].dt.date
        
        df = df.merge(
            fear_greed_df[['date', 'fear_greed_value', 'fear_greed_numeric']], 
            on='date', how='left'
        )
        df = df.drop(columns=['date'])
        
        df['fear_greed_value'] = df['fear_greed_value'].ffill().fillna(50)
        df['fear_greed_numeric'] = df['fear_greed_numeric'].ffill().fillna(2)
        
        # Extreme sentiment zones
        df['EXTREME_FEAR'] = (df['fear_greed_value'] < 25).astype(int)
        df['EXTREME_GREED'] = (df['fear_greed_value'] > 75).astype(int)
    else:
        df['fear_greed_value'] = 50
        df['fear_greed_numeric'] = 2
        df['EXTREME_FEAR'] = 0
        df['EXTREME_GREED'] = 0
    
    return df


def create_target_variable_hourly(df: pd.DataFrame, horizon_hours: int = PREDICTION_HORIZON_HOURS) -> pd.DataFrame:
    """
    Create target variable for 48-HOUR prediction.
    
    This is the KEY FIX - using hourly horizon instead of daily.
    """
    df = df.copy()
    
    # Look ahead 48 hours
    df['FUTURE_PRICE'] = df['close'].shift(-horizon_hours)
    df['FUTURE_CHANGE_PCT'] = (df['FUTURE_PRICE'] - df['close']) / df['close'] * 100
    
    # Binary classification: UP (>2%), DOWN (<-2%)
    # Using 2% threshold for 48h predictions (higher than daily since more time)
    df['TARGET'] = np.where(df['FUTURE_CHANGE_PCT'] > 2, 1, 
                            np.where(df['FUTURE_CHANGE_PCT'] < -2, 0, np.nan))
    
    return df


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def prepare_hourly_training_data(symbols: list[str] = None, hours: int = HOURS_OF_DATA) -> tuple:
    """Prepare training dataset with HOURLY data."""
    if symbols is None:
        symbols = TRAINING_SYMBOLS
    
    logger.info("=" * 60)
    logger.info("📊 PREPARING HOURLY TRAINING DATA (v3)")
    logger.info("=" * 60)
    logger.info(f"   Prediction horizon: {PREDICTION_HORIZON_HOURS} hours")
    logger.info(f"   Data points per symbol: ~{hours} hours")
    
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
    logger.info(f"\n📊 Processing {len(symbols)} symbols with HOURLY data...")
    all_data = []
    
    for i, symbol in enumerate(symbols):
        logger.info(f"  [{i+1}/{len(symbols)}] {symbol}...")
        
        df = fetch_hourly_data(symbol, hours=hours)
        if df is None or len(df) < 200:  # Need at least ~8 days of hourly data
            logger.warning(f"    ⚠️ Insufficient data, skipping")
            continue
        
        # Calculate technical indicators (hourly-adjusted)
        df = calculate_technical_indicators_hourly(df)
        
        # Add BTC correlation (THE KEY FEATURE)
        df = add_btc_correlation_features_hourly(df, btc_df, symbol)
        
        # Add sentiment features
        df = add_sentiment_features_hourly(df, fear_greed_df)
        
        # Create target variable (48-HOUR horizon)
        df = create_target_variable_hourly(df, horizon_hours=PREDICTION_HORIZON_HOURS)
        
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
    
    logger.info(f"\n✅ HOURLY training data prepared:")
    logger.info(f"   Total samples: {len(combined_df)}")
    logger.info(f"   Features: {len(feature_cols)}")
    logger.info(f"   Class balance: UP={int((y==1).sum())}, DOWN={int((y==0).sum())}")
    logger.info(f"   Prediction horizon: {PREDICTION_HORIZON_HOURS} hours")
    
    metadata = {
        "version": "v3-hourly",
        "granularity": "hourly",
        "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
        "symbols": symbols,
        "samples": len(combined_df),
        "features": len(feature_cols),
        "feature_names": feature_cols,
        "class_balance": {"UP": int((y==1).sum()), "DOWN": int((y==0).sum())}
    }
    
    return X, y, feature_cols, metadata


def train_xgboost_model_v3(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Train XGBoost classifier with hourly data."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError:
        return {"ok": False, "error": "XGBoost not installed"}
    
    logger.info("\n" + "=" * 60)
    logger.info("🤖 TRAINING XGBOOST MODEL v3 (HOURLY DATA)")
    logger.info("=" * 60)
    
    # Time-series split (CRITICAL: no data leakage!)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # XGBoost parameters
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Cross-validation
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
    for name, imp in top_features:
        logger.info(f"   {name}: {imp:.4f}")
    
    # Validation check
    ACCEPTANCE_CRITERIA = {
        'min_test_accuracy': 0.55,
        'min_cv_score': 0.52,
        'max_overfit': 0.15
    }
    
    overfit = train_accuracy - test_accuracy
    passed = (
        test_accuracy >= ACCEPTANCE_CRITERIA['min_test_accuracy'] and
        cv_mean >= ACCEPTANCE_CRITERIA['min_cv_score'] and
        overfit <= ACCEPTANCE_CRITERIA['max_overfit']
    )
    
    logger.info(f"\n✅ VALIDATION CHECKS:")
    logger.info(f"   Test Accuracy >= 55%: {'✅' if test_accuracy >= 0.55 else '❌'} ({test_accuracy:.1%})")
    logger.info(f"   CV Score >= 52%: {'✅' if cv_mean >= 0.52 else '❌'} ({cv_mean:.1%})")
    logger.info(f"   Overfit <= 15%: {'✅' if overfit <= 0.15 else '❌'} ({overfit:.1%})")
    logger.info(f"\n   {'✅ MODEL APPROVED FOR PRODUCTION' if passed else '⚠️ MODEL NEEDS IMPROVEMENT'}")
    
    # Save model
    model_path = MODELS_DIR / "ghost_xgboost_v3_hourly.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_score": cv_mean,
            "cv_std": cv_std,
            "feature_importance": importance,
            "top_features": top_features,
            "trained_at": datetime.now().isoformat(),
            "version": "v3.0-hourly",
            "granularity": "hourly",
            "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
            "passed_validation": passed
        }, f)
    
    logger.info(f"\n💾 Model saved: {model_path}")
    
    return {
        "ok": True,
        "model_path": str(model_path),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "cv_score": round(cv_mean, 4),
        "cv_std": round(cv_std, 4),
        "top_features": top_features,
        "passed_validation": passed
    }


def run_v3_training_pipeline() -> dict:
    """Run the complete v3 HOURLY training pipeline."""
    logger.info("=" * 70)
    logger.info("🚀 GHOST PROTOCOL - ML TRAINING v3 (HOURLY DATA)")
    logger.info("=" * 70)
    logger.info("\nKEY FIX: Using HOURLY data instead of DAILY")
    logger.info("  - Previous: Daily bars (2 data points for 48h)")
    logger.info("  - Now: Hourly bars (48 data points for 48h)")
    logger.info(f"  - Prediction horizon: {PREDICTION_HORIZON_HOURS} hours")
    
    start_time = time.time()
    results = {"ok": True, "version": "v3-hourly", "models": {}}
    
    try:
        # Prepare hourly data
        X, y, feature_names, metadata = prepare_hourly_training_data()
        results["data"] = metadata
        
        # Train XGBoost
        xgb_results = train_xgboost_model_v3(X, y, feature_names)
        results["models"]["xgboost"] = xgb_results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        results["ok"] = False
        results["error"] = str(e)
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    results["duration_seconds"] = round(elapsed, 1)
    
    logger.info("\n" + "=" * 70)
    logger.info("📋 FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Version: v3 (HOURLY DATA)")
    logger.info(f"Duration: {elapsed:.1f} seconds")
    
    if results.get("models", {}).get("xgboost", {}).get("ok"):
        xgb = results["models"]["xgboost"]
        logger.info(f"\n🤖 XGBoost v3 (Hourly):")
        logger.info(f"   Test Accuracy: {xgb['test_accuracy']:.1%}")
        logger.info(f"   CV Score: {xgb['cv_score']:.1%}")
        logger.info(f"   Validation: {'✅ PASSED' if xgb['passed_validation'] else '⚠️ NEEDS WORK'}")
    
    logger.info("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    results = run_v3_training_pipeline()
    
    # Save results
    results_path = MODELS_DIR / "training_results_v3.json"
    with open(results_path, "w") as f:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        json.dump(results, f, indent=2, default=convert)
    
    print(f"\n📁 Results saved: {results_path}")
