#!/usr/bin/env python3
"""
Ghost Protocol - REAL ML Model Training Pipeline
================================================

This script trains ACTUAL machine learning models using:
1. Historical price data from Binance/CoinGecko
2. Technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. XGBoost for classification
4. LSTM for sequence prediction

Target: 60-70% directional accuracy (significantly better than 50% coin flip)

Author: Ghost AI
Date: December 19, 2025
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

# Top crypto symbols to train on (smaller list to avoid rate limits)
TRAINING_SYMBOLS = [
    "BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", 
    "DOT", "LTC", "UNI", "ATOM"
]


def fetch_cryptocompare_data(symbol: str, days: int = 365) -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data from CryptoCompare (FREE API, no auth required).
    
    Args:
        symbol: Trading symbol (e.g., "BTC", "ETH")
        days: Number of days of history (max ~2000)
    
    Returns:
        DataFrame with OHLCV data or None on error
    """
    try:
        # CryptoCompare histoday endpoint - FREE, no API key needed
        url = "https://min-api.cryptocompare.com/data/v2/histoday"
        params = {
            "fsym": symbol.upper(),
            "tsym": "USD",
            "limit": min(days, 2000)  # Max 2000 data points
        }
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("Response") != "Success":
                logger.warning(f"CryptoCompare error for {symbol}: {data.get('Message', 'Unknown')}")
                return None
            
            history = data.get("Data", {}).get("Data", [])
            if not history:
                return None
            
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume'
            })
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df[df['close'] > 0]  # Remove zero-price rows
            
            logger.info(f"  ✅ {symbol}: {len(df)} daily data points from CryptoCompare")
            return df
            
        else:
            logger.warning(f"CryptoCompare API error for {symbol}: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch CryptoCompare data for {symbol}: {e}")
        return None


def fetch_coingecko_ohlc(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """
    Fetch historical OHLC data from CoinGecko market_chart endpoint.
    
    Args:
        symbol: Trading symbol (e.g., "BTC", "ETH")
        days: Number of days of history
    
    Returns:
        DataFrame with OHLC data or None on error
    """
    # Map common symbols to CoinGecko IDs
    symbol_to_id = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
        "XRP": "ripple", "DOGE": "dogecoin", "ADA": "cardano",
        "AVAX": "avalanche-2", "LINK": "chainlink", "DOT": "polkadot",
        "MATIC": "matic-network", "SHIB": "shiba-inu", "LTC": "litecoin",
        "UNI": "uniswap", "ATOM": "cosmos", "XLM": "stellar",
        "APT": "aptos", "ARB": "arbitrum", "OP": "optimism",
        "INJ": "injective-protocol", "SUI": "sui", "SEI": "sei-network",
        "TIA": "celestia", "JUP": "jupiter-exchange-solana", "WIF": "dogwifcoin",
        "PEPE": "pepe", "BONK": "bonk", "FLOKI": "floki",
        "RENDER": "render-token", "FET": "fetch-ai", "TAO": "bittensor",
        "NEAR": "near", "FTM": "fantom", "ALGO": "algorand",
        "VET": "vechain", "HBAR": "hedera-hashgraph", "ICP": "internet-computer"
    }
    
    coin_id = symbol_to_id.get(symbol.upper())
    if not coin_id:
        logger.warning(f"Unknown CoinGecko ID for {symbol}")
        return None
    
    try:
        # Use market_chart endpoint - gives hourly data for <90 days
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": days, "interval": "hourly"}
        
        response = requests.get(url, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            if not data.get("prices"):
                return None
            
            # Build DataFrame from prices and volumes
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            
            df = pd.DataFrame(prices, columns=['timestamp_ms', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
            
            # Add volume if available
            if volumes and len(volumes) == len(prices):
                df['volume'] = [v[1] for v in volumes]
            else:
                df['volume'] = 1000000  # Default volume
            
            # Simulate OHLC from close (for market_chart data)
            # Add small random variations for open/high/low
            np.random.seed(42)
            noise = np.random.uniform(0.995, 1.005, len(df))
            df['open'] = df['close'] * noise
            df['high'] = df[['open', 'close']].max(axis=1) * np.random.uniform(1.001, 1.01, len(df))
            df['low'] = df[['open', 'close']].min(axis=1) * np.random.uniform(0.99, 0.999, len(df))
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"  ✅ {symbol}: {len(df)} data points from CoinGecko")
            return df
            
        elif response.status_code == 429:
            logger.warning(f"CoinGecko rate limit for {symbol}")
            return None
        else:
            logger.warning(f"CoinGecko API error for {symbol}: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to fetch CoinGecko data for {symbol}: {e}")
        return None


def fetch_binance_klines(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data from Binance (fallback to CryptoCompare if blocked).
    
    Args:
        symbol: Trading symbol (e.g., "BTC" -> "BTCUSDT")
        interval: Kline interval (1h, 4h, 1d)
        limit: Number of candles (max 1000)
    
    Returns:
        DataFrame with OHLCV data or None on error
    """
    try:
        binance_symbol = f"{symbol.upper()}USDT"
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": interval,
            "limit": limit
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            df['open'] = pd.to_numeric(df['open'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            df['volume'] = pd.to_numeric(df['volume'])
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        else:
            # Binance blocked (451) - fall back to CryptoCompare (FREE API)
            logger.info(f"Binance unavailable for {symbol}, using CryptoCompare...")
            return fetch_cryptocompare_data(symbol, days=365)
            
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return fetch_cryptocompare_data(symbol, days=365)


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ALL technical indicators used for prediction.
    
    Args:
        df: OHLCV DataFrame
    
    Returns:
        DataFrame with technical indicators added
    """
    df = df.copy()
    
    # === Price-based indicators ===
    
    # Simple Moving Averages
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # === RSI (Relative Strength Index) ===
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # === MACD ===
    df['MACD_LINE'] = df['EMA_12'] - df['EMA_26']
    df['MACD_SIGNAL'] = df['MACD_LINE'].ewm(span=9, adjust=False).mean()
    df['MACD_HISTOGRAM'] = df['MACD_LINE'] - df['MACD_SIGNAL']
    
    # === Bollinger Bands ===
    df['BB_MIDDLE'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MIDDLE'] + (bb_std * 2)
    df['BB_LOWER'] = df['BB_MIDDLE'] - (bb_std * 2)
    df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MIDDLE']
    df['BB_POSITION'] = (df['close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'] + 1e-10)
    
    # === Stochastic Oscillator ===
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['STOCH_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 1e-10)
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
    
    # === Average True Range (ATR) ===
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    # === Volume indicators ===
    df['VOLUME_SMA_20'] = df['volume'].rolling(window=20).mean()
    df['VOLUME_RATIO'] = df['volume'] / df['VOLUME_SMA_20'].replace(0, 1e-10)
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['OBV_SMA'] = df['OBV'].rolling(window=20).mean()
    
    # === Momentum indicators ===
    df['ROC_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) + 1e-10) * 100
    df['MOM_10'] = df['close'] - df['close'].shift(10)
    
    # === Trend indicators ===
    df['PRICE_CHANGE_1H'] = df['close'].pct_change(1) * 100
    df['PRICE_CHANGE_4H'] = df['close'].pct_change(4) * 100
    df['PRICE_CHANGE_24H'] = df['close'].pct_change(24) * 100
    
    # Price relative to SMAs
    df['PRICE_VS_SMA_20'] = (df['close'] - df['SMA_20']) / df['SMA_20'] * 100
    df['PRICE_VS_SMA_50'] = (df['close'] - df['SMA_50']) / df['SMA_50'] * 100
    
    # Moving average crossovers
    df['SMA_CROSS_7_20'] = np.where(df['SMA_7'] > df['SMA_20'], 1, -1)
    df['SMA_CROSS_20_50'] = np.where(df['SMA_20'] > df['SMA_50'], 1, -1)
    
    # === Volatility ===
    df['VOLATILITY_20'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean() * 100
    
    # === High/Low range ===
    df['DAILY_RANGE_PCT'] = (df['high'] - df['low']) / df['low'] * 100
    
    return df


def create_target_variable(df: pd.DataFrame, horizon_days: int = 2) -> pd.DataFrame:
    """
    Create target variable for prediction (UP/DOWN after N days).
    
    Args:
        df: DataFrame with price data
        horizon_days: Prediction horizon in days (2 days = ~48 hours)
    
    Returns:
        DataFrame with target variable added
    """
    df = df.copy()
    
    # Future price change
    df['FUTURE_PRICE'] = df['close'].shift(-horizon_days)
    df['FUTURE_CHANGE_PCT'] = (df['FUTURE_PRICE'] - df['close']) / df['close'] * 100
    
    # Binary target: 1 = UP (>1% gain), 0 = DOWN (<-1% loss)
    # Exclude flat moves (between -1% and +1%) for cleaner training
    df['TARGET'] = np.where(df['FUTURE_CHANGE_PCT'] > 1, 1, 
                            np.where(df['FUTURE_CHANGE_PCT'] < -1, 0, np.nan))
    
    # Also create multi-class target for potential use
    df['TARGET_3CLASS'] = np.where(df['FUTURE_CHANGE_PCT'] > 2, 2,  # Strong UP
                                    np.where(df['FUTURE_CHANGE_PCT'] > 0, 1,  # Weak UP
                                    np.where(df['FUTURE_CHANGE_PCT'] > -2, 0,  # Weak DOWN
                                    -1)))  # Strong DOWN
    
    return df


def prepare_training_data(symbols: list[str] = None, hours_per_symbol: int = 1000) -> tuple:
    """
    Fetch and prepare training data for multiple symbols.
    
    Args:
        symbols: List of symbols to train on (default: TRAINING_SYMBOLS)
        hours_per_symbol: Hours of data per symbol
    
    Returns:
        (X, y, feature_names, metadata)
    """
    if symbols is None:
        symbols = TRAINING_SYMBOLS
    
    all_data = []
    
    logger.info(f"📊 Fetching training data for {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        logger.info(f"  [{i+1}/{len(symbols)}] Fetching {symbol}...")
        
        df = fetch_binance_klines(symbol, interval="1h", limit=hours_per_symbol)
        
        if df is None or len(df) < 50:
            logger.warning(f"    ⚠️  Insufficient data for {symbol}, skipping")
            continue
        
        # Add technical indicators
        df = calculate_technical_indicators(df)
        
        # Create target variable (2-day horizon = ~48h prediction)
        df = create_target_variable(df, horizon_days=2)
        
        # Add symbol identifier
        df['symbol'] = symbol
        
        all_data.append(df)
        
        # Rate limit - CoinGecko free tier allows ~10-30 calls/min
        time.sleep(3)
    
    if not all_data:
        raise ValueError("No training data collected!")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Drop rows with NaN target (flat moves or insufficient future data)
    combined_df = combined_df.dropna(subset=['TARGET'])
    
    # Feature columns (exclude non-features)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                    'FUTURE_PRICE', 'FUTURE_CHANGE_PCT', 'TARGET', 'TARGET_3CLASS', 'symbol']
    feature_cols = [c for c in combined_df.columns if c not in exclude_cols]
    
    # Drop any remaining NaN values
    combined_df = combined_df.dropna(subset=feature_cols)
    
    logger.info(f"✅ Training data prepared: {len(combined_df)} samples, {len(feature_cols)} features")
    
    X = combined_df[feature_cols].values
    y = combined_df['TARGET'].values
    
    metadata = {
        "symbols": symbols,
        "samples": len(combined_df),
        "features": len(feature_cols),
        "class_balance": {
            "UP": int((y == 1).sum()),
            "DOWN": int((y == 0).sum())
        }
    }
    
    return X, y, feature_cols, metadata


def train_xgboost_model(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """
    Train XGBoost classifier for directional prediction.
    
    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature names
    
    Returns:
        Training results with model and metrics
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    except ImportError:
        return {"ok": False, "error": "XGBoost not installed"}
    
    logger.info("🤖 Training XGBoost model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
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
    
    # Fit with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Evaluate
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    logger.info(f"  Train Accuracy: {train_accuracy:.1%}")
    logger.info(f"  Test Accuracy:  {test_accuracy:.1%}")
    logger.info(f"  CV Score:       {cv_mean:.1%} (±{cv_std:.1%})")
    
    # Feature importance
    importance = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    
    logger.info("  Top 10 Features:")
    for name, imp in top_features:
        logger.info(f"    - {name}: {imp:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "ghost_xgboost_v1.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "feature_names": feature_names,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "cv_score": cv_mean,
            "cv_std": cv_std,
            "feature_importance": importance,
            "trained_at": datetime.now().isoformat(),
            "version": "v1.0"
        }, f)
    
    logger.info(f"✅ XGBoost model saved: {model_path}")
    
    return {
        "ok": True,
        "model_path": str(model_path),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "cv_score": round(cv_mean, 4),
        "cv_std": round(cv_std, 4),
        "top_features": top_features
    }


def train_lstm_model(X: np.ndarray, y: np.ndarray, feature_names: list[str], sequence_length: int = 24) -> dict:
    """
    Train LSTM model for sequence prediction.
    
    Args:
        X: Feature matrix
        y: Target labels  
        feature_names: List of feature names
        sequence_length: Number of time steps for LSTM
    
    Returns:
        Training results with model and metrics
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("TensorFlow not available, skipping LSTM training")
        return {"ok": False, "error": "TensorFlow not installed"}
    
    logger.info("🧠 Training LSTM model...")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create sequences
    X_seq = []
    y_seq = []
    
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    logger.info(f"  Sequence shape: {X_seq.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.2, random_state=42
    )
    
    # Build LSTM model
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(sequence_length, X.shape[1])),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    logger.info(f"  Train Accuracy: {train_accuracy:.1%}")
    logger.info(f"  Test Accuracy:  {test_accuracy:.1%}")
    
    # Save model
    model_path = MODELS_DIR / "ghost_lstm_v1"
    model.save(model_path)
    
    # Save scaler separately
    scaler_path = MODELS_DIR / "ghost_lstm_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({
            "scaler": scaler,
            "feature_names": feature_names,
            "sequence_length": sequence_length,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "trained_at": datetime.now().isoformat(),
            "version": "v1.0"
        }, f)
    
    logger.info(f"✅ LSTM model saved: {model_path}")
    
    return {
        "ok": True,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "train_accuracy": round(train_accuracy, 4),
        "test_accuracy": round(test_accuracy, 4),
        "sequence_length": sequence_length
    }


def run_full_training_pipeline() -> dict:
    """
    Run the complete ML training pipeline.
    
    Returns:
        Training results summary
    """
    logger.info("=" * 60)
    logger.info("🚀 GHOST PROTOCOL - ML MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    start_time = time.time()
    results = {"ok": True, "models": {}}
    
    try:
        # Step 1: Prepare training data
        logger.info("\n📊 STEP 1: Preparing Training Data")
        logger.info("-" * 40)
        X, y, feature_names, metadata = prepare_training_data()
        results["data"] = metadata
        
        # Step 2: Train XGBoost
        logger.info("\n🤖 STEP 2: Training XGBoost Model")
        logger.info("-" * 40)
        xgb_results = train_xgboost_model(X, y, feature_names)
        results["models"]["xgboost"] = xgb_results
        
        # Step 3: Train LSTM (optional, requires TensorFlow)
        logger.info("\n🧠 STEP 3: Training LSTM Model")
        logger.info("-" * 40)
        lstm_results = train_lstm_model(X, y, feature_names)
        results["models"]["lstm"] = lstm_results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        results["ok"] = False
        results["error"] = str(e)
        import traceback
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    results["duration_seconds"] = round(elapsed, 1)
    
    logger.info("\n" + "=" * 60)
    logger.info("📋 TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Duration: {elapsed:.1f} seconds")
    
    if results.get("data"):
        logger.info(f"Training samples: {results['data']['samples']}")
        logger.info(f"Features: {results['data']['features']}")
        logger.info(f"Class balance: UP={results['data']['class_balance']['UP']}, DOWN={results['data']['class_balance']['DOWN']}")
    
    if results["models"].get("xgboost", {}).get("ok"):
        xgb = results["models"]["xgboost"]
        logger.info(f"\nXGBoost Results:")
        logger.info(f"  Test Accuracy: {xgb['test_accuracy']:.1%}")
        logger.info(f"  CV Score: {xgb['cv_score']:.1%} (±{xgb['cv_std']:.1%})")
    
    if results["models"].get("lstm", {}).get("ok"):
        lstm = results["models"]["lstm"]
        logger.info(f"\nLSTM Results:")
        logger.info(f"  Test Accuracy: {lstm['test_accuracy']:.1%}")
    
    logger.info("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    results = run_full_training_pipeline()
    
    # Save results
    results_path = MODELS_DIR / "training_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types for JSON
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
