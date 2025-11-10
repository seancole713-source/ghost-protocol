#!/bin/bash
# GHOST OmniBrain v10.3 - Startup Script
# Starts GHOST with full crypto + stocks + observability

set -e

echo "🚀 Starting GHOST OmniBrain v10.3..."

# Core Configuration
export SIM_MODE=0
export LOG_LEVEL=INFO
export LOG_JSON=1
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
export ADMIN_IP_ALLOWLIST="127.0.0.1"

# Crypto Module
export CRYPTO_ENABLED=1
export CRYPTO_SYMBOLS="BTC,ETH,SOL,BNB,DOGE,SHIB,PEPE,AVAX,DOT,MATIC,LINK,UNI,AAVE"
export CRYPTO_LOOKBACK_H=96
export CRYPTO_FORECAST_H=48
export CRYPTO_PRICE_SOURCE=coingecko
export CRYPTO_QUORUM="coingecko,binance,coinbase"

# Features
export NEWS_SENTIMENT_ON=1
export FUSION_AI_ON=1
export MACRO_BRAIN_ON=1

# AI Configuration
export AI_ON=1
export AI_PROVIDER=openai
export AI_MODEL=gpt-4o-mini

# Optional: Set these if you have keys
# export OPENAI_API_KEY="your-key-here"
# export ALPHA_VANTAGE_API_KEY="your-key-here"
# export POLYGON_API_KEY="your-key-here"
# export TELEGRAM_BOT_TOKEN="your-token-here"
# export TELEGRAM_CHAT_ID="your-chat-id-here"
# export GHOST_API_TOKEN="your-secret-token-here"

# Database
export WOLF_SQLITE_PATH="data/wolf.db"

# Create required directories
mkdir -p data
mkdir -p /tmp/ghost_prom

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found. Using system Python."
fi

# Check Python version
python --version

# Start server
PORT=${PORT:-5000}
echo "🌐 Starting server on port $PORT..."
echo "📊 Metrics: http://localhost:$PORT/metrics"
echo "🎛️  Cockpit: http://localhost:$PORT/cockpit"
echo "🔍 Health: http://localhost:$PORT/health"
echo ""

# Run with uvicorn
python -m uvicorn wolf_app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --reload \
    --log-level info

# Note: For production, remove --reload and use:
# python -m uvicorn wolf_app:app --host 0.0.0.0 --port $PORT --workers 2
