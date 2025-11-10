#!/bin/bash
# Ghost 24/7 Deployment Script - Railway.app
# Usage: ./deploy_railway.sh

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║       Ghost Trading System - Railway Deployment       ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    npm i -g @railway/cli
fi

# Login to Railway
echo "🔐 Logging in to Railway..."
railway login

# Initialize project
echo "🚀 Initializing Railway project..."
railway init

# Set environment variables
echo "⚙️  Setting environment variables..."
read -p "Enter GHOST_API_TOKEN (or press Enter for default): " TOKEN
TOKEN=${TOKEN:-supersecret123jamaica713}

read -p "Enter POLYGON_API_KEY: " POLYGON_KEY
read -p "Enter ALPHAVANTAGE_API_KEY: " ALPHA_KEY

# Optional: Telegram
read -p "Enter TELEGRAM_BOT_TOKEN (optional, press Enter to skip): " TELEGRAM_TOKEN
read -p "Enter TELEGRAM_CHAT_ID (optional, press Enter to skip): " TELEGRAM_CHAT

railway variables set GHOST_API_TOKEN="$TOKEN"
railway variables set POLYGON_API_KEY="$POLYGON_KEY"
railway variables set ALPHAVANTAGE_API_KEY="$ALPHA_KEY"

if [ -n "$TELEGRAM_TOKEN" ]; then
    railway variables set TELEGRAM_BOT_TOKEN="$TELEGRAM_TOKEN"
fi

if [ -n "$TELEGRAM_CHAT" ]; then
    railway variables set TELEGRAM_CHAT_ID="$TELEGRAM_CHAT"
fi

# Set Ghost config
railway variables set GHOST_FOCUS_TICKER=WOLF
railway variables set WOLF_SQLITE_PATH=/data/wolf.db
railway variables set PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom

echo "📤 Deploying Ghost to Railway..."
railway up

echo
echo "✅ Deployment complete!"
echo
echo "🌐 Getting your public URL..."
DOMAIN=$(railway domain 2>/dev/null || echo "Run 'railway domain' to get URL")
echo "   Your Ghost instance: $DOMAIN"
echo
echo "📊 View logs: railway logs"
echo "🔧 Manage: railway open"
echo
echo "🧪 Test your deployment:"
echo "   curl https://$DOMAIN/health | jq ."
echo "   curl https://$DOMAIN/health/detailed | jq ."
echo
echo "╔════════════════════════════════════════════════════════╗"
echo "║        Ghost is now running 24/7 on Railway! 🎉       ║"
echo "╚════════════════════════════════════════════════════════╝"
