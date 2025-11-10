#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║         Ghost Trading - Railway Deployment        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Login check
echo "🔐 Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    echo "Please login to Railway:"
    railway login
fi

# Initialize project if needed
if [ ! -f ".railway/project.json" ]; then
    echo "🚀 Initializing Railway project..."
    railway init
fi

# Set environment variables
echo "⚙️  Setting environment variables..."
echo "Please enter your secrets:"

read -p "GHOST_API_TOKEN: " GHOST_API_TOKEN
read -p "POLYGON_API_KEY: " POLYGON_API_KEY
read -p "ALPHAVANTAGE_API_KEY: " ALPHAVANTAGE_API_KEY
read -p "TELEGRAM_BOT_TOKEN (optional): " TELEGRAM_BOT_TOKEN
read -p "TELEGRAM_CHAT_ID (optional): " TELEGRAM_CHAT_ID

# Set variables
railway variables set GHOST_API_TOKEN="$GHOST_API_TOKEN"
railway variables set POLYGON_API_KEY="$POLYGON_API_KEY"
railway variables set ALPHAVANTAGE_API_KEY="$ALPHAVANTAGE_API_KEY"
railway variables set GHOST_FOCUS_TICKER="WOLF"
railway variables set WOLF_PERSIST_MODE="sqlite"

if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    railway variables set TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN"
fi

if [ -n "$TELEGRAM_CHAT_ID" ]; then
    railway variables set TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID"
fi

# Deploy
echo "🚀 Deploying to Railway..."
railway up

# Get deployment URL
echo
echo "✅ Deployment complete!"
echo
railway status
echo
echo "🔗 Your Ghost instance is now live at:"
railway domain

echo
echo "📊 Test your deployment:"
echo "  curl \$(railway domain)/health"
echo
echo "🎉 Ghost is now running 24/7 on Railway!"
