#!/bin/bash
# Ghost Trading - Railway Deployment (Secure)
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     GHOST → RAILWAY DEPLOYMENT (SECURE)                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Load credentials from environment or prompt user
if [ -f .env ]; then
    echo "📋 Loading credentials from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Prompt for missing credentials
if [ -z "$GHOST_API_TOKEN" ]; then
    read -sp "Enter GHOST_API_TOKEN: " GHOST_API_TOKEN
    echo
fi

if [ -z "$POLYGON_API_KEY" ]; then
    read -sp "Enter POLYGON_API_KEY: " POLYGON_API_KEY
    echo
fi

if [ -z "$ALPHAVANTAGE_API_KEY" ]; then
    read -sp "Enter ALPHAVANTAGE_API_KEY: " ALPHAVANTAGE_API_KEY
    echo
fi

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    read -sp "Enter TELEGRAM_BOT_TOKEN (optional, press Enter to skip): " TELEGRAM_BOT_TOKEN
    echo
fi

TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-940596997}"
GHOST_FOCUS_TICKER="${GHOST_FOCUS_TICKER:-WOLF}"

echo "📋 Configuration ready (credentials masked)"
echo

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "📦 Railway CLI not found. Installing..."
    npm install -g @railway/cli
    echo "✅ Railway CLI installed"
else
    echo "✅ Railway CLI found"
fi

# Login check
echo "🔐 Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    echo "Please login to Railway (browser will open):"
    railway login
    
    # Wait for login
    sleep 2
    if railway whoami &> /dev/null; then
        echo "✅ Logged in as: $(railway whoami)"
    else
        echo "❌ Login failed. Please try again."
        exit 1
    fi
else
    echo "✅ Already logged in as: $(railway whoami)"
fi

# Initialize project
echo
echo "🚀 Initializing Railway project..."
if [ ! -f ".railway/project.json" ]; then
    railway init --name ghost-trading
    echo "✅ Project 'ghost-trading' created"
else
    echo "✅ Project already initialized"
fi

# Set environment variables
echo
echo "⚙️  Setting environment variables..."

railway variables set GHOST_API_TOKEN="$GHOST_API_TOKEN"
railway variables set POLYGON_API_KEY="$POLYGON_API_KEY"
railway variables set ALPHAVANTAGE_API_KEY="$ALPHAVANTAGE_API_KEY"
railway variables set TELEGRAM_BOT_TOKEN="$TELEGRAM_BOT_TOKEN"
railway variables set TELEGRAM_CHAT_ID="$TELEGRAM_CHAT_ID"
railway variables set GHOST_FOCUS_TICKER="$GHOST_FOCUS_TICKER"
railway variables set WOLF_PERSIST_MODE="sqlite"
railway variables set SIM_MODE="0"

echo "✅ All environment variables set"

# Deploy
echo
echo "🚀 Deploying Ghost to Railway..."
echo "   (This takes 2-5 minutes...)"
echo

railway up

# Get deployment info
echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              🎉 DEPLOYMENT COMPLETE!                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# Get domain
DOMAIN=$(railway domain 2>/dev/null | grep -oP 'https?://[^\s]+' | head -1)

if [ -n "$DOMAIN" ]; then
    echo "🔗 Your Ghost is live at:"
    echo "   $DOMAIN"
    echo
    echo "📊 Test it now:"
    echo "   curl $DOMAIN/health"
    echo
    echo "🖥️  Railway Dashboard:"
    railway open
else
    echo "⚠️  Getting domain..."
    echo "   Run: railway domain"
fi

echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "✅ NEXT STEPS:"
echo
echo "1. Get your URL:"
echo "   railway domain"
echo
echo "2. Test health:"
echo "   curl https://your-app.railway.app/health"
echo
echo "3. Restore position:"
echo "   curl -X POST https://your-app.railway.app/api/position \\"
echo "     -H 'Authorization: Bearer $GHOST_API_TOKEN' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"qty\": 8.41959051, \"avg_cost\": 359.28}'"
echo
echo "4. View logs:"
echo "   railway logs"
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo
echo "🎊 Ghost is now running 24/7 on Railway!"
echo
