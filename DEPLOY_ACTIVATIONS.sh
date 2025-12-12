#!/bin/bash
# 🚀 GHOST PROTOCOL - DEPLOY ALL SYSTEM ACTIVATIONS TO RAILWAY
# ============================================================

echo "🚀 DEPLOYING GHOST PROTOCOL SYSTEM ACTIVATIONS TO RAILWAY"
echo "=========================================================="
echo ""

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Install it first:"
    echo "   npm i -g @railway/cli"
    echo "   railway login"
    exit 1
fi

echo "✅ Railway CLI found"
echo ""

# Set all activation environment variables
echo "📝 Setting environment variables on Railway..."
echo ""

railway variables set CRYPTO_ENABLED=1
echo "   ✅ CRYPTO_ENABLED=1"

railway variables set AI_ADVISOR_ENABLED=1
echo "   ✅ AI_ADVISOR_ENABLED=1"

railway variables set MULTI_TIMEFRAME_ENABLED=1
echo "   ✅ MULTI_TIMEFRAME_ENABLED=1"

railway variables set BACKTESTING_ENABLED=1
echo "   ✅ BACKTESTING_ENABLED=1"

railway variables set SOCIAL_SENTIMENT_ENABLED=1
echo "   ✅ SOCIAL_SENTIMENT_ENABLED=1"

railway variables set ECONOMIC_CALENDAR_ENABLED=1
echo "   ✅ ECONOMIC_CALENDAR_ENABLED=1"

echo ""
echo "=========================================================="
echo "✅ ENVIRONMENT VARIABLES SET"
echo "=========================================================="
echo ""

# Push code to trigger redeploy
echo "📦 Pushing code to Railway (commit 314df0a)..."
echo ""

git push railway main || git push origin main

echo ""
echo "=========================================================="
echo "🎉 DEPLOYMENT COMPLETE"
echo "=========================================================="
echo ""
echo "📋 NEXT STEPS:"
echo ""
echo "1. Wait for Railway deployment (2-3 minutes)"
echo "   https://railway.app/project/YOUR_PROJECT/deployments"
echo ""
echo "2. Verify activation:"
echo "   curl https://your-railway-url/api/v3/cockpit/status"
echo ""
echo "3. Test new endpoints:"
echo "   curl https://your-railway-url/api/crypto/forecast/BTC"
echo "   curl https://your-railway-url/api/advisor/recommendations"
echo "   curl https://your-railway-url/api/multi_timeframe/AAPL"
echo ""
echo "4. Optional: Add API keys for full functionality"
echo "   railway variables set TWITTER_BEARER_TOKEN=your_token"
echo "   railway variables set FRED_API_KEY=your_key"
echo ""
echo "🚀 Ghost Protocol now fully activated with all hidden systems!"
echo ""
