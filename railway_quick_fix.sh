#!/bin/bash

echo "🚀 Railway Quick Fix Script"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found"
    echo ""
    echo "Install it with:"
    echo "  npm install -g @railway/cli"
    echo ""
    echo "Or use the web dashboard:"
    echo "  https://railway.app/project/ghost-protocol"
    echo ""
    exit 1
fi

echo "✅ Railway CLI found"
echo ""

# Check if logged in
echo "🔐 Checking Railway login status..."
if ! railway whoami &> /dev/null; then
    echo "❌ Not logged in to Railway"
    echo ""
    echo "Login with:"
    echo "  railway login"
    echo ""
    exit 1
fi

echo "✅ Logged in to Railway"
echo ""

# Check if project is linked
echo "🔗 Checking project link..."
if ! railway status &> /dev/null; then
    echo "❌ Project not linked"
    echo ""
    echo "Link your project with:"
    echo "  railway link"
    echo ""
    exit 1
fi

echo "✅ Project linked"
echo ""

# Show recent logs
echo "📋 Recent Deployment Logs:"
echo "───────────────────────────────────────────────────────────────"
railway logs --lines 50

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🔍 What to look for in logs:"
echo ""
echo "  1. ❌ KeyError: Missing environment variables"
echo "  2. ❌ ModuleNotFoundError: Missing Python packages"
echo "  3. ❌ Port binding errors"
echo "  4. ❌ Database connection errors"
echo "  5. ❌ Timeout during startup"
echo ""
echo "📚 Next steps:"
echo "  1. Add missing environment variables in Railway dashboard"
echo "  2. Check RAILWAY_FIX_GUIDE.md for complete variable list"
echo "  3. Redeploy after adding variables"
echo ""
