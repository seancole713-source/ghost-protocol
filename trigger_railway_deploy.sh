#!/bin/bash

echo "🚀 Railway Deployment Trigger Script"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check Railway webhook or trigger options
echo "📋 OPTIONS TO TRIGGER RAILWAY DEPLOYMENT:"
echo ""
echo "Option 1: Use Railway Dashboard (RECOMMENDED)"
echo "────────────────────────────────────────────────────────────────"
echo "  1. Go to: https://railway.app"
echo "  2. Select your 'ghost-protocol' project"
echo "  3. Click on the service"
echo "  4. Click the 3-dot menu (⋮) → 'Redeploy'"
echo "  5. Confirm redeploy"
echo ""
echo "Option 2: Empty Commit (if auto-deploy is enabled)"
echo "────────────────────────────────────────────────────────────────"
echo "  git commit --allow-empty -m 'chore: trigger Railway redeploy'"
echo "  git push origin main"
echo ""
echo "Option 3: Railway CLI"
echo "────────────────────────────────────────────────────────────────"
echo "  railway up"
echo ""

# Check if Railway CLI is available
if command -v railway &> /dev/null; then
    echo "✅ Railway CLI is installed"
    echo ""
    read -p "Do you want to trigger deployment with Railway CLI? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Triggering Railway deployment..."
        railway up
    fi
else
    echo "⚠️  Railway CLI not installed"
    echo ""
    echo "Install with: npm install -g @railway/cli"
    echo ""
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "📊 CHECKING CURRENT DEPLOYMENT STATUS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check health endpoint
echo "Testing: https://ghost-sniper-bot-seancole713-production.up.railway.app/ui/health"
echo ""

response=$(curl -s -w "\nHTTP_CODE:%{http_code}" https://ghost-sniper-bot-seancole713-production.up.railway.app/ui/health)
http_code=$(echo "$response" | grep "HTTP_CODE" | cut -d: -f2)
body=$(echo "$response" | grep -v "HTTP_CODE")

if [ "$http_code" = "200" ]; then
    echo "✅ Deployment is HEALTHY"
    echo "$body" | python3 -m json.tool 2>/dev/null || echo "$body"
elif [ "$http_code" = "502" ]; then
    echo "❌ Deployment is DOWN (502 Bad Gateway)"
    echo "   App is not responding - needs redeploy"
else
    echo "⚠️  HTTP Status: $http_code"
    echo "$body"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "🔍 TROUBLESHOOTING:"
echo ""
echo "If Railway didn't auto-deploy:"
echo "  • Check GitHub Actions - may have failed"
echo "  • Check Railway settings - auto-deploy may be disabled"
echo "  • Use Railway Dashboard to manually redeploy"
echo ""
echo "Latest commit:"
git log --oneline -1
echo ""
