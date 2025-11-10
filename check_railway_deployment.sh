#!/bin/bash
# Quick script to check Railway deployment status
# Run this after Railway finishes deploying

echo "🔍 RAILWAY DEPLOYMENT CHECKER"
echo "=============================="
echo ""

# Check if logged in
echo "1️⃣ Checking Railway authentication..."
if railway whoami &>/dev/null; then
    echo "   ✅ Logged in as: $(railway whoami)"
else
    echo "   ❌ Not logged in. Run: railway login"
    exit 1
fi

echo ""
echo "2️⃣ Getting deployment status..."
railway status

echo ""
echo "3️⃣ Getting your app URL..."
APP_URL=$(railway domain 2>&1)
if [[ $APP_URL == *"up.railway.app"* ]]; then
    echo "   🌐 URL: $APP_URL"
    echo ""
    echo "4️⃣ Testing health endpoint..."
    
    # Test basic health
    HEALTH_RESPONSE=$(curl -s "https://$APP_URL/health" 2>&1)
    if [[ $HEALTH_RESPONSE == *'"ok":true'* ]]; then
        echo "   ✅ Ghost is ALIVE! Health check passed"
        echo "   Response: $HEALTH_RESPONSE"
    else
        echo "   ⚠️  Health check failed or still deploying..."
        echo "   Response: $HEALTH_RESPONSE"
        echo ""
        echo "   💡 Check logs: railway logs --tail 50"
        exit 1
    fi
    
    echo ""
    echo "5️⃣ Testing detailed health endpoint..."
    curl -s "https://$APP_URL/health/detailed" | head -30
    
    echo ""
    echo ""
    echo "✅ DEPLOYMENT SUCCESSFUL!"
    echo "========================"
    echo "🌐 Ghost URL: https://$APP_URL"
    echo "🏥 Health: https://$APP_URL/health"
    echo "📊 Cockpit: https://$APP_URL/cockpit"
    echo ""
    echo "Next: Restore position data with:"
    echo "curl -X POST 'https://$APP_URL/api/position' \\"
    echo "  -H 'Authorization: Bearer e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0' \\"
    echo "  -H 'Content-Type: application/json' \\"
    echo "  -d '{\"qty\": 8.41959051, \"avg_cost\": 359.28}'"
else
    echo "   ❌ Could not get domain. Response: $APP_URL"
    echo "   Railway might still be deploying..."
    echo ""
    echo "   💡 Check logs: railway logs"
fi
