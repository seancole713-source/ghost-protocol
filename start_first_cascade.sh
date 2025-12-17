#!/bin/bash
# Ghost Protocol - First Cascade Starter 🚀
# Run this AFTER Railway deploys the cascade system

RAILWAY_URL="https://ghost-protocol-production.up.railway.app"

echo "🎯 STARTING FIRST GHOST CASCADE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Wait for deployment
echo "⏳ Waiting for Railway deployment to complete..."
echo "   Checking health endpoint..."

for i in {1..30}; do
    HEALTH=$(curl -s --max-time 5 "$RAILWAY_URL/health" 2>/dev/null)
    GIT_SHA=$(echo "$HEALTH" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('git_sha', '')[:7])" 2>/dev/null)
    
    if [ "$GIT_SHA" = "b0bf235" ]; then
        echo "   ✅ New deployment LIVE! (SHA: $GIT_SHA)"
        break
    elif [ "$GIT_SHA" = "7f51722" ]; then
        echo "   ⏳ Old deployment still active (attempt $i/30)..."
        sleep 10
    else
        echo "   ⏳ Checking... (attempt $i/30)"
        sleep 5
    fi
done

echo ""
echo "🧪 Testing Cascade Endpoints..."
echo ""

# Test 1: Stats endpoint
echo "1️⃣ Cascade Stats:"
STATS=$(curl -s "$RAILWAY_URL/api/v3/cascade/stats?days=7")
echo "$STATS" | python3 -m json.tool 2>/dev/null || echo "$STATS"
echo ""

# Test 2: List endpoint
echo "2️⃣ Active Cascades:"
LIST=$(curl -s "$RAILWAY_URL/api/v3/cascade/list?active_only=true")
echo "$LIST" | python3 -m json.tool 2>/dev/null || echo "$LIST"
echo ""

# Test 3: Start BTC cascade!
echo "3️⃣ Starting BTC Cascade... 🚀"
RESPONSE=$(curl -s -X POST "$RAILWAY_URL/api/v3/cascade/start?symbol=BTC")
echo "$RESPONSE" | python3 -m json.tool

# Extract cascade ID
CASCADE_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('cascade_id', 'N/A'))" 2>/dev/null)

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$CASCADE_ID" != "N/A" ] && [ -n "$CASCADE_ID" ]; then
    echo "🎉 CASCADE STARTED!"
    echo ""
    echo "📊 Cascade ID: $CASCADE_ID"
    echo ""
    echo "📅 Timeline:"
    echo "   T+0h  (NOW):  48h alert sent to Telegram"
    echo "   T+24h:        24h update"
    echo "   T+42h:        6h final call (highest accuracy)"
    echo "   T+48h:        Outcome evaluation"
    echo ""
    echo "🔍 Monitor Progress:"
    echo "   curl $RAILWAY_URL/api/v3/cascade/$CASCADE_ID"
    echo ""
    echo "📱 Check Telegram for:"
    echo "   🔔 48H EARLY ALERT - BTC"
    echo ""
    echo "🎯 First cascade in motion. This is legendary! 🚀"
else
    echo "⚠️  Cascade start returned unexpected response"
    echo "   Try manual start:"
    echo "   curl -X POST $RAILWAY_URL/api/v3/cascade/start?symbol=BTC"
fi

echo ""
