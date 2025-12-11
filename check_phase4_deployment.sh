#!/bin/bash
# Phase 4 Deployment Status Checker

echo "🧠 Checking Phase 4 Self-Improvement Engine Deployment..."
echo "================================================================"
echo ""

# Railway deployment URL
RAILWAY_URL="https://ghost-production-1d4b.up.railway.app"

# Check if app is responding
echo "1️⃣  Checking if Ghost is online..."
if curl -s -f "${RAILWAY_URL}/" > /dev/null 2>&1; then
    echo "   ✅ Ghost is online"
else
    echo "   ⏳ Ghost is deploying (Railway build in progress)"
    echo "   💡 Wait 2-3 minutes for Railway to rebuild and restart"
    echo ""
    echo "   Check Railway logs:"
    echo "   https://railway.app/project/ghost-protocol/deployments"
    exit 0
fi

echo ""
echo "2️⃣  Checking self-improvement engine status..."
RESPONSE=$(curl -s "${RAILWAY_URL}/api/v3/self-improvement/status")

if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "   ✅ Self-improvement engine is active"
    
    # Extract key metrics
    ITERATIONS=$(echo "$RESPONSE" | grep -o '"iterations":[0-9]*' | cut -d':' -f2)
    THRESHOLD=$(echo "$RESPONSE" | grep -o '"current_threshold":[0-9.]*' | cut -d':' -f2)
    
    echo ""
    echo "📊 Engine Metrics:"
    echo "   - Iterations completed: ${ITERATIONS:-0}"
    echo "   - Current threshold: ${THRESHOLD:-N/A}%"
    echo ""
    echo "   View full status: ${RAILWAY_URL}/api/v3/self-improvement/status"
else
    echo "   ⚠️  Engine not responding yet"
    echo "   Response: ${RESPONSE:0:200}"
fi

echo ""
echo "3️⃣  Checking accuracy evaluator..."
ACCURACY_STATUS=$(curl -s "${RAILWAY_URL}/api/v3/accuracy/summary" | grep -o '"ok":[a-z]*' | cut -d':' -f2)

if [ "$ACCURACY_STATUS" = "true" ]; then
    echo "   ✅ Accuracy tracker operational"
else
    echo "   ⏳ Accuracy tracker initializing"
fi

echo ""
echo "4️⃣  Checking movers detection (2% threshold)..."
MOVERS=$(curl -s "${RAILWAY_URL}/api/v3/market/movers?limit=5")

if echo "$MOVERS" | grep -q '"ok":true'; then
    COUNT=$(echo "$MOVERS" | grep -o '"count":[0-9]*' | cut -d':' -f2)
    echo "   ✅ Movers scanner active (${COUNT:-0} movers detected)"
else
    echo "   ⏳ Movers scanner initializing"
fi

echo ""
echo "================================================================"
echo "🎉 Phase 4 deployment check complete!"
echo ""
echo "📖 Full documentation: PHASE_4_SELF_IMPROVEMENT_COMPLETE.md"
echo "🔍 Monitor logs: https://railway.app/project/ghost-protocol/deployments"
echo "🚀 Live API: ${RAILWAY_URL}"
