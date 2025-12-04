#!/bin/bash
BASE_URL="https://ghost-protocol-production.up.railway.app"

echo "Waiting for Railway deployment..."
for i in {1..30}; do
    echo -n "  Attempt $i/30: "
    STATUS=$(curl -sS --max-time 3 "$BASE_URL/health" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','fail'))" 2>/dev/null)
    if [ "$STATUS" = "ok" ]; then
        echo "✅ Server is UP!"
        break
    else
        echo "⏳ Waiting..."
        sleep 5
    fi
done

echo ""
echo "Testing prediction endpoint..."
PRED_TEST=$(curl -sS --max-time 60 "$BASE_URL/api/predictions/run?symbol=BTC" 2>&1)
if echo "$PRED_TEST" | grep -q '"ok".*true'; then
    echo "✅ Predictions working!"
    echo ""
    echo "Triggering predictions for top symbols..."
    
    for symbol in BTC ETH SOL AAPL MSFT GOOGL TSLA NVDA META; do
        echo -n "  $symbol ... "
        RESP=$(curl -sS --max-time 60 "$BASE_URL/api/predictions/run?symbol=$symbol" 2>&1)
        if echo "$RESP" | grep -q '"ok".*true'; then
            echo "✅"
        else
            echo "❌"
        fi
        sleep 3
    done
    
    echo ""
    echo "Checking Cockpit data..."
    sleep 2
    FEED_COUNT=$(curl -sS --max-time 5 "$BASE_URL/api/v3/hunter/feed?limit=1" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null)
    echo "Hunter Feed: $FEED_COUNT predictions available"
    
    if [ "$FEED_COUNT" -gt 0 ]; then
        echo ""
        echo "🎉 SUCCESS! Cockpit should now be populated"
        echo "Visit: $BASE_URL/cockpit"
    fi
else
    echo "❌ Prediction endpoint still broken. Check logs."
fi
