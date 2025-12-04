#!/bin/bash
# Sequential prediction trigger - populates Cockpit panels
# Runs ONE prediction at a time to avoid overload

BASE_URL="https://ghost-protocol-production.up.railway.app"
SYMBOLS=(BTC ETH SOL BNB AAPL MSFT GOOGL TSLA NVDA META AMZN ORCL CRM ADBE XRP ADA DOGE AVAX DOT MATIC)

echo "================================================"
echo "COCKPIT POPULATION SCRIPT"
echo "================================================"
echo "Triggering predictions for ${#SYMBOLS[@]} symbols..."
echo "This will take 5-10 minutes (30s per symbol + 5s delay)"
echo ""

SUCCESS=0
FAILED=0
TIMEOUT=0

for symbol in "${SYMBOLS[@]}"; do
    echo -n "[$((SUCCESS+FAILED+TIMEOUT+1))/${#SYMBOLS[@]}] $symbol ... "
    
    # Trigger prediction with 60s timeout
    RESPONSE=$(curl -sS --max-time 60 "$BASE_URL/api/predictions/run?symbol=$symbol" 2>&1)
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        # Check if response contains "ok":true
        if echo "$RESPONSE" | grep -q '"ok".*true'; then
            echo "✅ SUCCESS"
            ((SUCCESS++))
        else
            echo "❌ FAILED (bad response)"
            ((FAILED++))
        fi
    elif [ $EXIT_CODE -eq 28 ]; then
        echo "⏱️  TIMEOUT (still processing in background)"
        ((TIMEOUT++))
    else
        echo "❌ FAILED (curl error $EXIT_CODE)"
        ((FAILED++))
    fi
    
    # Rate limit: 5s between requests
    sleep 5
done

echo ""
echo "================================================"
echo "SUMMARY"
echo "================================================"
echo "Success:  $SUCCESS"
echo "Timeout:  $TIMEOUT (may complete in background)"
echo "Failed:   $FAILED"
echo ""
echo "Checking Cockpit data availability..."
sleep 3

# Test hunter feed
echo -n "Hunter Feed: "
FEED_COUNT=$(curl -sS --max-time 5 "$BASE_URL/api/v3/hunter/feed?limit=1" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null)
if [ -n "$FEED_COUNT" ] && [ "$FEED_COUNT" -gt 0 ]; then
    echo "✅ $FEED_COUNT predictions available"
else
    echo "⚠️  Still empty (predictions may still be processing)"
fi

echo ""
echo "Done! Check cockpit at: $BASE_URL/cockpit"
echo ""
