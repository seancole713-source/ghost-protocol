#!/bin/bash
BASE_URL="https://ghost-protocol-production.up.railway.app"

echo "Waiting 90s for Railway deployment..."
sleep 90

echo ""
echo "Testing health..."
for i in {1..20}; do
    STATUS=$(python3 -c "import requests; r=requests.get('$BASE_URL/health', timeout=3); print(r.json()['status'])" 2>/dev/null)
    if [ "$STATUS" = "ok" ]; then
        echo "✅ Server ready!"
        break
    fi
    echo "  Attempt $i/20... waiting"
    sleep 5
done

echo ""
echo "Testing prediction endpoint..."
TEST=$(python3 -c "import requests, json; r=requests.get('$BASE_URL/api/predictions/run?symbol=BTC', timeout=60); res=r.json(); print('SUCCESS' if res.get('ok') and res.get('result',{}).get('ok') else res.get('result',{}).get('error','FAILED'))" 2>&1)

if [ "$TEST" = "SUCCESS" ]; then
    echo "✅ Predictions working!"
    echo ""
    echo "Populating Cockpit with top 15 symbols..."
    
    for symbol in BTC ETH SOL BNB XRP AAPL MSFT GOOGL TSLA NVDA META AMZN ORCL CRM ADBE; do
        echo -n "  $symbol ... "
        RESULT=$(python3 -c "import requests; r=requests.get('$BASE_URL/api/predictions/run?symbol=$symbol', timeout=60); res=r.json(); print('✅' if res.get('ok') and res.get('result',{}).get('ok') else '❌')" 2>&1)
        echo "$RESULT"
        sleep 2
    done
    
    echo ""
    echo "Checking Cockpit..."
    sleep 3
    FEED=$(python3 -c "import requests; r=requests.get('$BASE_URL/api/v3/hunter/feed?limit=1', timeout=5); print(r.json().get('count',0))" 2>/dev/null)
    echo "Hunter Feed: $FEED predictions"
    
    if [ "$FEED" -gt 0 ]; then
        echo ""
        echo "🎉 SUCCESS! Cockpit is now populated!"
        echo "Visit: $BASE_URL/cockpit"
    else
        echo "⚠️ Feed still empty - may need more time"
    fi
else
    echo "❌ Prediction test failed: $TEST"
    echo "Check Railway logs for errors"
fi
