#!/bin/bash
BASE_URL="https://ghost-protocol-production.up.railway.app"
echo "Triggering predictions for top symbols..."

for symbol in BTC ETH SOL BNB AAPL MSFT GOOGL TSLA NVDA META; do
    echo "  - $symbol"
    curl -sS "$BASE_URL/api/predictions/run?symbol=$symbol" > /dev/null &
done

wait
echo ""
echo "All predictions triggered. Waiting 5s for processing..."
sleep 5

echo ""
echo "Checking hunter feed..."
curl -sS "$BASE_URL/api/v3/hunter/feed?limit=5" | python3 -m json.tool
