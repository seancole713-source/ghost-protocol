#!/bin/bash
# Test watchlist and alerts endpoints after deployment

BASE_URL="https://ghost-protocol-production.up.railway.app"

echo "=========================================="
echo "Testing Watchlist & Alerts Endpoints"
echo "=========================================="
echo ""

echo "1️⃣  Testing /api/v3/watchlist/enriched"
echo "----------------------------------------"
curl -sS "$BASE_URL/api/v3/watchlist/enriched" | python3 -m json.tool | head -20
echo ""

echo "2️⃣  Testing /api/v3/watchlist/user"
echo "----------------------------------------"
curl -sS "$BASE_URL/api/v3/watchlist/user" | python3 -m json.tool | head -20
echo ""

echo "3️⃣  Testing /api/recent_alerts (should be public)"
echo "----------------------------------------"
curl -sS "$BASE_URL/api/recent_alerts?limit=3" | python3 -m json.tool
echo ""

echo "=========================================="
echo "✅ All endpoint tests complete"
echo "=========================================="
