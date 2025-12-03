#!/bin/bash
# Quick endpoint testing script for Ghost Protocol
# Usage: bash QUICK_TEST.sh

BASE_URL="https://ghost-protocol-production.up.railway.app"

echo "=========================================="
echo "Ghost Protocol Endpoint Test Suite"
echo "=========================================="
echo ""

echo "1. Testing /health..."
curl --max-time 8 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n\n" \
  -sS "${BASE_URL}/health" | python3 -m json.tool 2>/dev/null || echo "ERROR: Invalid JSON or timeout"

echo ""
echo "2. Testing /api/v3/watchlist/enriched..."
curl --max-time 8 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n\n" \
  -sS "${BASE_URL}/api/v3/watchlist/enriched" | python3 -m json.tool 2>/dev/null | head -40 || echo "ERROR: Invalid JSON or timeout"

echo ""
echo "3. Testing /api/v3/predictions/latest..."
curl --max-time 8 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n\n" \
  -sS "${BASE_URL}/api/v3/predictions/latest?limit=3" | python3 -m json.tool 2>/dev/null | head -40 || echo "ERROR: Invalid JSON or timeout"

echo ""
echo "4. Testing /api/v3/goals/snapshot..."
curl --max-time 8 -w "\nHTTP:%{http_code} TIME:%{time_total}s\n\n" \
  -sS "${BASE_URL}/api/v3/goals/snapshot" | python3 -m json.tool 2>/dev/null | head -40 || echo "ERROR: Invalid JSON or timeout"

echo "=========================================="
echo "Test suite complete!"
echo "Endpoints should return HTTP 200 and valid JSON."
echo "=========================================="
