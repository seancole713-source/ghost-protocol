#!/bin/bash
# Test Ghost prediction wiring after deployment

BASE_URL="https://ghost-sniper-bot-seancole713-production.up.railway.app"

echo "🔍 Testing Ghost Prediction Wiring Fix"
echo "======================================"
echo ""

# Test 1: Health check
echo "1️⃣ Testing /api/health..."
curl -s "$BASE_URL/api/health" | python3 -m json.tool 2>/dev/null || curl -s "$BASE_URL/api/health"
echo -e "\n"

# Test 2: Debug endpoint (should show empty store initially)
echo "2️⃣ Testing /api/debug/predictions (before prediction)..."
curl -s "$BASE_URL/api/debug/predictions" | python3 -m json.tool 2>/dev/null || curl -s "$BASE_URL/api/debug/predictions"
echo -e "\n"

# Test 3: Create a prediction
echo "3️⃣ Creating prediction for WOLF..."
curl -s -X POST "$BASE_URL/api/predict/run" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"WOLF"}' | python3 -m json.tool 2>/dev/null || curl -s -X POST "$BASE_URL/api/predict/run" -H "Content-Type: application/json" -d '{"symbol":"WOLF"}'
echo -e "\n"

# Test 4: Debug endpoint (should now show WOLF)
echo "4️⃣ Testing /api/debug/predictions (after prediction)..."
curl -s "$BASE_URL/api/debug/predictions" | python3 -m json.tool 2>/dev/null || curl -s "$BASE_URL/api/debug/predictions"
echo -e "\n"

# Test 5: Cockpit snapshot
echo "5️⃣ Testing /api/cockpit/snapshot (predictions field)..."
curl -s "$BASE_URL/api/cockpit/snapshot" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({'timestamp': d.get('timestamp'), 'predictions': d.get('predictions')}, indent=2))" 2>/dev/null || echo "Failed to parse"
echo -e "\n"

# Test 6: Cockpit
echo "6️⃣ Testing /api/cockpit (predictions field)..."
curl -s "$BASE_URL/api/cockpit" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({'predictions': d.get('predictions')}, indent=2))" 2>/dev/null || echo "Failed to parse"
echo -e "\n"

echo "======================================"
echo "✅ Tests complete!"
echo ""
echo "Expected results:"
echo "  - Health: ok=true"
echo "  - Debug (before): empty store"
echo "  - Prediction: ok=true, prediction_id=N"
echo "  - Debug (after): store contains WOLF"
echo "  - Snapshot: predictions.stocks=[...WOLF...]"
echo "  - Cockpit: predictions={WOLF: {...}}"
