#!/bin/bash
# Production verification script - checks if fixes are working with REAL DATA

BASE_URL="https://ghost-protocol-production.up.railway.app"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         GHOST PRODUCTION DATA VERIFICATION                 ║"
echo "║    Checking if sentiment + world context fixes work        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Test 1: Check system status
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: System Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
COCKPIT=$(curl -s "$BASE_URL/api/v3/cockpit/status")
echo "$COCKPIT" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"✅ System OK: {d.get('ok', False)}\")
print(f\"📊 Predictions: {d.get('predictions_in_memory', 0)}\")
print(f\"💾 Database: {d.get('database', 'unknown')}\")
"

# Test 2: Check latest predictions
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: Recent Predictions"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Get multiple predictions to check for patterns
for symbol in BTC ETH RNDR ZEC TURBO; do
    echo ""
    echo "Checking $symbol..."
    PRED=$(curl -s "$BASE_URL/api/v3/predictions/latest?symbol=$symbol")
    echo "$PRED" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if d.get('predictions'):
        p = d['predictions'][0]
        print(f\"  ID: {p.get('prediction_id')}\")
        print(f\"  Confidence: {p.get('confidence', 0)*100:.1f}%\")
        print(f\"  Direction: {p.get('direction')}\")
    else:
        print(f\"  No predictions found\")
except:
    print(f\"  Error parsing response\")
" || echo "  ❌ Failed to fetch"
done

# Test 3: Check accuracy data (might have feature info)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Accuracy Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ACC=$(curl -s "$BASE_URL/api/v3/accuracy/summary")
echo "$ACC" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f\"30d Accuracy: {d.get('accuracy_30d', 0)*100:.1f}%\")
print(f\"Total Predictions: {d.get('total_predictions', 0)}\")
print(f\"Correct: {d.get('correct_predictions', 0)}\")
" || echo "❌ Failed to fetch accuracy"

# Test 4: Check for news/sentiment endpoints
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 4: News/Sentiment Data (if exposed)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Try to find any sentiment-related endpoints
curl -s "$BASE_URL/api/v3/sentiment/BTC" > /tmp/sentiment_test.json 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Sentiment endpoint exists"
    cat /tmp/sentiment_test.json | python3 -m json.tool | head -20
else
    echo "⚠️  No public sentiment endpoint (data may be internal only)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  LIMITATION: Production API doesn't expose feature-level data"
echo ""
echo "To verify sentiment_engine and world_context fixes work, need:"
echo "  1. Railway logs showing sentiment engine activity"
echo "  2. Direct database query of prediction features"
echo "  3. Railway run command to test modules directly"
echo ""
echo "Cannot verify fixes are working from external API alone."
echo ""
