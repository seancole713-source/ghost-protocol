#!/bin/bash

# TO THE MOON Endpoint Verification Script
# Tests all 7 advanced system endpoints on Railway production

set -e

BASE_URL="${RAILWAY_URL:-https://ghost-protocol-production.up.railway.app}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 TO THE MOON ENDPOINT VERIFICATION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Testing: $BASE_URL"
echo "Date: $(date)"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
RESULTS=()

# Function to test an endpoint
test_endpoint() {
    local name="$1"
    local path="$2"
    local expected_keys="$3"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: $name"
    echo "Path: $path"
    
    # Make request and capture response
    HTTP_CODE=$(curl -s -o /tmp/response.json -w "%{http_code}" "$BASE_URL$path")
    RESPONSE=$(cat /tmp/response.json)
    
    # Check HTTP status
    if [ "$HTTP_CODE" != "200" ]; then
        echo "❌ FAIL: HTTP $HTTP_CODE"
        echo "Response: $RESPONSE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("❌ $name|$HTTP_CODE|FAIL|HTTP error")
        echo ""
        return 1
    fi
    
    # Check JSON parse
    if ! echo "$RESPONSE" | jq empty 2>/dev/null; then
        echo "❌ FAIL: Invalid JSON"
        echo "Response: $RESPONSE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("❌ $name|200|FAIL|Invalid JSON")
        echo ""
        return 1
    fi
    
    # Check for error fields
    if echo "$RESPONSE" | jq -e '.error' >/dev/null 2>&1; then
        ERROR_MSG=$(echo "$RESPONSE" | jq -r '.error')
        echo "❌ FAIL: Error in response: $ERROR_MSG"
        echo "Response: $RESPONSE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("❌ $name|200|FAIL|$ERROR_MSG")
        echo ""
        return 1
    fi
    
    # Check expected keys
    MISSING_KEYS=""
    for key in $expected_keys; do
        if ! echo "$RESPONSE" | jq -e ".$key" >/dev/null 2>&1; then
            MISSING_KEYS="$MISSING_KEYS $key"
        fi
    done
    
    if [ -n "$MISSING_KEYS" ]; then
        echo "⚠️  PARTIAL: Missing keys:$MISSING_KEYS"
        echo "Response: $RESPONSE"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        RESULTS+=("⚠️  $name|200|PARTIAL|Missing keys:$MISSING_KEYS")
        echo ""
        return 1
    fi
    
    # Success
    echo "✅ PASS: HTTP 200, Valid JSON, All keys present"
    echo "Sample: $(echo "$RESPONSE" | jq -c '. | with_entries(select(.key | IN("enabled", "status", "uptime_seconds", "result", "shift_detected", "hedging", "response")))' 2>/dev/null || echo "$RESPONSE" | head -c 100)"
    PASS_COUNT=$((PASS_COUNT + 1))
    RESULTS+=("✅ $name|200|PASS|OK")
    echo ""
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RUNNING TESTS (7 endpoints)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test each endpoint
test_endpoint "System Status" \
    "/api/system_status" \
    "system_status uptime_seconds"

test_endpoint "Walk-Forward Analysis" \
    "/api/walk_forward_analysis/AAPL" \
    "enabled symbol"

test_endpoint "Monte Carlo Simulation" \
    "/api/monte_carlo/TSLA" \
    "enabled symbol"

test_endpoint "Momentum Shift Detection" \
    "/api/momentum_shift/NVDA" \
    "enabled symbol"

test_endpoint "Research Blueprint" \
    "/api/research/MSFT" \
    "enabled symbol"

test_endpoint "Hedging Recommendations" \
    "/api/hedging/recommendations" \
    "enabled hedging"

test_endpoint "AgentKit Chat" \
    "/api/agentkit/chat?message=What%20is%20Bitcoin%20price" \
    "enabled response"

# Print summary table
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TEST RESULTS SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
printf "%-35s %-10s %-10s %-30s\n" "ENDPOINT" "HTTP" "RESULT" "MESSAGE"
echo "───────────────────────────────────────────────────────────────────────────────"

for result in "${RESULTS[@]}"; do
    IFS='|' read -r status name http result_type message <<< "$result"
    printf "%-35s %-10s %-10s %-30s\n" "$name" "$http" "$result_type" "$message"
done

echo "───────────────────────────────────────────────────────────────────────────────"
echo ""
echo "✅ PASSED: $PASS_COUNT/7"
echo "❌ FAILED: $FAIL_COUNT/7"
echo ""

if [ $PASS_COUNT -eq 7 ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎉 ALL TESTS PASSED! TO THE MOON! 🚀🌙"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 0
else
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚠️  SOME TESTS FAILED - CHECK LOGS ABOVE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    exit 1
fi
