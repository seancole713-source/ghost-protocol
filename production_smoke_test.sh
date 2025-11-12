#!/usr/bin/env bash
#
# Ghost Production Validation - Comprehensive Smoke Tests
# Tests critical endpoints and validates responses
#

set -euo pipefail

GHOST_BASE_URL="${GHOST_BASE_URL:-https://ghost-sniper-bot-seancole713-production.up.railway.app}"
RESULTS_FILE="/tmp/ghost_smoke_results.json"
PASSED=0
FAILED=0

echo "================================================================================"
echo "GHOST PRODUCTION SMOKE TESTS"
echo "================================================================================"
echo "Target: $GHOST_BASE_URL"
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# Helper function to test endpoint
test_endpoint() {
    local name="$1"
    local path="$2"
    local expected_code="${3:-200}"
    local check_json="${4:-true}"
    
    echo -n "Testing $name ... "
    
    response=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer $GHOST_API_TOKEN" "$GHOST_BASE_URL$path" 2>&1) || {
        echo "❌ FAILED (curl error)"
        FAILED=$((FAILED + 1))
        echo "{\"endpoint\": \"$path\", \"status\": \"error\", \"error\": \"curl_failed\"}" >> "$RESULTS_FILE"
        return 1
    }
    
    body=$(echo "$response" | head -n -1)
    http_code=$(echo "$response" | tail -n 1)
    first_200_bytes=$(echo "$body" | head -c 200)
    
    # Check HTTP code
    if [ "$http_code" != "$expected_code" ]; then
        echo "❌ FAILED (HTTP $http_code, expected $expected_code)"
        echo "   Body: $first_200_bytes"
        FAILED=$((FAILED + 1))
        echo "{\"endpoint\": \"$path\", \"status\": \"failed\", \"http_code\": $http_code, \"expected\": $expected_code, \"body_preview\": \"$first_200_bytes\"}" >> "$RESULTS_FILE"
        return 1
    fi
    
    # Check if response is JSON (if required)
    if [ "$check_json" = "true" ]; then
        if ! echo "$body" | jq empty 2>/dev/null; then
            echo "❌ FAILED (invalid JSON)"
            echo "   Body: $first_200_bytes"
            FAILED=$((FAILED + 1))
            echo "{\"endpoint\": \"$path\", \"status\": \"failed\", \"error\": \"invalid_json\", \"body_preview\": \"$first_200_bytes\"}" >> "$RESULTS_FILE"
            return 1
        fi
    fi
    
    echo "✅ PASSED (HTTP $http_code)"
    PASSED=$((PASSED + 1))
    echo "{\"endpoint\": \"$path\", \"status\": \"passed\", \"http_code\": $http_code, \"body_preview\": \"$first_200_bytes\"}" >> "$RESULTS_FILE"
    return 0
}

# Initialize results file
echo "[" > "$RESULTS_FILE"

# Test 1: /api/status (should return 200 with env flags)
test_endpoint "/api/status" "/api/status" 200 true

# Test 2: /api/health (new endpoint)
test_endpoint "/api/health" "/api/health" 200 true

# Test 3: /api/tick (counter endpoint)
test_endpoint "/api/tick" "/api/tick" 200 true

# Test 4: /api/regime/current (market regime)
test_endpoint "/api/regime/current" "/api/regime/current" 200 true

# Test 5: /api/scan/movers (crypto/stock movers)
test_endpoint "/api/scan/movers" "/api/scan/movers" 200 true

# Test 6: /api/scan/health (health metrics)
test_endpoint "/api/scan/health" "/api/scan/health" 200 true

# Test 7: /api/_crash (canary - should return 500 JSON)
test_endpoint "/api/_crash" "/api/_crash" 500 true

# Test 8: Check x-ghost-mw header on /api/status
echo -n "Testing x-ghost-mw header ... "
headers=$(curl -s -I -H "Authorization: Bearer $GHOST_API_TOKEN" "$GHOST_BASE_URL/api/status" 2>&1)
if echo "$headers" | grep -qi "x-ghost-mw: on"; then
    echo "✅ PASSED (header present)"
    PASSED=$((PASSED + 1))
    echo "{\"endpoint\": \"/api/status (headers)\", \"status\": \"passed\", \"header\": \"x-ghost-mw: on\"}" >> "$RESULTS_FILE"
else
    echo "❌ FAILED (header missing)"
    FAILED=$((FAILED + 1))
    echo "{\"endpoint\": \"/api/status (headers)\", \"status\": \"failed\", \"error\": \"missing_x-ghost-mw_header\"}" >> "$RESULTS_FILE"
fi

# Test 9: Check env flags in /api/status
echo -n "Testing env flags in /api/status ... "
status_resp=$(curl -s -H "Authorization: Bearer $GHOST_API_TOKEN" "$GHOST_BASE_URL/api/status" 2>&1)
if echo "$status_resp" | jq -e '.env' >/dev/null 2>&1; then
    echo "✅ PASSED (env object present)"
    PASSED=$((PASSED + 1))
    env_flags=$(echo "$status_resp" | jq -c '.env')
    echo "{\"endpoint\": \"/api/status (env)\", \"status\": \"passed\", \"env_flags\": $env_flags}" >> "$RESULTS_FILE"
else
    echo "❌ FAILED (env object missing)"
    FAILED=$((FAILED + 1))
    echo "{\"endpoint\": \"/api/status (env)\", \"status\": \"failed\", \"error\": \"missing_env_object\"}" >> "$RESULTS_FILE"
fi

# Test 10: Check /openapi.json has paths
echo -n "Testing /openapi.json paths ... "
openapi_resp=$(curl -s "$GHOST_BASE_URL/api/openapi.json" 2>&1)
if echo "$openapi_resp" | jq -e '.paths' >/dev/null 2>&1; then
    path_count=$(echo "$openapi_resp" | jq -r '.paths | keys | length')
    if [ "$path_count" -gt 10 ]; then
        echo "✅ PASSED ($path_count paths exposed)"
        PASSED=$((PASSED + 1))
        echo "{\"endpoint\": \"/api/openapi.json\", \"status\": \"passed\", \"path_count\": $path_count}" >> "$RESULTS_FILE"
    else
        echo "❌ FAILED (only $path_count paths)"
        FAILED=$((FAILED + 1))
        echo "{\"endpoint\": \"/api/openapi.json\", \"status\": \"failed\", \"path_count\": $path_count}" >> "$RESULTS_FILE"
    fi
else
    echo "❌ FAILED (no paths object)"
    FAILED=$((FAILED + 1))
    echo "{\"endpoint\": \"/api/openapi.json\", \"status\": \"failed\", \"error\": \"missing_paths_object\"}" >> "$RESULTS_FILE"
fi

# Close JSON array
echo "]" >> "$RESULTS_FILE"

# Summary
echo ""
echo "================================================================================"
echo "TEST SUMMARY"
echo "================================================================================"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "Total:  $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    exit 0
else
    echo "❌ $FAILED TESTS FAILED"
    exit 1
fi
