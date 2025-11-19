#!/bin/bash
# Ghost Protocol - Comprehensive Health Check
# Tests all critical backend endpoints

set -e

BASE_URL="${1:-https://ghost-protocol-production.up.railway.app}"
FAILED=0

echo "🔍 Ghost Protocol Health Check"
echo "================================"
echo "Base URL: $BASE_URL"
echo ""

check_endpoint() {
    local name="$1"
    local endpoint="$2"
    local expected_status="${3:-200}"
    
    echo -n "Checking $name... "
    
    response=$(curl -s -w "\n%{http_code}" "$BASE_URL$endpoint" 2>&1)
    http_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" = "$expected_status" ]; then
        echo "✅ OK ($http_code)"
        return 0
    else
        echo "❌ FAIL ($http_code)"
        echo "   Response: $body" | head -c 200
        echo ""
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Core health endpoints
echo "📊 Core System"
check_endpoint "Basic Health" "/api/health"
check_endpoint "UI Health" "/ui/health"
check_endpoint "System Status" "/api/status"

# Data endpoints
echo ""
echo "📈 Market Data (Stage 1)"
check_endpoint "Market Mood" "/api/stage1/mood"

# AI/ML endpoints  
echo ""
echo "🤖 AI & Predictions (Stage 2)"
check_endpoint "Prediction Symbols" "/api/predictions/symbols"
check_endpoint "SPY Prediction" "/api/predict/run?symbol=SPY"
check_endpoint "Accuracy Tracker" "/api/stage2/accuracy"

# Configuration
echo ""
echo "⚙️  Configuration"
check_endpoint "Runtime Config" "/api/runtime/config"

# Cockpit
echo ""
echo "🎛️  Cockpit Dashboard"
check_endpoint "Cockpit Data" "/api/cockpit"
check_endpoint "Cockpit UI" "/cockpit"

# Summary
echo ""
echo "================================"
if [ $FAILED -eq 0 ]; then
    echo "✅ All checks passed!"
    exit 0
else
    echo "❌ $FAILED checks failed"
    exit 1
fi
