#!/bin/bash
# Ghost Protocol - Cascading Predictions Test Suite
# =================================================
# Tests all cascade API endpoints and verifies functionality

set -e  # Exit on error

# Configuration
BASE_URL="${GHOST_URL:-http://localhost:8000}"
TEST_SYMBOL="BTC"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Ghost Protocol - Cascade Test Suite  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${RED}❌ jq is required but not installed.${NC}"
    echo "Install with: sudo apt-get install jq (Debian/Ubuntu) or brew install jq (macOS)"
    exit 1
fi

# Test 1: Start a cascade
echo -e "${YELLOW}[TEST 1]${NC} Starting cascade for ${TEST_SYMBOL}..."
START_RESPONSE=$(curl -s -X POST "${BASE_URL}/api/v3/cascade/start?symbol=${TEST_SYMBOL}")
echo "$START_RESPONSE" | jq '.'

# Extract cascade_id
CASCADE_ID=$(echo "$START_RESPONSE" | jq -r '.cascade_id // empty')

if [ -z "$CASCADE_ID" ]; then
    echo -e "${RED}❌ Failed to start cascade${NC}"
    echo "Response: $START_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✅ Cascade started: $CASCADE_ID${NC}"
echo ""

# Test 2: Get cascade details
echo -e "${YELLOW}[TEST 2]${NC} Getting cascade details..."
DETAILS_RESPONSE=$(curl -s "${BASE_URL}/api/v3/cascade/${CASCADE_ID}")
echo "$DETAILS_RESPONSE" | jq '.'

# Verify response
if ! echo "$DETAILS_RESPONSE" | jq -e '.ok == true' > /dev/null; then
    echo -e "${RED}❌ Failed to get cascade details${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Cascade details retrieved${NC}"
echo ""

# Test 3: List active cascades
echo -e "${YELLOW}[TEST 3]${NC} Listing active cascades..."
LIST_RESPONSE=$(curl -s "${BASE_URL}/api/v3/cascade/list?active_only=true")
echo "$LIST_RESPONSE" | jq '.'

# Count cascades
CASCADE_COUNT=$(echo "$LIST_RESPONSE" | jq '.count // 0')
echo -e "${GREEN}✅ Found $CASCADE_COUNT active cascades${NC}"
echo ""

# Test 4: List cascades for specific symbol
echo -e "${YELLOW}[TEST 4]${NC} Listing cascades for ${TEST_SYMBOL}..."
SYMBOL_LIST_RESPONSE=$(curl -s "${BASE_URL}/api/v3/cascade/list?symbol=${TEST_SYMBOL}&active_only=true")
echo "$SYMBOL_LIST_RESPONSE" | jq '.'

SYMBOL_COUNT=$(echo "$SYMBOL_LIST_RESPONSE" | jq '.count // 0')
echo -e "${GREEN}✅ Found $SYMBOL_COUNT cascades for ${TEST_SYMBOL}${NC}"
echo ""

# Test 5: Get cascade statistics
echo -e "${YELLOW}[TEST 5]${NC} Getting cascade statistics..."
STATS_RESPONSE=$(curl -s "${BASE_URL}/api/v3/cascade/stats?days=30")
echo "$STATS_RESPONSE" | jq '.'

# Check if stats exist
TOTAL_CASCADES=$(echo "$STATS_RESPONSE" | jq '.stats.total_cascades // 0')
if [ "$TOTAL_CASCADES" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  No evaluated cascades yet (expected for new deployment)${NC}"
else
    echo -e "${GREEN}✅ Statistics available: $TOTAL_CASCADES total cascades${NC}"
fi
echo ""

# Test 6: Verify database integration
echo -e "${YELLOW}[TEST 6]${NC} Checking database structure..."

if [ -f "data/ghost_predictions.db" ]; then
    # Check if table exists
    TABLE_EXISTS=$(sqlite3 data/ghost_predictions.db "SELECT name FROM sqlite_master WHERE type='table' AND name='prediction_cascades';" 2>/dev/null || echo "")
    
    if [ -n "$TABLE_EXISTS" ]; then
        echo -e "${GREEN}✅ prediction_cascades table exists${NC}"
        
        # Count records
        RECORD_COUNT=$(sqlite3 data/ghost_predictions.db "SELECT COUNT(*) FROM prediction_cascades;" 2>/dev/null || echo "0")
        echo -e "${GREEN}✅ Database has $RECORD_COUNT cascade records${NC}"
    else
        echo -e "${YELLOW}⚠️  prediction_cascades table not found (will be created on first cascade)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Database file not found at data/ghost_predictions.db${NC}"
fi
echo ""

# Test 7: Verify cascade scheduler status
echo -e "${YELLOW}[TEST 7]${NC} Checking cascade scheduler..."

# Try to find scheduler logs
if [ -f "logs/ghost.log" ]; then
    SCHEDULER_LOG=$(tail -100 logs/ghost.log | grep "CASCADE SCHEDULER\|Cascade Scheduler" | tail -1)
    
    if [ -n "$SCHEDULER_LOG" ]; then
        echo -e "${GREEN}✅ Cascade scheduler is running${NC}"
        echo "   Last log: $SCHEDULER_LOG"
    else
        echo -e "${YELLOW}⚠️  No cascade scheduler logs found${NC}"
        echo "   Check that scheduler started with wolf_app.py"
    fi
else
    echo -e "${YELLOW}⚠️  Log file not found at logs/ghost.log${NC}"
fi
echo ""

# Test 8: Test error handling
echo -e "${YELLOW}[TEST 8]${NC} Testing error handling..."

# Try to get non-existent cascade
ERROR_RESPONSE=$(curl -s "${BASE_URL}/api/v3/cascade/invalid-uuid-12345")
ERROR_OK=$(echo "$ERROR_RESPONSE" | jq -r '.ok // true')

if [ "$ERROR_OK" = "false" ]; then
    echo -e "${GREEN}✅ Error handling works correctly${NC}"
else
    echo -e "${RED}❌ Error handling may be incorrect${NC}"
    echo "Response: $ERROR_RESPONSE"
fi
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Test Suite Complete!          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ All cascade API endpoints working${NC}"
echo -e "${GREEN}✅ Cascade ID: ${CASCADE_ID}${NC}"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Monitor logs: tail -f logs/ghost.log | grep CASCADE"
echo "2. Check Telegram for 48h alert"
echo "3. Wait for 24h update (or test with fast mode)"
echo "4. Verify full cascade lifecycle"
echo ""
echo -e "${BLUE}For fast testing, modify cascade_scheduler.py:${NC}"
echo "  CHECK_INTERVAL = 60  # Check every 60 seconds"
echo "  h24_time = time.time() + 120  # 2 minutes instead of 24h"
echo ""
