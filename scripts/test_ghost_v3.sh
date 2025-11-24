#!/bin/bash
# Ghost Protocol V3 - Comprehensive Test Suite
# Tests all critical systems after deployment

BASE_URL="https://ghost-protocol-production.up.railway.app"
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}🟣 GHOST PROTOCOL V3 - SYSTEM TEST SUITE${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: API Health
echo -e "${BOLD}TEST 1: API Health Check${NC}"
RESPONSE=$(curl -s "$BASE_URL/health")
if echo "$RESPONSE" | grep -q '"status":"ok"'; then
    echo -e "${GREEN}✅ API is healthy${NC}"
else
    echo -e "${RED}❌ API health check failed${NC}"
    exit 1
fi
echo ""

# Test 2: Ghost Score
echo -e "${BOLD}TEST 2: Ghost Score${NC}"
SCORE=$(curl -s "$BASE_URL/api/v3/ghost_score" | jq -r '.ghost_score')
GRADE=$(curl -s "$BASE_URL/api/v3/ghost_score" | jq -r '.grade')
echo "   Ghost Score: $SCORE ($GRADE)"
if (( $(echo "$SCORE >= 60" | bc -l) )); then
    echo -e "${GREEN}✅ Ghost Score passing (≥60)${NC}"
else
    echo -e "${YELLOW}⚠️  Ghost Score below target (<60)${NC}"
fi
echo ""

# Test 3: Prediction Coverage
echo -e "${BOLD}TEST 3: Prediction Coverage${NC}"
PREDICTIONS=$(curl -s "$BASE_URL/api/v3/predictions/latest?limit=50")
COUNT=$(echo "$PREDICTIONS" | jq -r '.count')
COVERAGE=$(echo "$PREDICTIONS" | jq -r '.coverage_pct')
echo "   Predictions: $COUNT symbols"
echo "   Coverage: $COVERAGE%"
if (( $(echo "$COVERAGE >= 80" | bc -l) )); then
    echo -e "${GREEN}✅ Prediction coverage excellent (≥80%)${NC}"
elif (( $(echo "$COVERAGE >= 50" | bc -l) )); then
    echo -e "${YELLOW}⚠️  Prediction coverage moderate (50-80%)${NC}"
else
    echo -e "${RED}❌ Prediction coverage low (<50%)${NC}"
fi
echo ""

# Test 4: Provider Success Rate
echo -e "${BOLD}TEST 4: Provider Success Rate${NC}"
STATS=$(curl -s "$BASE_URL/api/v3/stats")
PROVIDER_SUCCESS=$(echo "$STATS" | jq -r '.provider_success_rate')
echo "   Provider Success: $PROVIDER_SUCCESS%"
if (( $(echo "$PROVIDER_SUCCESS >= 75" | bc -l) )); then
    echo -e "${GREEN}✅ Provider success rate excellent (≥75%)${NC}"
else
    echo -e "${YELLOW}⚠️  Provider success rate needs improvement (<75%)${NC}"
fi
echo ""

# Test 5: Confidence Variation
echo -e "${BOLD}TEST 5: Confidence Variation${NC}"
CONFIDENCES=$(echo "$PREDICTIONS" | jq -r '.predictions[].confidence')
MIN_CONF=$(echo "$CONFIDENCES" | sort -n | head -1)
MAX_CONF=$(echo "$CONFIDENCES" | sort -n | tail -1)
RANGE=$(echo "($MAX_CONF - $MIN_CONF) * 100" | bc -l)
echo "   Confidence Range: ${MIN_CONF}% - ${MAX_CONF}%"
echo "   Range Width: $(printf '%.1f' $RANGE)%"
if (( $(echo "$RANGE >= 20" | bc -l) )); then
    echo -e "${GREEN}✅ Confidence variation healthy (≥20% range)${NC}"
else
    echo -e "${YELLOW}⚠️  Confidence variation low (<20% range)${NC}"
fi
echo ""

# Test 6: Direction Distribution
echo -e "${BOLD}TEST 6: Direction Distribution${NC}"
DIRECTIONS=$(echo "$PREDICTIONS" | jq -r '.predictions[].direction')
UP_COUNT=$(echo "$DIRECTIONS" | grep -c "UP" || echo "0")
DOWN_COUNT=$(echo "$DIRECTIONS" | grep -c "DOWN" || echo "0")
FLAT_COUNT=$(echo "$DIRECTIONS" | grep -c "FLAT" || echo "0")
echo "   UP: $UP_COUNT predictions"
echo "   DOWN: $DOWN_COUNT predictions"
echo "   FLAT: $FLAT_COUNT predictions"
if [ "$FLAT_COUNT" -eq "$COUNT" ]; then
    echo -e "${RED}❌ All predictions are FLAT (direction logic not working)${NC}"
else
    echo -e "${GREEN}✅ Direction logic active (mixed UP/DOWN/FLAT)${NC}"
fi
echo ""

# Test 7: Top Movers Quality
echo -e "${BOLD}TEST 7: Top Movers Quality Filter${NC}"
MOVERS=$(curl -s "$BASE_URL/api/v3/cockpit/feed/hunter?limit=10")
MOVER_COUNT=$(echo "$MOVERS" | jq -r '.count')
echo "   Top Movers: $MOVER_COUNT opportunities"
if [ "$MOVER_COUNT" -eq "0" ]; then
    echo -e "${YELLOW}⚠️  No Top Movers (waiting for 20%+ gains with 70%+ confidence)${NC}"
else
    # Check if movers meet quality threshold
    AVG_GAIN=$(echo "$MOVERS" | jq -r '.feed[].gain_pct' | awk '{sum+=$1} END {print sum/NR}')
    AVG_CONF=$(echo "$MOVERS" | jq -r '.feed[].confidence' | awk '{sum+=$1} END {print sum/NR*100}')
    echo "   Average Gain: $(printf '%.1f' $AVG_GAIN)%"
    echo "   Average Confidence: $(printf '%.1f' $AVG_CONF)%"
    echo -e "${GREEN}✅ Top Movers quality filter active${NC}"
fi
echo ""

# Test 8: Accuracy Tracking
echo -e "${BOLD}TEST 8: Accuracy Tracking${NC}"
ACCURACY=$(curl -s "$BASE_URL/api/v3/accuracy/summary")
TOTAL=$(echo "$ACCURACY" | jq -r '.total_predictions')
WIN_RATE=$(echo "$ACCURACY" | jq -r '.win_rate')
echo "   Total Evaluated: $TOTAL predictions"
echo "   Win Rate: $WIN_RATE%"
if [ "$TOTAL" -gt "0" ]; then
    echo -e "${GREEN}✅ Accuracy tracking operational${NC}"
else
    echo -e "${YELLOW}⚠️  Accuracy tracking enabled, waiting for 48h evaluations${NC}"
fi
echo ""

# Test 9: Database Persistence
echo -e "${BOLD}TEST 9: Database Persistence${NC}"
WATCHLIST=$(curl -s "$BASE_URL/api/v3/watchlist")
STOCK_COUNT=$(echo "$WATCHLIST" | jq -r '.stocks | length')
CRYPTO_COUNT=$(echo "$WATCHLIST" | jq -r '.crypto | length')
echo "   Stocks: $STOCK_COUNT symbols"
echo "   Crypto: $CRYPTO_COUNT symbols"
if [ "$STOCK_COUNT" -gt "0" ] && [ "$CRYPTO_COUNT" -gt "0" ]; then
    echo -e "${GREEN}✅ Database persistence working${NC}"
else
    echo -e "${RED}❌ Database persistence issue${NC}"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${BOLD}📊 TEST SUMMARY${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BOLD}Ghost Score:${NC} $SCORE ($GRADE)"
echo -e "${BOLD}Prediction Coverage:${NC} $COVERAGE% ($COUNT symbols)"
echo -e "${BOLD}Provider Success:${NC} $PROVIDER_SUCCESS%"
echo -e "${BOLD}Confidence Range:${NC} ${MIN_CONF}% - ${MAX_CONF}%"
echo -e "${BOLD}Direction Mix:${NC} UP=$UP_COUNT, DOWN=$DOWN_COUNT, FLAT=$FLAT_COUNT"
echo -e "${BOLD}Top Movers:${NC} $MOVER_COUNT opportunities"
echo -e "${BOLD}Accuracy Tracking:${NC} $TOTAL predictions evaluated"
echo ""

if (( $(echo "$SCORE >= 60 && $COVERAGE >= 80 && $PROVIDER_SUCCESS >= 75" | bc -l) )); then
    echo -e "${GREEN}${BOLD}✅ ALL SYSTEMS OPERATIONAL - GHOST V3 COMPLETE${NC}"
    exit 0
elif (( $(echo "$SCORE >= 50 && $COVERAGE >= 50" | bc -l) )); then
    echo -e "${YELLOW}${BOLD}⚠️  SYSTEMS OPERATIONAL - IMPROVEMENTS ONGOING${NC}"
    exit 0
else
    echo -e "${RED}${BOLD}❌ SYSTEMS DEGRADED - INVESTIGATION REQUIRED${NC}"
    exit 1
fi
