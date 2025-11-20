#!/bin/bash
# Ghost Hunter Implementation - Test Commands
# Run these to verify all features work correctly

set -e

echo "=================================================="
echo "GHOST HUNTER - FEATURE TESTING"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Test 1: Simple Alert Formatter${NC}"
python3 -c "
from core.telegram_alerts import Alert, format_simple_alert
import time

alert = Alert(
    symbol='WOLF',
    market='stock',
    direction='BUY',
    confidence=0.78,
    price_now=17.51,
    price_prev=16.62,
    change_pct=5.2,
    horizon_h=6
)

print('\nBalanced format:')
print(format_simple_alert(alert))
print(f'Length: {len(format_simple_alert(alert))} chars')
"
echo ""

echo -e "${YELLOW}Test 2: Feature Diagnostics${NC}"
python3 -c "
from core.feature_diagnostics import diagnose_features, build_confidence_with_diagnostics
import time
import json

# Test with good features
status = diagnose_features(
    symbol='WOLF',
    price_data={'price': 17.51, 'timestamp': time.time(), 'provider': 'polygon'},
    volume_data={'volume': 1000000, 'avg_volume': 500000},
    momentum_data={'momentum_score': 0.65, 'trend': 'up'},
    context_data={'market_regime': 'bull'},
    sentiment_data={'sentiment_score': 0.55}
)

print('\nFeature Status (all healthy):')
print(json.dumps(status.to_dict(), indent=2))
print(f'Usable: {status.is_usable()}')

# Test confidence adjustment
conf, meta = build_confidence_with_diagnostics(0.75, status)
print(f'\nBase confidence: 0.75 → Adjusted: {conf}')
"
echo ""

echo -e "${YELLOW}Test 3: Feature Diagnostics (Degraded)${NC}"
python3 -c "
from core.feature_diagnostics import diagnose_features, build_confidence_with_diagnostics
import json

# Test with degraded features (price missing)
status = diagnose_features(
    symbol='WOLF',
    price_data=None,  # Missing price = critical failure
    volume_data={'volume': 1000}
)

print('\nFeature Status (degraded - price missing):')
print(json.dumps(status.to_dict(), indent=2))
print(f'Usable: {status.is_usable()}')

# Confidence should be forced to 0
conf, meta = build_confidence_with_diagnostics(0.75, status)
print(f'\nBase confidence: 0.75 → Adjusted: {conf}')
print(f'Adjustment reason: {meta.get(\"confidence_adjustment\")}')
"
echo ""

echo -e "${YELLOW}Test 4: Price Reliability${NC}"
python3 -c "
from core.price_reliability import get_provider_stats, reset_provider_stats
import json

# Mock some provider activity for testing
from core.price_reliability import _record_provider_success, _record_provider_failure

reset_provider_stats()
_record_provider_success('polygon', 245.3)
_record_provider_success('polygon', 198.7)
_record_provider_failure('polygon', 1200.0)
_record_provider_success('yahoo', 456.2)

stats = get_provider_stats()
print('\nProvider Statistics:')
print(json.dumps(stats, indent=2))
"
echo ""

echo -e "${GREEN}=================================================="
echo "ALL TESTS COMPLETE"
echo "==================================================${NC}"
echo ""
echo "Next steps:"
echo "1. Commit changes: git add -A && git commit -m 'feat: Ghost Hunter Phase 1'"
echo "2. Deploy: git push origin main"
echo "3. Configure Railway env vars:"
echo "   - ALERT_STYLE=simple"
echo "   - ALERT_SIMPLE_FORMAT=balanced"
echo "   - MIN_ALERT_CONFIDENCE=0.60"
echo "   - PRICE_SOURCE_PRIMARY=polygon"
echo "   - PRICE_SOURCE_SECONDARY=yahoo"
echo ""
