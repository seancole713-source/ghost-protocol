#!/bin/bash
# Quick validation script that uses Railway's DATABASE_URL

echo "🔍 Ghost Protocol - Quick Validation"
echo "====================================="
echo ""

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set in environment"
    echo ""
    echo "To run this validation, you need DATABASE_URL from Railway:"
    echo ""
    echo "Option 1: Use Railway CLI"
    echo "  railway run python3 validate_ghost_predictions.py"
    echo ""
    echo "Option 2: Set manually (get from Railway dashboard)"
    echo "  export DATABASE_URL='postgresql://postgres:...@...railway.app:...'"
    echo "  python3 validate_ghost_predictions.py"
    echo ""
    echo "Option 3: Use the API endpoint directly"
    echo "  curl 'https://ghost-protocol-production.up.railway.app/api/v3/accuracy/summary?period_days=30'"
    echo ""
    exit 1
fi

# Run validation
python3 validate_ghost_predictions.py
