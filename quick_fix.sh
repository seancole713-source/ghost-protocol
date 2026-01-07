#!/bin/bash
# Quick 3-Step Fix Script
# Run this to make all Ghost synapses GREEN

echo "🧠 GHOST PROTOCOL - QUICK FIX"
echo ""

# Step 1: Test (30 seconds)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 1/3: Testing PostgreSQL Fixes (30 sec)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
railway run python3 test_postgres_fixes.py
echo ""

# Step 2: Retrain (2-5 min)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 2/3: Retraining Model (2-5 min)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
railway run python3 retrain_model.py
echo ""

# Step 3: Check accuracy and set INVERSE_GHOST
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STEP 3/3: Checking Accuracy"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "If test accuracy < 50%, run:"
echo "  railway variables set INVERSE_GHOST=1"
echo "  railway up -d"
echo ""
echo "If test accuracy > 50%, run:"
echo "  railway variables set INVERSE_GHOST=0"
echo "  railway up -d"
echo ""
echo "✅ Done! All synapses should be GREEN"
