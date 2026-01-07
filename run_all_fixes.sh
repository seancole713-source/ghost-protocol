#!/bin/bash
# Ghost Protocol - Complete Fix Deployment Script
# Runs all 3 steps to make synapses GREEN
# January 7, 2026

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║        🧠 GHOST PROTOCOL - SYNAPSE FIX DEPLOYMENT                    ║"
echo "║                                                                      ║"
echo "║   This script will:                                                  ║"
echo "║   1. Test PostgreSQL fixes (30 seconds)                             ║"
echo "║   2. Retrain XGBoost model (2-5 minutes)                            ║"
echo "║   3. Enable INVERSE_GHOST if needed (30 seconds)                    ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ ERROR: Railway CLI not installed"
    echo ""
    echo "Install it with:"
    echo "  npm install -g @railway/cli"
    echo "  or"
    echo "  brew install railway"
    echo ""
    exit 1
fi

echo "✅ Railway CLI found"
echo ""

# Check if linked to Railway project
echo "🔗 Checking Railway project link..."
if ! railway status &> /dev/null; then
    echo "⚠️  Not linked to Railway project"
    echo ""
    echo "Run: railway link"
    echo "Then run this script again"
    exit 1
fi

echo "✅ Linked to Railway project"
echo ""

# ============================================================================
# STEP 1: Test PostgreSQL Fixes
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════════"
echo "STEP 1: Testing PostgreSQL Fixes"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "⏱️  Estimated time: 30 seconds"
echo ""
echo "Running: railway run python3 test_postgres_fixes.py"
echo ""

railway run python3 test_postgres_fixes.py

STEP1_EXIT=$?

if [ $STEP1_EXIT -ne 0 ]; then
    echo ""
    echo "❌ STEP 1 FAILED: PostgreSQL tests failed"
    echo ""
    echo "This means:"
    echo "  - DATABASE_URL not configured, OR"
    echo "  - ml_trainer can't fetch from PostgreSQL, OR"
    echo "  - learning_loop can't query outcomes"
    echo ""
    echo "Check the error above and fix before proceeding"
    exit 1
fi

echo ""
echo "✅ STEP 1 PASSED: All PostgreSQL connections GREEN!"
echo ""
echo "Press ENTER to continue to Step 2 (retrain model)..."
read

# ============================================================================
# STEP 2: Retrain XGBoost Model
# ============================================================================
echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "STEP 2: Retraining XGBoost Model with PostgreSQL Data"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "⏱️  Estimated time: 2-5 minutes"
echo ""
echo "This will:"
echo "  - Fetch 25,691+ outcomes from PostgreSQL"
echo "  - Train XGBoost v3 on REAL data (not empty SQLite)"
echo "  - Evaluate train/test accuracy"
echo "  - Save to models/production/ghost_model_ALL.pkl"
echo ""
echo "Running: railway run python3 retrain_model.py"
echo ""

railway run python3 retrain_model.py | tee /tmp/retrain_output.txt

STEP2_EXIT=$?

if [ $STEP2_EXIT -ne 0 ]; then
    echo ""
    echo "❌ STEP 2 FAILED: Model retraining failed"
    echo ""
    echo "Check the error above"
    exit 1
fi

echo ""
echo "✅ STEP 2 COMPLETE: Model retrained!"
echo ""

# Extract test accuracy from output
TEST_ACC=$(grep "Test accuracy:" /tmp/retrain_output.txt | tail -1 | grep -oP '\d+\.\d+%' | grep -oP '\d+\.\d+' || echo "unknown")

echo "📊 Test Accuracy: $TEST_ACC%"
echo ""

# ============================================================================
# STEP 3: Decide on INVERSE_GHOST
# ============================================================================
echo "═══════════════════════════════════════════════════════════════════════"
echo "STEP 3: Check if INVERSE_GHOST needed"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""

if [ "$TEST_ACC" != "unknown" ]; then
    # Convert to integer for comparison (50.5% -> 50)
    ACC_INT=$(echo "$TEST_ACC" | cut -d. -f1)
    
    if [ "$ACC_INT" -lt 50 ]; then
        echo "⚠️  Test accuracy $TEST_ACC% is BELOW 50% (worse than random)"
        echo ""
        echo "This means the model is ANTI-CORRELATED!"
        echo ""
        echo "🔧 RECOMMENDED ACTION: Enable INVERSE_GHOST=1"
        echo ""
        echo "This will flip UP/DOWN predictions and turn:"
        echo "  $TEST_ACC% accuracy → ~$(echo "100 - $TEST_ACC" | bc)% accuracy"
        echo ""
        echo "Do you want to enable INVERSE_GHOST=1? (y/n)"
        read -r ENABLE_INVERSE
        
        if [[ "$ENABLE_INVERSE" =~ ^[Yy]$ ]]; then
            echo ""
            echo "Setting INVERSE_GHOST=1 on Railway..."
            railway variables set INVERSE_GHOST=1
            
            echo ""
            echo "✅ INVERSE_GHOST=1 enabled!"
            echo ""
            echo "Redeploying to apply changes..."
            railway up -d
            
            echo ""
            echo "✅ Deployment triggered!"
            echo ""
        else
            echo ""
            echo "⚠️  INVERSE_GHOST NOT enabled"
            echo "   Model will continue with $TEST_ACC% accuracy"
            echo ""
        fi
    else
        echo "✅ Test accuracy $TEST_ACC% is ABOVE 50%!"
        echo ""
        echo "Model is correctly correlated - no need for INVERSE_GHOST"
        echo ""
        
        # Check if INVERSE_GHOST is currently enabled
        CURRENT_INVERSE=$(railway variables get INVERSE_GHOST 2>/dev/null || echo "0")
        
        if [ "$CURRENT_INVERSE" = "1" ]; then
            echo "⚠️  But INVERSE_GHOST=1 is currently ENABLED"
            echo ""
            echo "This would FLIP your predictions and make accuracy WORSE!"
            echo ""
            echo "Do you want to DISABLE INVERSE_GHOST? (y/n)"
            read -r DISABLE_INVERSE
            
            if [[ "$DISABLE_INVERSE" =~ ^[Yy]$ ]]; then
                echo ""
                echo "Setting INVERSE_GHOST=0 on Railway..."
                railway variables set INVERSE_GHOST=0
                
                echo ""
                echo "✅ INVERSE_GHOST disabled!"
                echo ""
                echo "Redeploying to apply changes..."
                railway up -d
                
                echo ""
                echo "✅ Deployment triggered!"
                echo ""
            fi
        fi
    fi
else
    echo "⚠️  Could not determine test accuracy from output"
    echo ""
    echo "Check manually and decide if you need INVERSE_GHOST=1:"
    echo ""
    echo "  If accuracy < 50%: railway variables set INVERSE_GHOST=1"
    echo "  If accuracy > 50%: railway variables set INVERSE_GHOST=0"
    echo ""
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                      ║"
echo "║                    🎉 ALL STEPS COMPLETE!                            ║"
echo "║                                                                      ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 SUMMARY:"
echo ""
echo "  Step 1: ✅ PostgreSQL tests passed"
echo "  Step 2: ✅ Model retrained (test acc: $TEST_ACC%)"
if [ "$TEST_ACC" != "unknown" ]; then
    ACC_INT=$(echo "$TEST_ACC" | cut -d. -f1)
    if [ "$ACC_INT" -lt 50 ]; then
        echo "  Step 3: ⚠️  INVERSE_GHOST recommended (model anti-correlated)"
    else
        echo "  Step 3: ✅ Model correctly correlated, no INVERSE_GHOST needed"
    fi
else
    echo "  Step 3: ⚠️  Check accuracy manually"
fi
echo ""
echo "🔍 VERIFICATION COMMANDS:"
echo ""
echo "  # Check current accuracy"
echo "  railway run python3 -c \"from core.learning_loop import get_learning_loop; print(get_learning_loop()._get_postgres_direction_accuracy(days=7))\""
echo ""
echo "  # Check INVERSE_GHOST setting"
echo "  railway variables get INVERSE_GHOST"
echo ""
echo "  # View recent predictions"
echo "  railway logs --tail 100"
echo ""
echo "✅ All synapses should now be GREEN!"
echo ""
