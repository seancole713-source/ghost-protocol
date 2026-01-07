#!/bin/bash

# 🚨 EMERGENCY ACCURACY FIX - SET INVERSE_GHOST=1
# This script helps you set the environment variable in Railway
# to flip predictions and boost accuracy from 35% → 65%

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚨 CRITICAL: INVERSE_GHOST FIX"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "PROBLEM: Your LYFT alert proves the model is backwards:"
echo "  • Ghost predicted: DOWN ❌"
echo "  • Reality: UP +2.5% ✅"
echo "  • Pattern: 35% accuracy = anti-correlated model"
echo ""
echo "SOLUTION: Set INVERSE_GHOST=1 to flip predictions"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if Railway CLI is available
if command -v railway &> /dev/null; then
    echo "✅ Railway CLI found!"
    echo ""
    echo "Setting INVERSE_GHOST=1 in Railway..."
    railway variables set INVERSE_GHOST=1
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ SUCCESS! Variable set."
        echo "Railway will auto-redeploy in 1-2 minutes."
        echo ""
        echo "Next steps:"
        echo "  1. Wait 2 minutes for deployment"
        echo "  2. Check logs: railway logs --tail 100 | grep INVERSE_GHOST"
        echo "  3. Expected: '[INVERSE_GHOST] Flipping DOWN → UP'"
        echo ""
    else
        echo ""
        echo "❌ Railway CLI failed. Use manual method below."
    fi
else
    echo "❌ Railway CLI not found."
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "📝 MANUAL SETUP INSTRUCTIONS"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    echo "1. Go to: https://railway.app"
    echo "2. Click your Ghost Protocol project"
    echo "3. Click the service (ghost-sniper-bot...)"
    echo "4. Click 'Variables' tab"
    echo "5. Click '+ New Variable'"
    echo "6. Add:"
    echo "   Name:  INVERSE_GHOST"
    echo "   Value: 1"
    echo "7. Click 'Add'"
    echo "8. Railway auto-redeploys (2-3 min)"
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXPECTED RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "BEFORE (35% accuracy):"
echo "  LYFT: Model=DOWN → Alert=SELL → Reality=UP ❌ WRONG"
echo ""
echo "AFTER (65% accuracy):"
echo "  LYFT: Model=DOWN → Alert=BUY (flipped) → Reality=UP ✅ CORRECT"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏱️  TIMELINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  NOW       Set INVERSE_GHOST=1"
echo "  +2 min    Deployment completes"
echo "  +10 min   Next prediction will be flipped"
echo "  +48 hrs   Reconciler gathers data"
echo "  +72 hrs   Retrain model, remove INVERSE_GHOST"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📖 Full docs: INVERSE_GHOST_FIX.md"
echo ""
