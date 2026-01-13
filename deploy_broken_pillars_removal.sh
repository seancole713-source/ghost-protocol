#!/bin/bash
# Quick deployment script for broken pillars removal

echo "==========================================="
echo "Deploying Broken Pillars Removal"
echo "==========================================="
echo ""

# Show what's being deployed
echo "📦 Changes to deploy:"
echo "   ✅ Disabled sentiment_engine (dummy 0.0 data)"
echo "   ✅ Disabled world_context_engine (dummy null/50 data)"
echo "   ✅ Updated health check (4 pillars instead of 6)"
echo "   ✅ Created Ghost News Brain verification script"
echo ""

# Confirm deployment
read -p "Deploy to production? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 1
fi

# Git operations
echo ""
echo "📝 Committing changes..."
git add core/data_pillars/feature_orchestrator.py
git add verify_ghost_news_brain.sh
git add BROKEN_PILLARS_REMOVAL.md
git add deploy_broken_pillars_removal.sh

git commit -m "Remove broken sentiment and world_context pillars

Changes:
- Disabled sentiment_engine (Alpha Vantage API timeout, returns 0.0 dummy)
- Disabled world_context_engine (SPY/VIX price fetch fails, returns null/50)
- Updated health check to expect 4 pillars instead of 6
- Created verification script for Ghost News Brain status

Impact:
- Faster predictions (no API timeouts)
- Fewer errors in logs
- No loss of accuracy (pillars returned constant values)
- Simpler system with fewer failure points

Verification:
- Local tests passed ✅
- Feature extraction working ✅
- Sentiment/world_context features removed ✅
- Health check updated ✅"

echo ""
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "==========================================="
echo "✅ Deployment Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Railway will auto-deploy from main branch"
echo "2. Monitor Railway logs for successful startup"
echo "3. Run: railway run bash"
echo "4. Then run: ./verify_ghost_news_brain.sh"
echo "5. Decide: Keep News Brain enabled or disable it"
echo "6. Monitor win rate for 24-48 hours"
echo ""
echo "Expected outcome:"
echo "✅ Same or better win rate (70%+)"
echo "✅ Faster predictions"
echo "✅ Fewer errors in logs"
echo ""
