#!/bin/bash
# Deploy Personal Watchlist to Railway
# This script commits changes and triggers Railway deployment

set -e

echo "=========================================="
echo "Personal Watchlist Deployment Script"
echo "=========================================="
echo ""

# Check if running in dev container
if [ ! -d "/workspaces/ghost-protocol" ]; then
    echo "❌ Must run from Ghost Protocol workspace"
    exit 1
fi

cd /workspaces/ghost-protocol

echo "📋 Step 1: Checking new files..."
NEW_FILES=(
    "migrations/001_personal_watchlist.sql"
    "core/personal_watchlist.py"
    "core/watchlist_prediction_scheduler.py"
    "core/watchlist_telegram_alerts.py"
    "api/personal_watchlist_endpoints.py"
    "static/personal_watchlist_ui.js"
    "PERSONAL_WATCHLIST_README.md"
    "scripts/integrate_personal_watchlist.py"
    "tests/test_personal_watchlist.py"
)

for file in "${NEW_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ MISSING: $file"
        exit 1
    fi
done

echo ""
echo "📋 Step 2: Verifying wolf_app.py integration..."
if grep -q "from api.personal_watchlist_endpoints import router as watchlist_router" wolf_app.py; then
    echo "  ✅ API router imported"
else
    echo "  ❌ API router NOT imported"
    exit 1
fi

if grep -q "start_watchlist_scheduler()" wolf_app.py; then
    echo "  ✅ Scheduler startup hook added"
else
    echo "  ❌ Scheduler startup hook NOT added"
    exit 1
fi

echo ""
echo "📋 Step 3: Committing changes..."
if command -v git &> /dev/null; then
    git add migrations/ core/ api/ static/ scripts/ tests/ PERSONAL_WATCHLIST_README.md wolf_app.py deploy_personal_watchlist.sh 2>/dev/null || true
    git status --short
    
    echo ""
    read -p "Commit these changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git commit -m "feat: Add personal watchlist module with Postgres persistence

- Database schema with 4 tables (watchlist_items, prediction_tracking, price_snapshots, alerts_log)
- Core PersonalWatchlistManager with CRUD operations
- WatchlistPredictionScheduler for daily/intraday predictions
- WatchlistTelegramAlerter for market alerts
- 7 REST API endpoints (/add, /remove, /user, /update-position, /history, /trigger-prediction, /stats)
- Cockpit UI module with add/remove/toggle ownership/history viewer
- Comprehensive test suite (25 tests)
- Integration with wolf_app.py (router + scheduler)

See PERSONAL_WATCHLIST_README.md for full documentation"
        
        echo ""
        echo "📋 Step 4: Pushing to Railway..."
        CURRENT_BRANCH=$(git branch --show-current)
        git push origin "$CURRENT_BRANCH"
        
        echo ""
        echo "✅ Code pushed! Railway will auto-deploy."
        echo ""
        echo "📋 Step 5: Database migration required..."
        echo ""
        echo "Run this command on Railway (via railway run or exec):"
        echo ""
        echo "  psql \$DATABASE_URL -f migrations/001_personal_watchlist.sql"
        echo ""
        echo "Or manually via Railway dashboard > Database > Query:"
        echo "  Copy/paste contents of migrations/001_personal_watchlist.sql"
        echo ""
    else
        echo "❌ Commit cancelled"
        exit 1
    fi
else
    echo "⚠️ git not found - manual deployment required"
    echo ""
    echo "Next steps:"
    echo "1. Commit all files to git repository"
    echo "2. Push to main/production branch"
    echo "3. Railway will auto-deploy"
    echo "4. Run database migration on Railway"
fi

echo ""
echo "📋 Step 6: Post-deployment checklist..."
echo ""
echo "After Railway deployment completes:"
echo ""
echo "1. Run database migration (see command above)"
echo "2. Set environment variables in Railway dashboard:"
echo "   - WATCHLIST_SCHEDULER_ENABLED=1"
echo "   - WATCHLIST_ALERTS_ENABLED=1"
echo "   - WATCHLIST_OPEN_HOUR=9"
echo "   - WATCHLIST_CLOSE_HOUR=16"
echo "   - WATCHLIST_BIG_MOVE_CHECK_MINUTES=15"
echo "   - WATCHLIST_BIG_MOVE_THRESHOLD_PCT=5.0"
echo "   - WATCHLIST_ALERT_COOLDOWN_HOURS=4"
echo "   - WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR=5"
echo ""
echo "3. Verify deployment:"
echo "   curl https://ghost-protocol-production.up.railway.app/api/v3/watchlist/user"
echo ""
echo "4. Seed default symbols (optional):"
echo "   python3 scripts/integrate_personal_watchlist.py --seed-default"
echo ""
echo "5. Test in Cockpit UI:"
echo "   - Open https://ghost-protocol-production.up.railway.app/cockpit"
echo "   - Look for 'Add Symbol' button in watchlist panel"
echo "   - Add a test symbol (e.g., BTC crypto)"
echo "   - Verify it appears with predictions"
echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
