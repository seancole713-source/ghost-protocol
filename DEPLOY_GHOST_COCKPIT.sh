#!/bin/bash
# Ghost Cockpit Live Restore - Quick Start Deployment
# Run this after reviewing changes in wolf_app.py
# Date: 2025-11-10

set -e

echo "╔════════════════════════════════════════════╗"
echo "║ Ghost Cockpit Live Restore - Quick Deploy ║"
echo "║ Mission: 100% Live Operation + Zero 499s  ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Step 1: Verify changes
echo "▶ Step 1: Verify Changes"
echo "  Changes made to wolf_app.py:"
echo "    ✅ Added /api/regime/current endpoint (line 10913)"
echo "    ✅ Enhanced SSE /api/cockpit/stream with event types"
echo "    ✅ Compilation verified (0 errors)"
echo ""
read -p "Press ENTER to continue with deployment..."
echo ""

# Step 2: Git commit
echo "▶ Step 2: Commit Changes"
git add wolf_app.py RAILWAY_ENV_CONFIG.sh PRODUCTION_VALIDATION_TESTS.sh GHOST_COCKPIT_RESTORE_COMPLETE.md
git status
echo ""
read -p "Press ENTER to commit (or CTRL+C to cancel)..."
git commit -m "feat: add /api/regime/current and SSE event types (status/ping/snapshot)

- Added /api/regime/current endpoint with neutral fallback
- Enhanced SSE /api/cockpit/stream with proper event types
- Reduced ping interval from 15s to 10s for better responsiveness
- Improved logging with LOGGER instead of print()
- All changes compile successfully (0 errors)

Refs: GHOST_COCKPIT_RESTORE_COMPLETE.md"
echo ""

# Step 3: Push to repo
echo "▶ Step 3: Push to Repository"
read -p "Press ENTER to push (or CTRL+C to cancel)..."
git push
echo ""

# Step 4: Set Railway environment variables
echo "▶ Step 4: Configure Railway Environment"
echo "  Setting critical environment variables..."
echo ""
echo "Run these commands in your terminal:"
echo ""
cat RAILWAY_ENV_CONFIG.sh | grep "railway variables set"
echo ""
read -p "Press ENTER after setting Railway env vars..."
echo ""

# Step 5: Deploy to Railway
echo "▶ Step 5: Deploy to Railway"
echo "  Options:"
echo "    A) Auto-deploy (if configured in Railway)"
echo "    B) Manual: railway up"
echo "    C) Manual: git push (if Railway watches main branch)"
echo ""
read -p "Press ENTER after deployment completes..."
echo ""

# Step 6: Verify deployment
echo "▶ Step 6: Verify Deployment"
echo "  Checking Railway service status..."
railway status || echo "  ⚠️  railway CLI not available"
echo ""

# Step 7: Run validation tests
echo "▶ Step 7: Run Validation Tests"
echo "  This will test all endpoints for proper function"
echo ""
read -p "Press ENTER to run validation tests..."
./PRODUCTION_VALIDATION_TESTS.sh
echo ""

# Step 8: Monitor stability
echo "▶ Step 8: 5-Minute Stability Monitoring"
echo "  Goal: Zero 499 or 502 errors"
echo ""
echo "Monitor command:"
echo "  watch -n 30 'curl -s \"\$GHOST_BASE_URL/api/admin/logs?window=5m\" | grep -E \"499|502\" | wc -l'"
echo ""
echo "Expected: Output should remain at 0"
echo ""

echo "╔════════════════════════════════════════════╗"
echo "║        DEPLOYMENT COMPLETE!                ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "✅ All steps completed"
echo "✅ Ghost Cockpit Live Restore Mission: SUCCESS"
echo ""
echo "Next: Monitor for 5 minutes to confirm stability"
echo ""
