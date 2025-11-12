#!/bin/bash
#
# Railway Deployment Monitor & Smoke Test
# Monitors Railway deployment status and runs smoke tests when ready
#

set -e

GHOST_BASE_URL="${GHOST_BASE_URL:-https://ghost-sniper-bot-seancole713-production.up.railway.app}"
GHOST_API_TOKEN="${GHOST_API_TOKEN:-edaa4eac-6455-4693-a745-142cb6deef03}"

echo "========================================"
echo "Railway Deployment Monitor"
echo "========================================"
echo "Base URL: $GHOST_BASE_URL"
echo "Commit: 77e4971"
echo ""

# Function to test if new routes are live
test_new_routes() {
    local tick_status=$(curl -s -o /dev/null -w "%{http_code}" "$GHOST_BASE_URL/api/tick" 2>/dev/null)
    local regime_status=$(curl -s -o /dev/null -w "%{http_code}" "$GHOST_BASE_URL/api/regime/current" 2>/dev/null)
    
    if [ "$tick_status" = "200" ] && [ "$regime_status" = "200" ]; then
        return 0  # New routes are live
    else
        return 1  # Still old deployment
    fi
}

echo "Checking deployment status..."
echo ""

if test_new_routes; then
    echo "✅ NEW DEPLOYMENT DETECTED!"
    echo ""
    echo "Running smoke tests..."
    echo "========================================"
    bash /app/deployment_smoke_test.sh
else
    echo "⏳ Old deployment still running (404 on new routes)"
    echo ""
    echo "Waiting for Railway to deploy commit 77e4971..."
    echo ""
    echo "Manual trigger options:"
    echo "  1. Railway Dashboard → Deployments → Redeploy"
    echo "  2. Railway CLI: railway up --detach"
    echo ""
    echo "Once deployed, re-run this script to verify:"
    echo "  bash /app/railway_monitor.sh"
    echo ""
    
    # Wait and retry
    echo "Auto-checking every 30 seconds (max 5 minutes)..."
    for i in {1..10}; do
        echo "Check $i/10..."
        sleep 30
        
        if test_new_routes; then
            echo ""
            echo "✅ DEPLOYMENT DETECTED!"
            echo ""
            echo "Running smoke tests..."
            echo "========================================"
            bash /app/deployment_smoke_test.sh
            exit 0
        fi
    done
    
    echo ""
    echo "⚠️  Deployment not detected after 5 minutes"
    echo "   Please manually trigger deployment in Railway Dashboard"
    exit 1
fi
