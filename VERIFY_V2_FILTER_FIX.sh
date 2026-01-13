#!/bin/bash
# Verification script to prove DASH and LRC are now BLOCKED
# Run this AFTER Railway deploys commit 583a1f2

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     V2 FILTER FIX VERIFICATION - DASH & LRC BLOCKED        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check 1: Verify Railway has latest deployment
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECK 1: Deployment Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Required commit: 583a1f2 (Fix V2 filter bypass)"
echo "Current commits:"
git log --oneline -3
echo ""
echo "⚠️  Check Railway dashboard manually:"
echo "    Go to: https://railway.app/project/ghost-protocol"
echo "    Verify: Latest deployment shows commit 583a1f2"
echo ""
read -p "Press ENTER once Railway shows 583a1f2 deployed..."

# Check 2: Railway logs for V2-CLEANUP message
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECK 2: Startup Cleanup Logs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Looking for V2-CLEANUP logs showing DASH/LRC removed from cache..."
echo ""
railway logs --tail 1000 | grep -E "V2-CLEANUP|BLOCKED.*DASH|BLOCKED.*LRC" || echo "No V2-CLEANUP logs yet (may need to wait for restart)"
echo ""

# Check 3: Try to generate DASH prediction (should be BLOCKED)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECK 3: Test DASH Prediction (Should BLOCK)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Attempting to predict DASH..."
railway run python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    from wolf_app import run_single_prediction
    result = run_single_prediction('DASH')
    print(f\"Result: {result.get('ok')}\")
    print(f\"Direction: {result.get('direction')}\")
    print(f\"Error: {result.get('error', 'N/A')}\")
    print(f\"V2 Filtered: {result.get('v2_filtered', False)}\")
    if result.get('direction') == 'BLOCKED':
        print('✅ DASH BLOCKED by V2 filter')
    else:
        print('❌ DASH NOT BLOCKED - FIX FAILED')
except Exception as e:
    print(f'Error testing: {e}')
" 2>&1 || echo "❌ Command failed (Railway CLI issue or Python path problem)"
echo ""

# Check 4: Try to generate LRC prediction (should be BLOCKED)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECK 4: Test LRC Prediction (Should BLOCK)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Attempting to predict LRC..."
railway run python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    from wolf_app import run_single_prediction
    result = run_single_prediction('LRC')
    print(f\"Result: {result.get('ok')}\")
    print(f\"Direction: {result.get('direction')}\")
    print(f\"Error: {result.get('error', 'N/A')}\")
    print(f\"V2 Filtered: {result.get('v2_filtered', False)}\")
    if result.get('direction') == 'BLOCKED':
        print('✅ LRC BLOCKED by V2 filter')
    else:
        print('❌ LRC NOT BLOCKED - FIX FAILED')
except Exception as e:
    print(f'Error testing: {e}')
" 2>&1 || echo "❌ Command failed (Railway CLI issue or Python path problem)"
echo ""

# Check 5: Query paper_trades for new DASH/LRC entries
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECK 5: Database - No New DASH/LRC Paper Trades"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Checking for DASH/LRC paper trades created AFTER fix deployed..."
railway run python3 -c "
import sys
sys.path.insert(0, '/app')
import os
try:
    import psycopg2
    from datetime import datetime, timedelta
    
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cur = conn.cursor()
    
    # Check for DASH/LRC paper trades in last 6 hours
    cutoff = datetime.now() - timedelta(hours=6)
    cur.execute('''
        SELECT symbol, COUNT(*) as count, MAX(created_at) as latest
        FROM paper_trades 
        WHERE symbol IN ('DASH', 'LRC')
        AND created_at > %s
        GROUP BY symbol
    ''', (cutoff,))
    
    results = cur.fetchall()
    if results:
        print('❌ FOUND new DASH/LRC paper trades:')
        for row in results:
            print(f'   {row[0]}: {row[1]} trades, latest: {row[2]}')
        print('FIX FAILED - predictions still being logged')
    else:
        print('✅ No new DASH/LRC paper trades (fix working)')
    
    cur.close()
    conn.close()
except Exception as e:
    print(f'Database query error: {e}')
" 2>&1 || echo "❌ Database query failed"
echo ""

# Check 6: Verify whitelist symbols CAN still predict
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CHECK 6: Whitelist Symbol (RNDR) Can Still Predict"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Attempting to predict RNDR (should SUCCEED)..."
railway run python3 -c "
import sys
sys.path.insert(0, '/app')
try:
    from wolf_app import run_single_prediction
    result = run_single_prediction('RNDR')
    print(f\"Result: {result.get('ok')}\")
    print(f\"Direction: {result.get('direction')}\")
    print(f\"Confidence: {result.get('confidence', 0):.1%}\")
    if result.get('ok'):
        print('✅ RNDR prediction succeeded (whitelist working)')
    else:
        print(f\"❌ RNDR blocked: {result.get('error')} - whitelist broken!\")
except Exception as e:
    print(f'Error testing: {e}')
" 2>&1 || echo "❌ Command failed"
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Expected Results:"
echo "  ✅ V2-CLEANUP logs show DASH/LRC removed from cache"
echo "  ✅ DASH prediction returns direction='BLOCKED'"
echo "  ✅ LRC prediction returns direction='BLOCKED'"
echo "  ✅ No new DASH/LRC paper trades in last 6 hours"
echo "  ✅ RNDR prediction succeeds (whitelist works)"
echo ""
echo "If all checks pass, V2 filter is working correctly."
echo "DASH and LRC will NO LONGER appear in TOP 10 or alerts."
echo ""
