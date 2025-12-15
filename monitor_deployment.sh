#!/bin/bash
# Monitor Ghost Protocol deployment progress
# Tracks when Polygon integration activates outcome reconciliation

BASE_URL="https://ghost-protocol-production.up.railway.app"

echo "════════════════════════════════════════════════════════════════"
echo "  GHOST PROTOCOL — DEPLOYMENT MONITOR"
echo "  Tracking Polygon Historical Price Integration"
echo "════════════════════════════════════════════════════════════════"
echo ""

check_count=0
max_checks=20  # Monitor for ~10 minutes

while [ $check_count -lt $max_checks ]; do
    check_count=$((check_count + 1))
    timestamp=$(date '+%H:%M:%S')
    
    echo "[$timestamp] Check #$check_count/$max_checks"
    echo "─────────────────────────────────────────────────────────────"
    
    # 1. Check if deployment is live
    health=$(curl -s "$BASE_URL/health" 2>/dev/null | grep -o '"ok":true' || echo "")
    if [ -z "$health" ]; then
        echo "⏳ Railway deploying... (server not responding)"
        echo ""
        sleep 30
        continue
    fi
    
    # 2. Trigger manual reconciliation
    echo "▸ Triggering reconciliation..."
    recon_result=$(curl -s -X POST "$BASE_URL/api/v3/accuracy/reconcile" 2>/dev/null)
    reconciled=$(echo "$recon_result" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('reconciled', 0))" 2>/dev/null || echo "0")
    
    # 3. Check outcome status
    echo "▸ Checking outcomes..."
    accuracy=$(curl -s "$BASE_URL/api/v3/accuracy/dashboard" 2>/dev/null)
    total_outcomes=$(echo "$accuracy" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('reconciled', 0))" 2>/dev/null || echo "0")
    actual_price=$(echo "$accuracy" | python3 -c "import sys, json; d=json.load(sys.stdin); o=d.get('recent_outcomes', [{}])[0] if d.get('recent_outcomes') else {}; print(o.get('actual_price', 0))" 2>/dev/null || echo "0")
    
    # 4. Check predictions
    echo "▸ Checking predictions..."
    predictions=$(curl -s "$BASE_URL/api/v3/predictions/latest?limit=1" 2>/dev/null)
    stage5=$(echo "$predictions" | python3 -c "import sys, json; d=json.load(sys.stdin); p=d.get('predictions', [{}])[0]; print('true' if p.get('stage5_ok') else 'false')" 2>/dev/null || echo "false")
    stage6=$(echo "$predictions" | python3 -c "import sys, json; d=json.load(sys.stdin); p=d.get('predictions', [{}])[0]; print('true' if p.get('stage6_ok') else 'false')" 2>/dev/null || echo "false")
    
    # 5. Check alerts
    echo "▸ Checking alerts..."
    alerts=$(curl -s "$BASE_URL/api/recent_alerts" 2>/dev/null)
    alert_count=$(echo "$alerts" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('count', 0))" 2>/dev/null || echo "0")
    
    echo ""
    echo "RESULTS:"
    echo "  Reconciled (this run): $reconciled"
    echo "  Total Outcomes: $total_outcomes"
    echo "  Sample actual_price: $actual_price"
    echo "  Stage5 OK: $stage5"
    echo "  Stage6 OK: $stage6"
    echo "  Alerts Sent: $alert_count"
    
    # Check success conditions
    if [ "$actual_price" != "0" ] && [ "$actual_price" != "0.0" ]; then
        echo ""
        echo "✅ SUCCESS: Outcomes have real prices!"
        echo "   → Polygon integration is working"
        
        if [ "$stage5" = "true" ]; then
            echo "✅ SUCCESS: Stage5 gates passing!"
            echo "   → Calibration accumulated 30+ samples"
            
            if [ "$alert_count" -gt "0" ]; then
                echo "✅ SUCCESS: Telegram alerts sending!"
                echo "   → ALL BLOCKERS RESOLVED"
                echo ""
                echo "════════════════════════════════════════════════════════════════"
                echo "  🚀 GHOST PROTOCOL IS PRODUCTION-COMPLETE"
                echo "════════════════════════════════════════════════════════════════"
                exit 0
            fi
        fi
    fi
    
    echo ""
    echo "⏳ Waiting 30s for next check..."
    echo ""
    sleep 30
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ⏰ Monitor timeout (10 minutes elapsed)"
echo "  Check status manually: $BASE_URL/api/v3/accuracy/dashboard"
echo "════════════════════════════════════════════════════════════════"
