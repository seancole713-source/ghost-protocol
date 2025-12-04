#!/bin/bash
echo "========================================="
echo "GHOST PROTOCOL PRODUCTION STATUS CHECK"
echo "========================================="
echo ""

echo "1. Health Endpoint (3 tests):"
for i in {1..3}; do
  response=$(curl -s --max-time 2 "https://ghost-protocol-production.up.railway.app/health")
  status=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"{d['status']} - Uptime: {d.get('uptime','N/A')}s\")" 2>/dev/null || echo "Error")
  echo "  Test $i: $status"
  sleep 0.5
done

echo ""
echo "2. Cockpit UI:"
cockpit=$(curl -s --max-time 3 "https://ghost-protocol-production.up.railway.app/cockpit" | head -1)
if [[ "$cockpit" == *"<!DOCTYPE"* ]]; then
  echo "  ✅ HTML being served"
else
  echo "  ❌ Not loading"
fi

echo ""
echo "3. XRP Tracker:"
xrp=$(curl -s --max-time 2 "https://ghost-protocol-production.up.railway.app/api/xrp/tracker" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"Price: {d.get('price','N/A')}, Signals: {len(d.get('signals',[]))}\")" 2>/dev/null || echo "Error")
echo "  $xrp"

echo ""
echo "4. Latest Production Logs (VIP Scanner):"
railway logs --tail=50 2>&1 | grep "VIP scan" | tail -3

echo ""
echo "5. Background Services Status:"
railway logs --tail=200 2>&1 | grep -E "(VIP.*STARTED|Pre-Market.*STARTED|Telegram.*initialized)" | tail -3

echo ""
echo "========================================="
