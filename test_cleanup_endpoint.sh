#!/bin/bash
# Test the cleanup endpoint

echo "🔍 Testing admin cleanup endpoint..."
echo ""

# Step 1: Dry run first (see what would be expired without making changes)
echo "1️⃣  DRY RUN - See what would be expired:"
curl -s -X POST "https://ghost-protocol-production.up.railway.app/api/v3/paper/admin/expire-old-pending?cutoff_date=2026-01-14&dry_run=true" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('ok'):
        print(f\"✅ Would expire: {data.get('would_expire', 0):,} trades\")
        print(f\"   Oldest: {data.get('oldest_trade', 'N/A')}\")
        print(f\"   Newest: {data.get('newest_trade', 'N/A')}\")
        print()
        print('Current outcome counts:')
        for outcome, count in data.get('current_outcome_counts', {}).items():
            print(f\"   {outcome}: {count:,}\")
    else:
        print(f\"❌ Error: {data.get('error', 'Unknown error')}\")
except Exception as e:
    print(f\"❌ Failed to parse response: {e}\")
"

echo ""
echo ""
echo "2️⃣  Ready to actually expire old trades?"
echo "   This will mark ~26K pending trades as EXPIRED."
echo ""
read -p "   Continue? (yes/no): " confirm

if [ "$confirm" = "yes" ]; then
    echo ""
    echo "🧹 Running cleanup (for real)..."
    curl -s -X POST "https://ghost-protocol-production.up.railway.app/api/v3/paper/admin/expire-old-pending?cutoff_date=2026-01-14" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('ok'):
        print(f\"✅ {data.get('message', 'Success')}\")
        print(f\"   Expired: {data.get('expired_count', 0):,} trades\")
        print()
        print('Updated outcome counts:')
        for outcome, count in data.get('outcome_counts', {}).items():
            print(f\"   {outcome}: {count:,}\")
    else:
        print(f\"❌ Error: {data.get('error', 'Unknown error')}\")
except Exception as e:
    print(f\"❌ Failed to parse response: {e}\")
    print('Raw response:', file=sys.stderr)
    sys.stdin.seek(0)
    print(sys.stdin.read(), file=sys.stderr)
"
else
    echo "❌ Cancelled - no changes made"
fi

echo ""
echo "3️⃣  Checking updated stats..."
bash /workspaces/ghost-protocol/check_pending_trades.sh
