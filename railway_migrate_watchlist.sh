#!/bin/bash
# Railway Database Migration for Personal Watchlist
# Run this on Railway via: railway run bash railway_migrate_watchlist.sh

set -e

echo "=========================================="
echo "Personal Watchlist Database Migration"
echo "=========================================="
echo ""

if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL not set"
    echo "This script must run on Railway where DATABASE_URL is configured"
    exit 1
fi

echo "📋 Database URL: ${DATABASE_URL:0:30}..."
echo ""

# Check if tables already exist
echo "Checking if watchlist tables already exist..."
EXISTING=$(psql "$DATABASE_URL" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'ghost_watchlist_items'" 2>/dev/null || echo "0")

if [ "$EXISTING" != "0" ]; then
    echo "⚠️  WARNING: ghost_watchlist_items table already exists"
    echo ""
    read -p "Drop and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Dropping existing tables..."
        psql "$DATABASE_URL" << 'EOF'
DROP TABLE IF EXISTS watchlist_alerts_log CASCADE;
DROP TABLE IF EXISTS watchlist_price_snapshots CASCADE;
DROP TABLE IF EXISTS watchlist_prediction_tracking CASCADE;
DROP TABLE IF EXISTS ghost_watchlist_items CASCADE;
EOF
        echo "✅ Old tables dropped"
    else
        echo "❌ Migration cancelled"
        exit 1
    fi
fi

echo ""
echo "Running migration..."
psql "$DATABASE_URL" -f migrations/001_personal_watchlist.sql

echo ""
echo "=========================================="
echo "✅ Migration Complete!"
echo "=========================================="
echo ""

# Verify tables created
echo "Verifying tables..."
psql "$DATABASE_URL" << 'EOF'
SELECT 
    table_name,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count
FROM information_schema.tables t
WHERE table_name LIKE '%watchlist%'
ORDER BY table_name;
EOF

echo ""
echo "Checking seed data..."
psql "$DATABASE_URL" << 'EOF'
SELECT COUNT(*) as seed_symbols FROM ghost_watchlist_items WHERE active = TRUE;
EOF

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. Restart Railway service to load new code"
echo "2. Check logs for: '✅ Personal Watchlist endpoints registered'"
echo "3. Test API: curl https://YOUR-DOMAIN/api/v3/watchlist/user"
echo "4. Set environment variables (if not already set):"
echo "   - WATCHLIST_SCHEDULER_ENABLED=1"
echo "   - WATCHLIST_ALERTS_ENABLED=1"
echo ""
