#!/bin/bash
# Emergency fix: Disable personal watchlist scheduler if it's causing crashes

echo "🚨 EMERGENCY FIX: Disabling personal watchlist scheduler"
echo ""
echo "This will:"
echo "  1. Comment out watchlist scheduler in wolf_app.py"
echo "  2. Prevent SQL errors from blocking startup"
echo "  3. Allow other endpoints to work normally"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted"
    exit 1
fi

cd "$(dirname "$0")/.." || exit 1

# Backup wolf_app.py
cp wolf_app.py wolf_app.py.backup.$(date +%s)

# Comment out the watchlist scheduler start
sed -i.tmp '/\[GHOST STARTUP\].*Personal watchlist scheduler/,/^$/s/^/# DISABLED: /' wolf_app.py

# Check if successful
if grep -q "# DISABLED:.*Personal watchlist scheduler" wolf_app.py; then
    echo "✅ Watchlist scheduler disabled"
    echo ""
    echo "Next steps:"
    echo "  1. git add wolf_app.py"
    echo "  2. git commit -m 'Emergency: disable watchlist scheduler'"
    echo "  3. git push origin main"
    echo ""
    echo "To re-enable later:"
    echo "  git revert HEAD"
else
    echo "❌ Failed to disable scheduler"
    echo "Manual fix required"
    exit 1
fi
