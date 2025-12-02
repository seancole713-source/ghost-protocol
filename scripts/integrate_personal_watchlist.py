#!/usr/bin/env python3
"""
Ghost Protocol Personal Watchlist - Integration Script
=======================================================

Automates integration of personal watchlist module into wolf_app.py.

What this does:
1. Adds API endpoint imports
2. Mounts FastAPI router
3. Adds scheduler startup/shutdown hooks
4. Verifies database tables exist
5. Seeds default watchlist (optional)

Usage:
    python3 scripts/integrate_personal_watchlist.py
    python3 scripts/integrate_personal_watchlist.py --seed-default
"""

import argparse
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)


def verify_database_tables():
    """Verify personal watchlist tables exist in Postgres."""
    LOGGER.info("🔍 Verifying database tables...")

    try:
        from core.db_engine import execute_query

        # Check if ghost_watchlist_items exists
        result = execute_query(
            """
            SELECT COUNT(*) as cnt
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'ghost_watchlist_items'
            """,
            fetch="one",
        )

        if result and result[0] > 0:
            LOGGER.info("✅ ghost_watchlist_items table exists")
            return True
        else:
            LOGGER.error("❌ ghost_watchlist_items table NOT found")
            LOGGER.error("⚠️  Run migration first: psql $DATABASE_URL -f migrations/001_personal_watchlist.sql")
            return False

    except Exception as e:
        LOGGER.error(f"❌ Database verification failed: {e}")
        LOGGER.error("⚠️  Make sure DATABASE_URL is set and Postgres is accessible")
        return False


def seed_default_watchlist():
    """Seed default watchlist with common symbols."""
    LOGGER.info("🌱 Seeding default watchlist...")

    try:
        from core.personal_watchlist import get_personal_watchlist_manager

        pwm = get_personal_watchlist_manager()

        default_symbols = [
            # Top crypto
            ("BTC", "crypto", "Bitcoin - flagship crypto asset"),
            ("ETH", "crypto", "Ethereum - smart contract platform"),
            ("SOL", "crypto", "Solana - high-speed blockchain"),
            # Top stocks
            ("AAPL", "stock", "Apple Inc. - mega cap tech"),
            ("MSFT", "stock", "Microsoft - cloud + software"),
            ("NVDA", "stock", "NVIDIA - AI chips"),
            ("TSLA", "stock", "Tesla - EV and energy"),
        ]

        added_count = 0
        for symbol, asset_type, notes in default_symbols:
            result = pwm.add_symbol(symbol=symbol, asset_type=asset_type, owns_position=False, notes=notes, priority=2)

            if result.get("ok"):
                added_count += 1
                LOGGER.info(f"  ✅ {symbol} ({asset_type})")
            else:
                LOGGER.warning(f"  ⚠️  {symbol} already exists or failed: {result.get('error')}")

        LOGGER.info(f"✅ Seeded {added_count}/{len(default_symbols)} symbols")
        return True

    except Exception as e:
        LOGGER.error(f"❌ Seed failed: {e}")
        return False


def check_wolf_app_integration():
    """Check if wolf_app.py already has personal watchlist integration."""
    LOGGER.info("🔍 Checking wolf_app.py integration status...")

    wolf_app_path = "wolf_app.py"

    if not os.path.exists(wolf_app_path):
        LOGGER.error(f"❌ {wolf_app_path} not found")
        return False

    with open(wolf_app_path, "r") as f:
        content = f.read()

    has_router_import = "personal_watchlist_endpoints" in content
    has_scheduler_import = "watchlist_prediction_scheduler" in content
    has_router_mount = "APP.include_router(watchlist_router)" in content or "include_router(personal_watchlist_endpoints" in content

    if has_router_import and has_scheduler_import and has_router_mount:
        LOGGER.info("✅ wolf_app.py already integrated")
        return True
    elif has_router_import or has_scheduler_import or has_router_mount:
        LOGGER.warning("⚠️  wolf_app.py partially integrated (some imports missing)")
        return False
    else:
        LOGGER.info("📝 wolf_app.py not yet integrated")
        return False


def add_wolf_app_integration():
    """Add personal watchlist integration to wolf_app.py (manual guide)."""
    LOGGER.info("📝 Integration guide for wolf_app.py...")

    integration_code = """
# ============================================================================
# PERSONAL WATCHLIST INTEGRATION
# ============================================================================

# Add these imports near top of file (after other imports):
from api.personal_watchlist_endpoints import router as watchlist_router
from core.watchlist_prediction_scheduler import (
    start_watchlist_scheduler,
    stop_watchlist_scheduler,
)

# Mount router (after other APP.include_router() calls):
APP.include_router(watchlist_router)

# Add startup hook for scheduler (in or after main() function):
@APP.on_event("startup")
async def startup_watchlist_scheduler():
    start_watchlist_scheduler()

# Add shutdown hook for scheduler:
atexit.register(stop_watchlist_scheduler)
"""

    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("📋 ADD THIS TO wolf_app.py:")
    LOGGER.info("=" * 70)
    print(integration_code)
    LOGGER.info("=" * 70)
    LOGGER.info("\n⚠️  Manual integration required (auto-patching too risky)")
    LOGGER.info("✅ Copy the code above and add it to wolf_app.py")


def verify_ui_integration():
    """Check if Cockpit UI has personal watchlist JavaScript."""
    LOGGER.info("🔍 Checking Cockpit UI integration...")

    cockpit_html_path = "templates/cockpit_v3.html"

    if not os.path.exists(cockpit_html_path):
        LOGGER.warning(f"⚠️  {cockpit_html_path} not found (may be in different location)")
        return False

    with open(cockpit_html_path, "r") as f:
        content = f.read()

    if "personal_watchlist_ui.js" in content:
        LOGGER.info("✅ Cockpit UI already has personal_watchlist_ui.js")
        return True
    else:
        LOGGER.info("📝 Cockpit UI not yet integrated")
        LOGGER.info("\n" + "=" * 70)
        LOGGER.info("📋 ADD THIS TO templates/cockpit_v3.html (before </body>):")
        LOGGER.info("=" * 70)
        print('    <!-- Personal Watchlist UI Module -->')
        print('    <script src="/static/personal_watchlist_ui.js"></script>')
        LOGGER.info("=" * 70)
        return False


def show_env_vars():
    """Show required environment variables."""
    LOGGER.info("\n📝 Environment Variables (add to Railway/production):")
    LOGGER.info("=" * 70)

    env_vars = {
        # Scheduler
        "WATCHLIST_SCHEDULER_ENABLED": "1",
        "WATCHLIST_OPEN_HOUR": "9",
        "WATCHLIST_CLOSE_HOUR": "16",
        "WATCHLIST_BIG_MOVE_CHECK_MINUTES": "15",
        "WATCHLIST_BIG_MOVE_THRESHOLD_PCT": "5.0",
        # Alerts
        "WATCHLIST_ALERTS_ENABLED": "1",
        "WATCHLIST_ALERTS_INCLUDE_OPEN_CLOSE": "1",
        "WATCHLIST_ALERTS_INCLUDE_BIG_MOVES": "1",
        "WATCHLIST_ALERT_COOLDOWN_HOURS": "4",
        "WATCHLIST_ALERT_GLOBAL_LIMIT_PER_HOUR": "5",
    }

    for key, value in env_vars.items():
        current = os.getenv(key, "")
        status = "✅ SET" if current else "⚠️  NOT SET"
        print(f"{status}  {key}={value}")

    LOGGER.info("=" * 70)


def main():
    """Main integration script."""
    parser = argparse.ArgumentParser(description="Integrate Personal Watchlist module into Ghost Protocol")
    parser.add_argument("--seed-default", action="store_true", help="Seed default watchlist symbols")
    parser.add_argument("--skip-db-check", action="store_true", help="Skip database verification")
    args = parser.parse_args()

    LOGGER.info("🚀 Ghost Protocol Personal Watchlist Integration")
    LOGGER.info("=" * 70)

    # Step 1: Verify database tables
    if not args.skip_db_check:
        if not verify_database_tables():
            LOGGER.error("❌ Integration aborted: database tables not found")
            LOGGER.error("⚠️  Run migration first:")
            LOGGER.error("   psql $DATABASE_URL -f migrations/001_personal_watchlist.sql")
            sys.exit(1)
    else:
        LOGGER.warning("⚠️  Skipping database verification (--skip-db-check)")

    # Step 2: Check wolf_app.py integration
    wolf_app_integrated = check_wolf_app_integration()
    if not wolf_app_integrated:
        add_wolf_app_integration()

    # Step 3: Check UI integration
    verify_ui_integration()

    # Step 4: Seed default watchlist (optional)
    if args.seed_default:
        seed_default_watchlist()

    # Step 5: Show environment variables
    show_env_vars()

    # Summary
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("📊 INTEGRATION STATUS")
    LOGGER.info("=" * 70)
    LOGGER.info(f"✅ Database tables:    {'OK' if not args.skip_db_check else 'SKIPPED'}")
    LOGGER.info(f"{'✅' if wolf_app_integrated else '⚠️'}  wolf_app.py:       {'Integrated' if wolf_app_integrated else 'Manual steps required'}")
    LOGGER.info("⚠️  Cockpit UI:         Manual steps required")
    LOGGER.info("⚠️  Environment vars:   Manual configuration required")
    LOGGER.info("=" * 70)

    if not wolf_app_integrated:
        LOGGER.warning("\n⚠️  MANUAL INTEGRATION REQUIRED")
        LOGGER.warning("Follow the code snippets printed above to complete integration")

    LOGGER.info("\n✅ Integration script complete!")
    LOGGER.info("📖 Full documentation: PERSONAL_WATCHLIST_README.md")


if __name__ == "__main__":
    main()
