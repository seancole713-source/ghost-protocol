#!/usr/bin/env python3
"""
Portfolio Migration Script: WOLF → NVDA
Preserves cost basis, AI memory, and trading history.
"""

import argparse
import json
import os
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def backup_databases(backup_dir: Path):
    """Create backups of all databases before migration."""
    print("📦 Creating backups...")
    backup_dir.mkdir(parents=True, exist_ok=True)

    files_to_backup = [
        "wolf_state.db",
        "ai_memory.db",
        "watchlist.db",
        "ghost.duckdb",
        "data/forecast_WOLF.json",
        "ghost_state.json",
    ]

    for file in files_to_backup:
        if os.path.exists(file):
            dest = backup_dir / Path(file).name
            shutil.copy2(file, dest)
            print(f"  ✅ Backed up {file} → {dest}")

    print(f"✅ Backups created in {backup_dir}\n")


def migrate_portfolio(dry_run: bool = True):
    """Migrate portfolio from WOLF to NVDA."""
    print(f"{'🔍 DRY RUN' if dry_run else '🚀 EXECUTING'}: Portfolio Migration\n")

    # Constants
    OLD_SYMBOL = "WOLF"
    NEW_SYMBOL = "NVDA"

    # Migration settings
    # You can adjust these based on your migration strategy
    migration_config = {
        "preserve_cost_basis": True,  # Keep original WOLF cost basis
        "adjust_quantity": True,  # Adjust quantity based on price ratio
        "migrate_history": True,  # Migrate AI memory records
        "create_forecast": True,  # Create new NVDA forecast
    }

    results = {
        "positions_migrated": 0,
        "ai_records_updated": 0,
        "forecasts_created": 0,
        "errors": [],
    }

    # 1. Get current WOLF position
    try:
        conn = sqlite3.connect("wolf_state.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT qty, avg_cost, last_updated
            FROM positions
            WHERE symbol = ?
        """,
            (OLD_SYMBOL,),
        )

        wolf_position = cursor.fetchone()

        if not wolf_position:
            print(f"⚠️  No {OLD_SYMBOL} position found in database")
            return results

        wolf_qty, wolf_avg_cost, wolf_updated = wolf_position
        print(f"📊 Current {OLD_SYMBOL} Position:")
        print(f"   Quantity: {wolf_qty}")
        print(f"   Avg Cost: ${wolf_avg_cost:.2f}")
        print(f"   Last Updated: {wolf_updated}\n")

        # 2. Calculate NVDA equivalent
        # Get current prices
        import yfinance as yf

        wolf_ticker = yf.Ticker(OLD_SYMBOL)
        nvda_ticker = yf.Ticker(NEW_SYMBOL)

        try:
            wolf_current = wolf_ticker.info.get("previousClose") or wolf_ticker.info.get(
                "regularMarketPrice", 24.69
            )
        except Exception:
            wolf_current = 24.69  # Fallback from health report

        try:
            nvda_current = nvda_ticker.info.get("previousClose") or nvda_ticker.info.get(
                "regularMarketPrice"
            )
        except Exception:
            nvda_current = 130.0  # Approximate NVDA price

        print("💵 Current Prices:")
        print(f"   {OLD_SYMBOL}: ${wolf_current:.2f}")
        print(f"   {NEW_SYMBOL}: ${nvda_current:.2f}\n")

        # Calculate NVDA quantity to preserve portfolio value
        wolf_market_value = wolf_qty * wolf_current
        nvda_qty = wolf_market_value / nvda_current

        # For cost basis, you can either:
        # A) Use NVDA current price (no unrealized P&L)
        # B) Scale WOLF cost basis to maintain P&L ratio
        if migration_config["preserve_cost_basis"]:
            # Option B: Preserve P&L ratio
            wolf_pnl_pct = (wolf_current - wolf_avg_cost) / wolf_avg_cost
            nvda_avg_cost = nvda_current / (1 + wolf_pnl_pct)
        else:
            # Option A: Start fresh at current price
            nvda_avg_cost = nvda_current

        print("🔄 Migration Plan:")
        print(f"   {OLD_SYMBOL}: {wolf_qty} shares @ ${wolf_avg_cost:.2f}")
        print(f"   → {NEW_SYMBOL}: {nvda_qty:.4f} shares @ ${nvda_avg_cost:.2f}")
        print(f"   Market Value: ${wolf_market_value:.2f} (preserved)")
        print(
            f"   P&L %: {((wolf_current - wolf_avg_cost) / wolf_avg_cost * 100):.2f}% → "
            f"{((nvda_current - nvda_avg_cost) / nvda_avg_cost * 100):.2f}%\n"
        )

        if dry_run:
            print("🔍 DRY RUN - No changes made\n")
            return results

        # 3. Execute migration
        print("🚀 Executing migration...\n")

        # Update position in database
        cursor.execute(
            """
            UPDATE positions
            SET symbol = ?,
                qty = ?,
                avg_cost = ?,
                last_updated = ?
            WHERE symbol = ?
        """,
            (NEW_SYMBOL, nvda_qty, nvda_avg_cost, int(datetime.now(UTC).timestamp()), OLD_SYMBOL),
        )

        results["positions_migrated"] = cursor.rowcount
        conn.commit()
        print(f"  ✅ Updated position: {OLD_SYMBOL} → {NEW_SYMBOL}")

        # 4. Update AI memory records (optional - for historical continuity)
        if migration_config["migrate_history"]:
            ai_conn = sqlite3.connect("ai_memory.db")
            ai_cursor = ai_conn.cursor()

            # Add migration note instead of changing history
            ai_cursor.execute(
                """
                INSERT INTO decisions (
                    ts, symbol, action, confidence, reasoning,
                    features, params, tag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    int(datetime.now(UTC).timestamp()),
                    NEW_SYMBOL,
                    "MIGRATE",
                    100,
                    f"Portfolio migrated from {OLD_SYMBOL} to {NEW_SYMBOL}. "
                    f"Qty: {wolf_qty} → {nvda_qty:.4f}, Avg Cost: ${wolf_avg_cost:.2f} → ${nvda_avg_cost:.2f}",
                    json.dumps(
                        {"old_symbol": OLD_SYMBOL, "old_qty": wolf_qty, "old_avg": wolf_avg_cost}
                    ),
                    json.dumps({"migration_date": datetime.now(UTC).isoformat()}),
                    "portfolio_migration",
                ),
            )

            ai_conn.commit()
            ai_conn.close()
            print("  ✅ Added migration record to AI memory")
            results["ai_records_updated"] = 1

        # 5. Create new forecast file
        if migration_config["create_forecast"]:
            forecast_data = {
                "symbol": NEW_SYMBOL,
                "generated_at": datetime.now(UTC).isoformat(),
                "current_price": nvda_current,
                "forecast_48h": [],
                "confidence": 0.65,
                "note": f"Initial forecast after migration from {OLD_SYMBOL}",
            }

            with open(f"data/forecast_{NEW_SYMBOL}.json", "w") as f:
                json.dump(forecast_data, f, indent=2)

            print(f"  ✅ Created forecast file for {NEW_SYMBOL}")
            results["forecasts_created"] = 1

        conn.close()
        print("\n✅ Migration complete!")

    except Exception as e:
        error_msg = f"Migration error: {e}"
        print(f"❌ {error_msg}")
        results["errors"].append(error_msg)

    return results


def update_environment_variables():
    """Print instructions for updating environment variables."""
    print("\n" + "=" * 60)
    print("📝 MANUAL STEPS REQUIRED")
    print("=" * 60)
    print("\n1. Update Railway Environment Variables:")
    print("   railway variables set DEFAULT_SYMBOL=NVDA")
    print("   railway variables set FOCUS_WOLF_ONLY=0")
    print()
    print("2. Update wolf_app.py (or set env var):")
    print('   WOLF = os.getenv("DEFAULT_SYMBOL", "NVDA")')
    print()
    print("3. Restart the server:")
    print("   railway restart")
    print()
    print("4. Update UI/docs references from WOLF to NVDA")
    print()


def main():
    parser = argparse.ArgumentParser(description="Migrate GHOST portfolio from WOLF to NVDA")
    parser.add_argument("--backup", action="store_true", help="Create backup before migration")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    parser.add_argument("--execute", action="store_true", help="Execute the migration")

    args = parser.parse_args()

    if not (args.dry_run or args.execute):
        print("❌ Please specify --dry-run or --execute")
        parser.print_help()
        return

    print("=" * 60)
    print("🔄 GHOST PORTFOLIO MIGRATION: WOLF → NVDA")
    print("=" * 60)
    print()

    # Create backup
    if args.backup:
        backup_dir = Path(f"backups/migration_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}")
        backup_databases(backup_dir)

    # Run migration
    results = migrate_portfolio(dry_run=args.dry_run)

    # Print results
    print("\n" + "=" * 60)
    print("📊 MIGRATION RESULTS")
    print("=" * 60)
    print(f"Positions migrated: {results['positions_migrated']}")
    print(f"AI records updated: {results['ai_records_updated']}")
    print(f"Forecasts created: {results['forecasts_created']}")
    if results["errors"]:
        print(f"\n❌ Errors: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"   - {error}")

    if args.execute and not results["errors"]:
        update_environment_variables()


if __name__ == "__main__":
    main()
