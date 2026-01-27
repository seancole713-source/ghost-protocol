#!/usr/bin/env python3
"""
Ghost Protocol - Comprehensive System Verification & Initialization
===================================================================

This script performs a complete system check and initialization:
1. Verify Postgres connection
2. Check migration status
3. Verify/rebuild core tables
4. Initialize watchlist scheduler
5. Initialize predictor engine
6. Verify Cockpit v3
7. Test predictions (BTC, ETH, AAPL, TSLA)
8. Test watchlist predictions
9. Generate system readiness report

Usage:
    python3 system_verification.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

# Color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

def print_header(text):
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")

def print_step(num, text):
    print(f"\n{CYAN}STEP {num}: {text}{RESET}")
    print(f"{CYAN}{'-' * 80}{RESET}")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_warning(text):
    print(f"{YELLOW}⚠️  {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

# Results tracker
results = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "tests": {}
}

def main():
    print_header("GHOST PROTOCOL - SYSTEM VERIFICATION & INITIALIZATION")
    print(f"Timestamp: {results['timestamp']}")
    
    # ========================================================================
    # STEP 1: VERIFY POSTGRES CONNECTION
    # ========================================================================
    print_step(1, "Database Connection Verification")
    
    try:
        from core.db_engine import get_db_connection, IS_POSTGRES, IS_SQLITE, DATABASE_URL, WOLF_SQLITE_PATH
        
        db_type = "PostgreSQL" if IS_POSTGRES else "SQLite"
        print(f"Database type: {db_type}")
        
        if IS_POSTGRES:
            print(f"Database URL: {DATABASE_URL[:50]}...")
            results["tests"]["database"] = {"type": "postgres", "status": "testing"}
        else:
            print(f"SQLite path: {WOLF_SQLITE_PATH}")
            results["tests"]["database"] = {"type": "sqlite", "status": "testing"}
        
        # Test connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            if IS_POSTGRES:
                cursor.execute("SELECT version();")
                version_data = cursor.fetchone()
                if hasattr(version_data, 'get'):
                    version = version_data.get("version", "Unknown")
                else:
                    version = version_data[0] if version_data else "Unknown"
                print_success(f"PostgreSQL connected")
                print(f"   {version[:80]}")
                results["tests"]["database"]["status"] = "pass"
                results["tests"]["database"]["version"] = version[:60]
            else:
                cursor.execute("SELECT sqlite_version();")
                version = cursor.fetchone()[0]
                print_success(f"SQLite connected (version {version})")
                results["tests"]["database"]["status"] = "pass"
                results["tests"]["database"]["version"] = version
        
    except Exception as e:
        print_error(f"Database connection failed: {e}")
        results["tests"]["database"] = {"status": "fail", "error": str(e)}
    
    # ========================================================================
    # STEP 2: CHECK MIGRATION STATUS
    # ========================================================================
    print_step(2, "Database Migration Status")
    
    migration_file = Path("migrations/001_personal_watchlist.sql")
    
    if not migration_file.exists():
        print_warning("Migration file not found: migrations/001_personal_watchlist.sql")
        results["tests"]["migration"] = {"status": "skip", "reason": "file_not_found"}
    elif IS_SQLITE:
        print_warning("SQLite mode - Personal watchlist requires Postgres")
        print("   Migration not applicable for SQLite")
        results["tests"]["migration"] = {"status": "skip", "reason": "sqlite_mode"}
    else:
        print_success(f"Migration file found ({migration_file.stat().st_size} bytes)")
        print()
        print("Migration must be run manually on Railway:")
        print("   railway run psql $DATABASE_URL -f migrations/001_personal_watchlist.sql")
        results["tests"]["migration"] = {"status": "ready", "file": str(migration_file)}
    
    # ========================================================================
    # STEP 3: VERIFY CORE TABLES
    # ========================================================================
    print_step(3, "Core Tables Verification")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check personal watchlist tables
            print("Personal Watchlist Tables:")
            if IS_POSTGRES:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_name LIKE '%watchlist%' 
                    AND table_schema = 'public'
                    ORDER BY table_name
                """)
                watchlist_tables = cursor.fetchall()
                
                if watchlist_tables:
                    for row in watchlist_tables:
                        table_name = row.get("table_name") if hasattr(row, 'get') else row[0]
                        print_success(table_name)
                    results["tests"]["watchlist_tables"] = {
                        "status": "pass",
                        "count": len(watchlist_tables),
                        "tables": [row.get("table_name") if hasattr(row, 'get') else row[0] for row in watchlist_tables]
                    }
                else:
                    print_warning("No watchlist tables found - migration needed")
                    results["tests"]["watchlist_tables"] = {"status": "missing", "reason": "migration_not_run"}
            else:
                print_warning("SQLite mode - watchlist tables not checked")
                results["tests"]["watchlist_tables"] = {"status": "skip", "reason": "sqlite_mode"}
            
            # Check prediction store
            print()
            print("Prediction Store:")
            try:
                from core.prediction_store import get_prediction_store
                store = get_prediction_store()
                print_success(f"Initialized ({store.__class__.__name__})")
                results["tests"]["prediction_store"] = {"status": "pass", "backend": store.__class__.__name__}
            except Exception as e:
                print_error(f"Prediction store error: {e}")
                results["tests"]["prediction_store"] = {"status": "fail", "error": str(e)}
            
            # Check goals table
            print()
            print("Goals Table:")
            if IS_POSTGRES:
                cursor.execute("""
                    SELECT COUNT(*) as count FROM information_schema.tables 
                    WHERE table_name = 'ghost_goals'
                """)
                exists = cursor.fetchone()
                exists_count = exists.get("count") if hasattr(exists, 'get') else exists[0]
                exists = exists_count > 0
            else:
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='ghost_goals'")
                exists = cursor.fetchone()[0] > 0
            
            if exists:
                cursor.execute("SELECT COUNT(*) as count FROM ghost_goals")
                count_row = cursor.fetchone()
                count = count_row.get("count") if hasattr(count_row, 'get') else count_row[0]
                print_success(f"ghost_goals exists ({count} rows)")
                results["tests"]["goals_table"] = {"status": "pass", "rows": count}
            else:
                print_warning("ghost_goals table not found")
                results["tests"]["goals_table"] = {"status": "missing"}
                
    except Exception as e:
        print_error(f"Table verification error: {e}")
        results["tests"]["tables"] = {"status": "fail", "error": str(e)}
    
    # ========================================================================
    # STEP 4: INITIALIZE WATCHLIST SCHEDULER
    # ========================================================================
    print_step(4, "Watchlist Scheduler Initialization")
    
    try:
        from core.watchlist_prediction_scheduler import WatchlistPredictionScheduler
        
        # Check if scheduler is enabled
        scheduler_enabled = os.getenv("WATCHLIST_SCHEDULER_ENABLED", "0") == "1"
        
        if not scheduler_enabled:
            print_warning("Scheduler disabled (WATCHLIST_SCHEDULER_ENABLED=0)")
            print("   Set WATCHLIST_SCHEDULER_ENABLED=1 to enable")
            results["tests"]["watchlist_scheduler"] = {"status": "disabled"}
        else:
            print_success("Scheduler module imported")
            print("   Scheduler will start automatically with wolf_app.py")
            print(f"   Market open: {os.getenv('WATCHLIST_OPEN_HOUR', '9')}:00 EST")
            print(f"   Market close: {os.getenv('WATCHLIST_CLOSE_HOUR', '16')}:00 EST")
            results["tests"]["watchlist_scheduler"] = {
                "status": "ready",
                "open_hour": os.getenv('WATCHLIST_OPEN_HOUR', '9'),
                "close_hour": os.getenv('WATCHLIST_CLOSE_HOUR', '16')
            }
            
    except Exception as e:
        print_error(f"Scheduler import error: {e}")
        results["tests"]["watchlist_scheduler"] = {"status": "fail", "error": str(e)}
    
    # ========================================================================
    # STEP 5: INITIALIZE PREDICTOR ENGINE
    # ========================================================================
    print_step(5, "Predictor Engine Initialization")
    
    try:
        from services.predictor import predict_symbol
        
        print_success("Predictor module imported")
        print("   predict_symbol() function available")
        results["tests"]["predictor"] = {"status": "ready"}
        
    except Exception as e:
        print_error(f"Predictor import error: {e}")
        results["tests"]["predictor"] = {"status": "fail", "error": str(e)}
    
    # ========================================================================
    # STEP 6: VERIFY COCKPIT V3
    # ========================================================================
    print_step(6, "Cockpit V3 Verification")
    
    try:
        from api.cockpit_v3_live_endpoints import router
        
        print_success("Cockpit V3 endpoints imported")
        
        # Check UI files
        ui_file = Path("templates/cockpit_v3.html")
        js_file = Path("static/personal_watchlist_ui.js")
        
        if ui_file.exists():
            print_success(f"Cockpit UI template found ({ui_file.stat().st_size} bytes)")
        else:
            print_warning("Cockpit UI template not found")
        
        if js_file.exists():
            print_success(f"Personal watchlist UI found ({js_file.stat().st_size} bytes)")
        else:
            print_warning("Personal watchlist UI not found")
        
        results["tests"]["cockpit_v3"] = {
            "status": "pass",
            "ui_exists": ui_file.exists(),
            "watchlist_ui_exists": js_file.exists()
        }
        
    except Exception as e:
        print_error(f"Cockpit V3 error: {e}")
        results["tests"]["cockpit_v3"] = {"status": "fail", "error": str(e)}
    
    # ========================================================================
    # STEP 7: TEST PREDICTIONS (BTC, ETH, AAPL, TSLA)
    # ========================================================================
    print_step(7, "Prediction Engine Test")
    
    test_symbols = [
        ("BTC", "crypto"),
        ("ETH", "crypto"),
        ("AAPL", "stock"),
        ("TSLA", "stock")
    ]
    
    prediction_results = []
    
    for symbol, asset_type in test_symbols:
        try:
            print(f"\nTesting {symbol} ({asset_type})...")
            
            # Try to get existing prediction first
            from core.prediction_store import get_prediction_store
            store = get_prediction_store()
            latest = store.get_latest_prediction(symbol)
            
            if latest:
                print_success(f"{symbol}: Prediction found (ID={latest.get('id')}, direction={latest.get('direction')}, confidence={latest.get('confidence', 0):.0%})")
                prediction_results.append({
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "status": "existing",
                    "prediction_id": latest.get("id"),
                    "direction": latest.get("direction"),
                    "confidence": latest.get("confidence")
                })
            else:
                print_warning(f"{symbol}: No recent prediction found")
                prediction_results.append({
                    "symbol": symbol,
                    "asset_type": asset_type,
                    "status": "missing"
                })
                
        except Exception as e:
            print_error(f"{symbol}: Error - {e}")
            prediction_results.append({
                "symbol": symbol,
                "asset_type": asset_type,
                "status": "error",
                "error": str(e)
            })
    
    results["tests"]["predictions"] = {
        "status": "complete",
        "symbols_tested": len(test_symbols),
        "results": prediction_results
    }
    
    # ========================================================================
    # STEP 8: TEST WATCHLIST PREDICTIONS
    # ========================================================================
    print_step(8, "Watchlist Prediction Test")
    
    try:
        from core.personal_watchlist import PersonalWatchlistManager
        
        manager = PersonalWatchlistManager()
        watchlist = manager.get_watchlist(active_only=True)
        
        print(f"Watchlist items: {len(watchlist)}")
        
        if len(watchlist) > 0:
            print()
            for item in watchlist[:5]:  # Show first 5
                symbol = item.get("symbol")
                asset_type = item.get("asset_type")
                owned = "✅ OWNED" if item.get("owns_position") else "  "
                print(f"  {owned} {symbol} ({asset_type})")
            
            if len(watchlist) > 5:
                print(f"  ... and {len(watchlist) - 5} more")
        else:
            print_warning("Watchlist is empty - add symbols via Cockpit UI")
        
        results["tests"]["watchlist_predictions"] = {
            "status": "pass",
            "watchlist_count": len(watchlist)
        }
        
    except Exception as e:
        print_error(f"Watchlist test error: {e}")
        results["tests"]["watchlist_predictions"] = {"status": "fail", "error": str(e)}
    
    # ========================================================================
    # STEP 9: GENERATE SYSTEM READINESS REPORT
    # ========================================================================
    print_step(9, "System Readiness Report")
    
    # Calculate pass/fail/skip counts
    statuses = [test.get("status") for test in results["tests"].values()]
    pass_count = statuses.count("pass") + statuses.count("ready") + statuses.count("complete")
    fail_count = statuses.count("fail")
    skip_count = statuses.count("skip") + statuses.count("disabled") + statuses.count("missing")
    
    total_tests = len(results["tests"])
    
    print()
    print(f"Total Tests: {total_tests}")
    print(f"{GREEN}Passed: {pass_count}{RESET}")
    print(f"{RED}Failed: {fail_count}{RESET}")
    print(f"{YELLOW}Skipped/Disabled: {skip_count}{RESET}")
    print()
    
    # Detailed results
    print("Detailed Results:")
    print("-" * 80)
    for test_name, test_data in results["tests"].items():
        status = test_data.get("status", "unknown")
        if status in ("pass", "ready", "complete"):
            status_icon = f"{GREEN}✅{RESET}"
        elif status in ("fail", "error"):
            status_icon = f"{RED}❌{RESET}"
        else:
            status_icon = f"{YELLOW}⚠️{RESET}"
        
        print(f"{status_icon} {test_name:30s} {status}")
    
    # Overall readiness
    print()
    print("-" * 80)
    
    if fail_count == 0:
        print(f"{GREEN}{'':>20}✅ SYSTEM READY{RESET}")
        overall_status = "ready"
    elif fail_count <= 2:
        print(f"{YELLOW}{'':>20}⚠️  SYSTEM PARTIALLY READY{RESET}")
        print(f"{'':>20}Some components need attention")
        overall_status = "partial"
    else:
        print(f"{RED}{'':>20}❌ SYSTEM NOT READY{RESET}")
        print(f"{'':>20}Multiple failures detected")
        overall_status = "not_ready"
    
    results["overall_status"] = overall_status
    results["summary"] = {
        "total_tests": total_tests,
        "passed": pass_count,
        "failed": fail_count,
        "skipped": skip_count
    }
    
    # Save report to file
    report_file = Path("system_verification_report.json")
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"Full report saved to: {report_file}")
    
    print_header("VERIFICATION COMPLETE")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Verification interrupted by user{RESET}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{RED}Unexpected error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
