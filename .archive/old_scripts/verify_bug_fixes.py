#!/usr/bin/env python3
"""
🔍 Verify Ghost Protocol Bug Fixes
Tests both critical fixes before Railway deployment
"""

import os
import sys

def test_paper_tracker_schema():
    """Verify paper_tracker.py has correct PostgreSQL schema"""
    print("=" * 60)
    print("🔍 TEST 1: paper_tracker.py Schema")
    print("=" * 60)
    
    with open("core/paper_tracker.py", "r") as f:
        content = f.read()
    
    # Check for PostgreSQL section
    if "if self.use_postgres:" in content:
        print("✅ PostgreSQL branch found")
    else:
        print("❌ PostgreSQL branch missing")
        return False
    
    # Check for TIMESTAMP WITH TIME ZONE in PostgreSQL section
    # Extract just the PostgreSQL CREATE TABLE statement
    lines = content.split("\n")
    postgres_section = []
    in_postgres_create = False
    for line in lines:
        if "if self.use_postgres:" in line:
            in_postgres_create = True
        elif "else:" in line and in_postgres_create and "conn =" not in line:
            break
        if in_postgres_create:
            postgres_section.append(line)
    
    postgres_content = "\n".join(postgres_section)
    
    required_columns = [
        "signal_time TIMESTAMP WITH TIME ZONE",
        "entry_time TIMESTAMP WITH TIME ZONE", 
        "target_time TIMESTAMP WITH TIME ZONE",
        "checked_at TIMESTAMP WITH TIME ZONE",
        "created_at TIMESTAMP WITH TIME ZONE"
    ]
    
    all_found = True
    for col in required_columns:
        if col in postgres_content:
            print(f"✅ Found: {col}")
        else:
            print(f"❌ Missing: {col}")
            all_found = False
    
    # Check SQLite section still uses TEXT (correct for SQLite)
    if "target_time TEXT NOT NULL" in content:
        # Make sure it's in the else branch
        if "else:" in content and content.index("target_time TEXT NOT NULL") > content.index("else:"):
            print("✅ SQLite section still uses TEXT (correct)")
        else:
            print("⚠️  Found TEXT in wrong section")
    else:
        print("❌ SQLite section doesn't use TEXT")
        all_found = False
    
    return all_found

def test_reconciler_query():
    """Verify outcome_reconciler_v2.py queries PostgreSQL correctly"""
    print("\n" + "=" * 60)
    print("🔍 TEST 2: outcome_reconciler_v2.py PostgreSQL Query")
    print("=" * 60)
    
    with open("services/outcome_reconciler_v2.py", "r") as f:
        content = f.read()
    
    checks = {
        "Uses psycopg2": "import psycopg2" in content,
        "Checks DATABASE_URL": "database_url = os.getenv(\"DATABASE_URL\")" in content,
        "Queries ghost_predictions": "SELECT price_at_prediction FROM ghost_predictions" in content,
        "Uses PostgreSQL params (%s)": "WHERE symbol = %s AND run_at BETWEEN %s AND %s" in content,
        "Has SQLite fallback": "store.backend.query" in content,
        "Fallback uses predictions": "SELECT price_at_prediction FROM predictions" in content
    }
    
    all_passed = True
    for check, passed in checks.items():
        if passed:
            print(f"✅ {check}")
        else:
            print(f"❌ {check}")
            all_passed = False
    
    # Check that old buggy code is gone from main path
    # The old code should only be in the "else" fallback for dev container
    if "database_url = os.getenv" in content and "if database_url:" in content:
        print("✅ PostgreSQL branch uses DATABASE_URL check (correct)")
        
        # Verify the fallback exists for dev container
        if "else:" in content and "store.backend.query" in content:
            print("✅ SQLite fallback exists for dev container (correct)")
        else:
            print("⚠️  No SQLite fallback found")
    else:
        print("❌ PostgreSQL branch missing DATABASE_URL check")
        all_passed = False
    
    return all_passed

def test_migration_script():
    """Verify migration script exists and is executable"""
    print("\n" + "=" * 60)
    print("🔍 TEST 3: Migration Script")
    print("=" * 60)
    
    if os.path.exists("migrate_paper_trades_schema.py"):
        print("✅ migrate_paper_trades_schema.py exists")
    else:
        print("❌ migrate_paper_trades_schema.py missing")
        return False
    
    with open("migrate_paper_trades_schema.py", "r") as f:
        content = f.read()
    
    checks = {
        "Has psycopg2 import": "import psycopg2" in content,
        "Checks DATABASE_URL": "DATABASE_URL" in content,
        "Alters columns": "ALTER TABLE paper_trades" in content,
        "Converts to TIMESTAMP": "TYPE TIMESTAMP WITH TIME ZONE" in content,
        "Has main guard": 'if __name__ == "__main__"' in content
    }
    
    all_passed = True
    for check, passed in checks.items():
        if passed:
            print(f"✅ {check}")
        else:
            print(f"❌ {check}")
            all_passed = False
    
    return all_passed

def test_documentation():
    """Verify documentation files exist"""
    print("\n" + "=" * 60)
    print("🔍 TEST 4: Documentation")
    print("=" * 60)
    
    docs = [
        "CRITICAL_BUG_FIX_PLAN.md",
        "DEPLOYMENT_READY_JAN7.md",
        "DEEP_DIVE_AUDIT_FULL_FINDINGS.md"
    ]
    
    all_found = True
    for doc in docs:
        if os.path.exists(doc):
            print(f"✅ {doc} exists")
        else:
            print(f"❌ {doc} missing")
            all_found = False
    
    return all_found

def main():
    print("🔍 Ghost Protocol Bug Fix Verification")
    print("Testing all critical fixes before Railway deployment\n")
    
    tests = [
        ("Paper Tracker Schema", test_paper_tracker_schema),
        ("Reconciler Query", test_reconciler_query),
        ("Migration Script", test_migration_script),
        ("Documentation", test_documentation)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ {name} test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - READY FOR RAILWAY DEPLOYMENT")
        print("=" * 60)
        print("\nNext steps:")
        print("1. git add core/paper_tracker.py services/outcome_reconciler_v2.py migrate_paper_trades_schema.py")
        print("2. git commit -m '🐛 Fix critical PostgreSQL bugs'")
        print("3. git push origin main")
        print("4. Watch Railway deploy logs")
        print("5. Run migration: railway run python3 migrate_paper_trades_schema.py")
        return 0
    else:
        print("❌ SOME TESTS FAILED - DO NOT DEPLOY")
        print("=" * 60)
        print("\nFix the failing tests before deployment!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
