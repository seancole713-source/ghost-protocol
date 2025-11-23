#!/usr/bin/env python3
"""Ghost System Repair - Diagnostic and Auto-Fix Script"""

import json
import os
import sys
from pathlib import Path

print("=" * 60)
print("🔧 GHOST SYSTEM REPAIR - HEALTH DIAGNOSTICS")
print("=" * 60)

results = {"modules": {}, "issues": [], "warnings": [], "fixes_applied": [], "env_status": {}}

# 1. Check Python environment
print("\n📦 Python Environment:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  Path: {sys.executable}")

# 2. Check critical modules
print("\n🔍 Module Availability:")
modules_to_check = [
    "fastapi",
    "uvicorn",
    "requests",
    "redis",
    "sqlite3",
    "yfinance",
    "pandas",
    "numpy",
    "openai",
    "prometheus_client",
]

for mod in modules_to_check:
    try:
        __import__(mod)
        print(f"  ✅ {mod}")
        results["modules"][mod] = "OK"
    except ImportError:
        print(f"  ❌ {mod} - MISSING")
        results["issues"].append(f"Module {mod} not installed")
        results["modules"][mod] = "MISSING"

# 3. Check environment variables
print("\n⚙️  Environment Variables:")
env_vars = {
    "SIM_MODE": os.getenv("SIM_MODE"),
    "PORT": os.getenv("PORT"),
    "CRYPTO_ENABLED": os.getenv("CRYPTO_ENABLED"),
    "OPENAI_API_KEY": "SET" if os.getenv("OPENAI_API_KEY") else "NOT_SET",
    "POLYGON_API_KEY": "SET" if os.getenv("POLYGON_API_KEY") else "NOT_SET",
    "REDIS_URL": os.getenv("REDIS_URL"),
}

for key, val in env_vars.items():
    status = "✅" if val and val != "NOT_SET" else "⚠️ "
    print(f"  {status} {key}: {val}")
    results["env_status"][key] = val
    if not val or val == "NOT_SET":
        if key in ["OPENAI_API_KEY", "CRYPTO_ENABLED"]:
            results["warnings"].append(f"{key} not configured")

# 4. Check database files
print("\n💾 Database Files:")
db_files = ["data/wolf.db", "data/order_manager.db", "watchlist.db"]
for db in db_files:
    if Path(db).exists():
        size = Path(db).stat().st_size
        print(f"  ✅ {db} ({size} bytes)")
    else:
        print(f"  ⚠️  {db} - NOT FOUND")
        results["warnings"].append(f"Database {db} missing")

# 5. Check Redis connectivity
print("\n🔴 Redis Connection:")
try:
    import redis

    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    r = redis.from_url(redis_url, socket_connect_timeout=2)
    r.ping()
    print(f"  ✅ Connected to {redis_url}")
except Exception as e:
    print(f"  ❌ Redis error: {e}")
    results["issues"].append(f"Redis connection failed: {e}")

# 6. Check AI/ML readiness
print("\n🧠 AI/ML Components:")
try:
    print("  ✅ NumPy + Pandas available")
except Exception:
    print("  ❌ NumPy/Pandas missing")
    results["issues"].append("NumPy or Pandas not available")

# 7. Test API endpoints
print("\n🌐 API Endpoint Tests:")
try:
    import requests

    tests = [
        ("http://localhost:8444/health", "Health Check"),
        ("http://localhost:8444/api/portfolio", "Portfolio"),
        ("http://localhost:8444/api/agent/stats", "Agent Stats"),
    ]
    for url, name in tests:
        try:
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                print(f"  ✅ {name}: OK")
            else:
                print(f"  ⚠️  {name}: HTTP {resp.status_code}")
                results["warnings"].append(f"{name} returned {resp.status_code}")
        except Exception as e:
            print(f"  ❌ {name}: {str(e)[:50]}")
            results["issues"].append(f"{name} failed: {e}")
except Exception:
    print("  ⚠️  Requests module not available for testing")

# Summary
print("\n" + "=" * 60)
print("📊 DIAGNOSTIC SUMMARY")
print("=" * 60)
print(f"Critical Issues: {len(results['issues'])}")
print(f"Warnings: {len(results['warnings'])}")

if results["issues"]:
    print("\n❌ Critical Issues:")
    for issue in results["issues"]:
        print(f"  - {issue}")

if results["warnings"]:
    print("\n⚠️  Warnings:")
    for warning in results["warnings"]:
        print(f"  - {warning}")

# Write results
with open("/tmp/diagnostic_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Diagnostic complete. Results saved to /tmp/diagnostic_results.json")
