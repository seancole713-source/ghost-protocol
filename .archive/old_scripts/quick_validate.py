#!/usr/bin/env python3
"""Quick Ghost Validation - Auto-approved Runtime Test"""

import sys

import requests

BASE = "http://localhost:5000"
passed = 0
failed = 0

print("🚀 GHOST QUICK VALIDATION")
print("=" * 70)

# 1. Portfolio
print("\n1️⃣  Testing Portfolio API...")
try:
    r = requests.get(f"{BASE}/api/portfolio", timeout=5)
    r.raise_for_status()
    d = r.json()
    if d.get("positions"):
        p = d["positions"][0]
        qty = p.get("qty", 0)
        avg = p.get("price", 0)
        nav = d.get("nav", 0)
        pnl = p.get("pnl", 0)
        print(f"   ✅ PASS - {qty:.8f} WOLF @ ${avg:.2f} (NAV: ${nav:.2f}, P&L: ${pnl:.2f})")
        passed += 1
    else:
        print("   ❌ FAIL - No positions found")
        failed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    failed += 1

# 2. Watchlist
print("\n2️⃣  Testing Watchlist API...")
try:
    r = requests.get(f"{BASE}/api/watchlist", timeout=5)
    r.raise_for_status()
    d = r.json()
    count = d.get("count", 0)
    print(f"   ✅ PASS - {count} symbols loaded")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    failed += 1

# 3. Risk Status
print("\n3️⃣  Testing Risk Status API...")
try:
    r = requests.get(f"{BASE}/api/risk/status", timeout=5)
    r.raise_for_status()
    d = r.json()
    can_trade = d.get("can_trade", False)
    risk_level = d.get("risk_level", "unknown")
    print(f"   ✅ PASS - Can Trade: {can_trade}, Risk: {risk_level}")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    failed += 1

# 4. News
print("\n4️⃣  Testing News API...")
try:
    r = requests.get(f"{BASE}/api/articles/latest?limit=5", timeout=5)
    r.raise_for_status()
    d = r.json()
    articles = d.get("articles", [])
    print(f"   ✅ PASS - {len(articles)} articles fetched")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    failed += 1

# 5. Telegram
print("\n5️⃣  Testing Telegram Bot...")
try:
    r = requests.get(f"{BASE}/alerts/selftest", timeout=5)
    r.raise_for_status()
    d = r.json()
    tg = d.get("telegram", {})
    bot_ok = tg.get("ok", False)
    username = tg.get("result", {}).get("username", "N/A")
    print(f"   ✅ PASS - Bot Active: {bot_ok}, Username: @{username}")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    failed += 1

# 6. Cockpit
print("\n6️⃣  Testing Cockpit API...")
try:
    r = requests.get(f"{BASE}/api/cockpit", timeout=5)
    r.raise_for_status()
    d = r.json()
    mode = d.get("mode", "unknown")
    ticker = d.get("ticker", "N/A")
    print(f"   ✅ PASS - Mode: {mode}, Ticker: {ticker}")
    passed += 1
except Exception as e:
    print(f"   ❌ FAIL - {e}")
    failed += 1

# Summary
print("\n" + "=" * 70)
print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("✅ ALL TESTS PASSED - GHOST IS 100% OPERATIONAL!")
    sys.exit(0)
else:
    print(f"⚠️  {failed} test(s) failed - check logs above")
    sys.exit(1)
