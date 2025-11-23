#!/usr/bin/env python3
"""
Railway Deployment Status Checker
Monitors Railway deployment and tests critical endpoints
"""

import json
import subprocess
import sys

PROD_URL = "https://ghost-protocol-production.up.railway.app"

def check_endpoint(path: str, expected_keys: list[str] = None) -> tuple[bool, dict]:
    """
    Check if endpoint is responsive and returns expected data using curl
    
    Returns:
        (success: bool, data: dict)
    """
    url = f"{PROD_URL}{path}"
    try:
        result = subprocess.run(
            ['curl', '-s', '-w', '\\n%{http_code}', url],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        lines = result.stdout.strip().split('\n')
        status_code = lines[-1] if lines else '000'
        response_body = '\n'.join(lines[:-1]) if len(lines) > 1 else ''
        
        if status_code != '200':
            return False, {"error": f"HTTP {status_code}"}
        
        data = json.loads(response_body)
        
        # Check expected keys if provided
        if expected_keys:
            missing = [k for k in expected_keys if k not in data]
            if missing:
                return False, {"error": f"Missing keys: {missing}", "data": data}
        
        return True, data
    except json.JSONDecodeError as e:
        return False, {"error": f"Invalid JSON: {e}"}
    except subprocess.TimeoutExpired:
        return False, {"error": "Request timeout"}
    except Exception as e:
        return False, {"error": str(e)}

def main():
    print("🚀 Ghost Protocol - Railway Deployment Status")
    print("=" * 60)
    print(f"Production URL: {PROD_URL}")
    print("")
    
    tests = [
        {
            "name": "Health Check",
            "path": "/health",
            "keys": ["status", "service", "uptime"]
        },
        {
            "name": "V3 Watchlist",
            "path": "/api/v3/watchlist",
            "keys": ["stocks", "crypto", "count"]
        },
        {
            "name": "V3 Cockpit Status",
            "path": "/api/v3/cockpit/status",
            "keys": ["timestamp"]
        },
        {
            "name": "V3 Goals Snapshot",
            "path": "/api/v3/goals/snapshot",
            "keys": ["timestamp"]
        },
    ]
    
    results = []
    for test in tests:
        print(f"Testing: {test['name']} ({test['path']})")
        success, data = check_endpoint(test['path'], test.get('keys'))
        
        if success:
            print(f"  ✅ PASS")
            # Show key metrics
            if test['path'] == '/health':
                print(f"     Uptime: {data.get('uptime', 0)}s")
            elif test['path'] == '/api/v3/watchlist':
                count = data.get('count', 0)
                stocks = len(data.get('stocks', []))
                crypto = len(data.get('crypto', []))
                print(f"     Symbols: {count} total ({stocks} stocks, {crypto} crypto)")
                if count == 0:
                    print(f"     ⚠️  WARNING: Watchlist is empty!")
            elif test['path'] == '/api/v3/goals/snapshot':
                goals = data.get('goals', {})
                if goals:
                    daily = goals.get('daily', {})
                    print(f"     Daily goal: ${daily.get('target', 0):,.0f}")
        else:
            print(f"  ❌ FAIL: {data.get('error', 'Unknown error')}")
        
        results.append({"test": test['name'], "success": success, "data": data})
        print("")
    
    # Summary
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    pct = (passed / total * 100) if total > 0 else 0
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed ({pct:.0f}%)")
    
    if passed == total:
        print("✅ All systems operational!")
        return 0
    elif passed >= total * 0.75:
        print("⚠️  Most systems operational, some warnings")
        return 0
    else:
        print("❌ Critical failures detected")
        return 1

if __name__ == "__main__":
    sys.exit(main())
