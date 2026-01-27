#!/usr/bin/env python3
"""
Railway health check and diagnostic script.
Run this to verify the app is working correctly on Railway.
"""
import requests
import sys
import time

RAILWAY_URL = "https://ghost-sniper-bot-seancole713-production.up.railway.app"

def check_endpoint(path, expected_keys=None):
    """Check if an endpoint is responding."""
    url = f"{RAILWAY_URL}{path}"
    try:
        response = requests.get(url, timeout=10)
        print(f"✓ {path}: {response.status_code}")
        
        if response.status_code == 200 and expected_keys:
            data = response.json()
            missing = [k for k in expected_keys if k not in data]
            if missing:
                print(f"  ⚠️  Missing keys: {missing}")
            else:
                print(f"  ✓ All expected keys present")
                
        return response.status_code == 200
    except requests.Timeout:
        print(f"✗ {path}: TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ {path}: {type(e).__name__}: {e}")
        return False

def main():
    print("🔍 Railway Health Check\n")
    print(f"Target: {RAILWAY_URL}\n")
    
    checks = [
        ("/", None),
        ("/api/health", ["status"]),
        ("/api/health/predictions", ["health"]),
        ("/api/cockpit", ["ghost_2x"]),
        ("/api/predictions/multi/run", ["ok", "predictions"]),
    ]
    
    results = []
    for path, keys in checks:
        print(f"Checking {path}...")
        success = check_endpoint(path, keys)
        results.append((path, success))
        time.sleep(2)
    
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for path, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {path}")
    
    print(f"\nResult: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Railway is healthy.")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Railway may have issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
