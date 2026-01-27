#!/usr/bin/env python3
"""
RAILWAY DIAGNOSTIC TOOL
Minimal test to confirm Railway can run Python and access environment
"""

import os
import sys

def main():
    print("=" * 60)
    print("RAILWAY DIAGNOSTIC START")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"PORT: {os.getenv('PORT', 'NOT_SET')}")
    print(f"RAILWAY_ENVIRONMENT: {os.getenv('RAILWAY_ENVIRONMENT', 'NOT_SET')}")
    print(f"REDIS_URL: {'SET (len=' + str(len(os.getenv('REDIS_URL', ''))) + ')' if os.getenv('REDIS_URL') else 'NOT_SET'}")
    print(f"POLYGON_API_KEY: {'SET' if os.getenv('POLYGON_API_KEY') else 'NOT_SET'}")
    print(f"ALPHAVANTAGE_API_KEY: {'SET' if os.getenv('ALPHAVANTAGE_API_KEY') else 'NOT_SET'}")
    print("=" * 60)
    
    # Try importing wolf_app
    try:
        print("Attempting to import wolf_app...")
        import wolf_app
        print("✅ wolf_app imported successfully!")
        print(f"REDIS variable exists: {hasattr(wolf_app, 'REDIS')}")
        print(f"_get_redis function exists: {hasattr(wolf_app, '_get_redis')}")
    except Exception as e:
        print(f"❌ wolf_app import FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 60)
    print("RAILWAY DIAGNOSTIC COMPLETE")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
