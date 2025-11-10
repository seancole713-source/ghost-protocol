#!/usr/bin/env python3
"""
Lightweight in-process smoke checks using FastAPI TestClient so we don't need a live uvicorn.
Verifies: /health, /api/status, enhanced BTC price, AAPL quote, alerts/test.
"""

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import main


def run():
    client = TestClient(main.app)

    def check(path: str):
        try:
            resp = client.get(path)
            print(f"GET {path} -> {resp.status_code}")
            txt = resp.text or ""
            print(txt[:400])
            print("\n---\n")
            return resp
        except Exception as e:
            print(f"ERR {path}: {e}")
            print("\n---\n")
            return None

    print("Booting TestClient and running smoke checks...\n")
    check("/health")
    check("/api/status")
    # Enhanced price for BTC
    check("/api/prices/enhanced?ids=bitcoin&asset_type=crypto")
    # AAPL quote
    check("/api/quotes?symbols=AAPL")
    # Alerts selftest (read-only, should never send)
    check("/alerts/selftest")
    print("Done.")


if __name__ == "__main__":
    run()
