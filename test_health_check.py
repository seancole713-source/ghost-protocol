#!/usr/bin/env python3
"""Test the actual health check price feed function"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.system_doctor import _check_price_feed

print("Testing health check price feed...")
result = _check_price_feed()
print(f"\nResult: {result}")
print(f"Pass: {result.get('pass')}")
print(f"Severity: {result.get('severity')}")
print(f"Detail: {result.get('detail')}")
