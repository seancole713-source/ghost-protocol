#!/usr/bin/env python3
"""
GATE 8: AI Meta Truthiness Test
Tests meta queries do NOT leak trading content
"""

import time

import pytest
import requests

pytestmark = pytest.mark.skip(reason="Manual integration script — requires running server at localhost:5000")

BASE_URL = "http://localhost:5000"
AUTH_TOKEN = "supersecret123jamaica713"


def test_meta_query(query: str) -> dict:
    """Send meta query to Ghost AI and check response"""
    url = f"{BASE_URL}/api/agent/ask"

    try:
        response = requests.post(
            url,
            json={"question": query},
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            timeout=15,
        )

        if response.status_code != 200:
            return {
                "query": query,
                "status": "ERROR",
                "code": response.status_code,
                "text": response.text[:200],
            }

        answer = response.json().get("answer", "")

        # Check for trading content contamination
        bad_words = [
            "BUY",
            "SELL",
            "HOLD",
            "BULLISH",
            "BEARISH",
            "TRADE",
            "STOCK",
            "PRICE",
            "VOLUME",
            "RECOMMENDATION",
        ]
        found_bad = [w for w in bad_words if w.lower() in answer.lower()]

        return {
            "query": query,
            "status": "FAIL" if found_bad else "PASS",
            "answer": answer[:300],
            "contamination": found_bad,
        }
    except Exception as e:
        return {"query": query, "status": "ERROR", "error": str(e)}


def main():
    print("=== STEP 8: AI META TRUTHINESS TEST ===\n")

    # Test queries from user's complaint
    queries = [
        "what time is it",
        "what time is it?",
        "what's the time",
        "current time",
        "ghost health",
        "system status",
        "are you alive",
    ]

    results = []
    for q in queries:
        print(f"Testing: '{q}'")
        result = test_meta_query(q)
        results.append(result)
        print(f"  → {result['status']}")
        if result.get("contamination"):
            print(f"  ⚠️  Found: {result['contamination']}")
        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)

    for r in results:
        print(f"\nQ: {r['query']}")
        print(f"Status: {r['status']}")
        if r["status"] == "PASS":
            print(f"✅ Answer: {r.get('answer', 'N/A')}")
        elif r["status"] == "FAIL":
            print(f"❌ Contamination: {r['contamination']}")
            print(f"   Answer: {r.get('answer', 'N/A')}")
        else:
            print(f"🚨 ERROR: {r.get('error', r.get('text', 'Unknown'))}")

    # Gate verdict
    print("\n" + "=" * 60)
    all_pass = all(r["status"] == "PASS" for r in results)
    if all_pass:
        print("✅ G8: AI META TRUTHINESS - PASS")
        return 0
    else:
        print("🚨 G8: AI META TRUTHINESS - FAIL")
        return 1


if __name__ == "__main__":
    exit(main())
