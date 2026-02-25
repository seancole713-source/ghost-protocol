#!/usr/bin/env python3
"""
Test general questions (non-meta, non-WOLF) to ensure NO contamination
"""

import pytest
import requests

pytestmark = pytest.mark.skip(reason="Manual integration script — requires running server at localhost:5000")

BASE_URL = "http://localhost:5000"
AUTH_TOKEN = "supersecret123jamaica713"


def test_general_query(query: str) -> dict:
    """Send general query and check for WOLF contamination"""
    url = f"{BASE_URL}/api/agent/ask"

    try:
        response = requests.post(
            url,
            json={"question": query},
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            timeout=20,
        )

        if response.status_code != 200:
            return {
                "query": query,
                "status": "ERROR",
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
            }

        answer = response.json().get("answer", "")

        # Check for WOLF contamination
        wolf_indicators = [
            "wolfspeed",
            "$32.57",
            "previous close",
            "fusion score",
            "buy condition",
            "sell condition",
            "hold condition",
            "trading at",
            "market sentiment",
            "bearish",
            "bullish",
        ]

        found_wolf = [w for w in wolf_indicators if w.lower() in answer.lower()]

        return {
            "query": query,
            "status": "CONTAMINATED" if found_wolf else "CLEAN",
            "answer": answer[:400],
            "wolf_indicators": found_wolf,
        }
    except Exception as e:
        return {"query": query, "status": "ERROR", "error": str(e)}


def main():
    print("=== GENERAL QUERY CONTAMINATION TEST ===\n")

    # Questions that should NOT mention WOLF
    queries = [
        "What's the top crypto?",
        "Tell me about Bitcoin",
        "What's happening with Ethereum?",
        "Who won the election?",
        "What's the weather like?",
        "Explain quantum computing",
        "What is AI?",
    ]

    results = []
    for q in queries:
        print(f"Testing: '{q}'")
        result = test_general_query(q)
        results.append(result)
        print(f"  → {result['status']}")
        if result.get("wolf_indicators"):
            print(f"  ⚠️  WOLF contamination found: {result['wolf_indicators']}")

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)

    for r in results:
        print(f"\nQ: {r['query']}")
        print(f"Status: {r['status']}")
        if r["status"] == "CLEAN":
            print(f"✅ Answer: {r.get('answer', 'N/A')}")
        elif r["status"] == "CONTAMINATED":
            print(f"❌ WOLF Contamination: {r['wolf_indicators']}")
            print(f"   Answer: {r.get('answer', 'N/A')}")
        else:
            print(f"🚨 ERROR: {r.get('error', 'Unknown')}")

    # Verdict
    print("\n" + "=" * 70)
    clean_count = sum(1 for r in results if r["status"] == "CLEAN")
    total = len(results)

    if clean_count == total:
        print(f"✅ ANTI-CONTAMINATION TEST: PASS ({clean_count}/{total} clean)")
        return 0
    else:
        print(f"❌ ANTI-CONTAMINATION TEST: FAIL ({clean_count}/{total} clean)")
        return 1


if __name__ == "__main__":
    exit(main())
