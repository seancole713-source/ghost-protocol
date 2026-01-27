#!/usr/bin/env python3
"""
Test WOLF-specific questions to ensure they STILL get context
"""

import requests

BASE_URL = "http://localhost:5000"
AUTH_TOKEN = "supersecret123jamaica713"


def test_wolf_query(query: str):
    """Send WOLF query and check for proper context"""
    url = f"{BASE_URL}/api/agent/ask"

    try:
        response = requests.post(
            url,
            json={"question": query},
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
            timeout=20,
        )

        if response.status_code != 200:
            return {"query": query, "status": "ERROR", "error": f"HTTP {response.status_code}"}

        answer = response.json().get("answer", "")

        # WOLF questions SHOULD have trading context
        has_context = any(
            word in answer.lower() for word in ["price", "trading", "wolf", "$", "buy", "sell"]
        )

        return {
            "query": query,
            "status": "HAS_CONTEXT" if has_context else "MISSING_CONTEXT",
            "answer": answer[:300],
        }
    except Exception as e:
        return {"query": query, "status": "ERROR", "error": str(e)}


def main():
    print("=== WOLF CONTEXT TEST ===\n")

    # Questions that SHOULD have WOLF context
    queries = [
        "What's the current WOLF price?",
        "Should I buy WOLF stock?",
        "How is Wolfspeed performing?",
        "Show me WOLF trading signals",
    ]

    results = []
    for q in queries:
        print(f"Testing: '{q}'")
        result = test_wolf_query(q)
        results.append(result)
        print(f"  → {result['status']}")

    print("\n" + "=" * 70)
    for r in results:
        print(f"\nQ: {r['query']}")
        print(f"Status: {r['status']}")
        if r["status"] == "HAS_CONTEXT":
            print(f"✅ Answer: {r.get('answer', 'N/A')}...")
        else:
            print(f"❌ Answer: {r.get('answer', 'N/A')}")

    # Verdict
    print("\n" + "=" * 70)
    success = sum(1 for r in results if r["status"] == "HAS_CONTEXT")
    total = len(results)

    if success == total:
        print(f"✅ WOLF CONTEXT TEST: PASS ({success}/{total} have context)")
        return 0
    else:
        print(f"❌ WOLF CONTEXT TEST: FAIL ({success}/{total} have context)")
        return 1


if __name__ == "__main__":
    exit(main())
