#!/usr/bin/env python3
"""
Ghost Chat - Ask Ghost investment questions
Usage: python chat_with_ghost.py "What's the best crypto under $1?"
"""

import sys

import requests

BASE_URL = "http://localhost:8444"


def chat(message: str):
    """Send message to Ghost and get response"""
    try:
        response = requests.post(
            f"{BASE_URL}/api/advisor/chat", params={"message": message}, timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            print("\n" + "=" * 80)
            print("🤖 GHOST AI ADVISOR")
            print("=" * 80)
            print(f"\n📝 Your Question: {data['message']}")
            print("\n💬 Ghost's Response:\n")
            print(data["response"])
            print("\n📊 Context Used:")
            print(f"   - Opportunities analyzed: {data['context_used']['opportunities_count']}")
            print(f"   - Cryptos under $1: {data['context_used']['under_1_dollar_count']}")
            print(f"   - Market regime: {data['context_used']['market_regime']}")
            print("\n" + "=" * 80 + "\n")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to Ghost. Is the server running?")
        print("   Start it with: ./start_ai_advisor.sh")
    except Exception as e:
        print(f"❌ Error: {e}")


def interactive_mode():
    """Run in interactive chat mode"""
    print("\n" + "=" * 80)
    print("🤖 GHOST AI ADVISOR - Interactive Chat Mode")
    print("=" * 80)
    print("\nAsk Ghost anything about investments!")
    print("Examples:")
    print("  - What's the best crypto under $1?")
    print("  - Should I buy Bitcoin right now?")
    print("  - If I invest $1000 in SOL, what profit in 30 days?")
    print("  - What are the top 3 stocks today?")
    print("\nType 'exit' or 'quit' to end.\n")

    while True:
        try:
            message = input("💬 You: ").strip()

            if message.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!\n")
                break

            if not message:
                continue

            chat(message)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except EOFError:
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single question mode
        message = " ".join(sys.argv[1:])
        chat(message)
    else:
        # Interactive mode
        interactive_mode()
