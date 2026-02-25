"""
Quick test to verify AgentKit integration works
Run with: python test_agentkit_integration.py
"""

import os
import sys

# Set test environment
os.environ["AGENTKIT_ENABLED"] = "false"  # Test fallback first
os.environ["OPENAI_API_KEY"] = "test-key-placeholder"

sys.path.insert(0, "/workspaces/ghost-protocol")


def test_agent_import():
    """Test that agent module imports without errors."""
    from llm import agent  # noqa: F401


def test_agentkit_import():
    """Test that agentkit module exists and imports."""
    from llm import agentkit  # noqa: F401


def test_agent_disabled():
    """Test agent with no API key returns disabled message."""
    from llm.agent import run_once

    # Clear API key so run_once sees empty key at call time
    old_key = os.environ.get("OPENAI_API_KEY", "")
    old_agent_key = os.environ.get("OPENAI_AGENT_API_KEY", "")
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["OPENAI_AGENT_API_KEY"] = ""

    try:
        def mock_tool_router(func_name, args):
            return {"ok": True}

        result = run_once(mock_tool_router)

        assert result["action"] == "HOLD", f"Expected HOLD, got {result['action']}"
        assert "disabled" in result["rationale"].lower(), f"Expected 'disabled' in rationale, got: {result['rationale']}"
    finally:
        # Restore keys
        os.environ["OPENAI_API_KEY"] = old_key
        os.environ["OPENAI_AGENT_API_KEY"] = old_agent_key


def test_agentkit_client():
    """Test AgentKit client initialization."""
    from llm.agentkit import AgentKitClient

    try:
        # Should raise without API key
        AgentKitClient(api_key="")
        print("❌ AgentKitClient should raise ValueError with no API key")
        return False
    except ValueError as e:
        if "OPENAI_API_KEY" in str(e):
            print("✅ AgentKitClient correctly validates API key")
            return True
        else:
            print(f"❌ Wrong error: {e}")
            return False


def test_normalize_decision():
    """Test decision normalization."""
    from llm.agentkit import _normalize_decision

    test_cases = [
        ({"action": "buy", "confidence": 75}, {"action": "BUY", "confidence": 75}),
        ({"action": "INVALID", "confidence": 150}, {"action": "HOLD", "confidence": 100}),
        ({}, {"action": "HOLD", "confidence": 50}),
    ]

    for input_data, expected_subset in test_cases:
        result = _normalize_decision(input_data)
        if (
            result["action"] == expected_subset["action"]
            and result["confidence"] == expected_subset["confidence"]
        ):
            print(
                f"✅ Normalize decision: {input_data} → {result['action']}/{result['confidence']}"
            )
        else:
            print(f"❌ Normalize failed: {input_data} → {result}")
            return False

    return True


def main():
    print("=" * 60)
    print("GHOST AgentKit Integration Test")
    print("=" * 60)

    tests = [
        ("Agent module import", test_agent_import),
        ("AgentKit module import", test_agentkit_import),
        ("Agent disabled behavior", test_agent_disabled),
        ("AgentKit client validation", test_agentkit_client),
        ("Decision normalization", test_normalize_decision),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✅ All tests passed! AgentKit integration is working.")
        return 0
    else:
        print(f"❌ {failed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
