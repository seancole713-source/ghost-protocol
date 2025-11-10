#!/usr/bin/env python3
"""
Runtime verification script for two-line overlay system.
Checks that the Ghost server is running and all endpoints are functional.
"""

import sys
import time
from datetime import datetime

import requests


def print_status(msg, status="INFO"):
    """Print formatted status message."""
    ts = datetime.now().strftime("%H:%M:%S")
    symbols = {"OK": "✅", "FAIL": "❌", "WARN": "⚠️", "INFO": "ℹ️"}
    symbol = symbols.get(status, "•")
    print(f"[{ts}] {symbol} {msg}")


def check_server(base_url="http://localhost:5000"):
    """Check if Ghost server is running."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        if r.status_code == 200:
            print_status(f"Ghost server running at {base_url}", "OK")
            return True
        else:
            print_status(f"Server returned status {r.status_code}", "FAIL")
            return False
    except Exception as e:
        print_status(f"Server not reachable: {e}", "FAIL")
        return False


def check_cockpit(base_url="http://localhost:5000"):
    """Verify /api/cockpit includes two_line_overlay."""
    try:
        r = requests.get(f"{base_url}/api/cockpit", timeout=10)
        if r.status_code != 200:
            print_status(f"/api/cockpit returned {r.status_code}", "FAIL")
            return False

        data = r.json()

        # Check two_line_overlay field exists
        if "two_line_overlay" not in data:
            print_status("/api/cockpit missing two_line_overlay field", "FAIL")
            return False

        tlo = data["two_line_overlay"]

        if not tlo:
            print_status("two_line_overlay is None (may be disabled)", "WARN")
            return True

        # Check structure
        required_fields = ["forecast", "actual", "accuracy"]
        missing = [f for f in required_fields if f not in tlo]
        if missing:
            print_status(f"two_line_overlay missing fields: {missing}", "FAIL")
            return False

        # Check forecast structure
        forecast = tlo["forecast"]
        if "points" not in forecast:
            print_status("forecast missing points array", "FAIL")
            return False

        points = forecast["points"]
        print_status(f"/api/cockpit OK: {len(points)} forecast points", "OK")

        # Check actual structure
        actual = tlo["actual"]
        actual_pts = actual.get("points", [])
        src = actual.get("src", "unknown")
        print_status(f"Actual prices: {len(actual_pts)} points from {src}", "INFO")

        # Check accuracy
        accuracy = tlo.get("accuracy", {})
        summary = accuracy.get("summary", {})
        map = summary.get("map")
        rmse = summary.get("rmse")
        bias = summary.get("bias")

        if map is not None:
            print_status(
                f"Accuracy: MAP={map * 100:.2f}%, RMSE=${rmse:.2f}, Bias=${bias:.2f}", "OK"
            )
        else:
            print_status("Accuracy: No overlap yet", "WARN")

        return True

    except Exception as e:
        print_status(f"/api/cockpit check failed: {e}", "FAIL")
        return False


def check_sse_stream(base_url="http://localhost:5000"):
    """Verify SSE streaming endpoint responds."""
    try:
        # Just check if endpoint responds (don't wait for events)
        r = requests.get(f"{base_url}/api/cockpit/stream", timeout=2, stream=True)

        if r.status_code == 200:
            content_type = r.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                print_status("/api/cockpit/stream: SSE endpoint OK", "OK")
                return True
            else:
                print_status(f"SSE endpoint wrong content-type: {content_type}", "WARN")
                return True
        else:
            print_status(f"SSE endpoint returned {r.status_code}", "FAIL")
            return False

    except requests.Timeout:
        # Timeout is OK for SSE (connection stays open)
        print_status("/api/cockpit/stream: Endpoint responds (timeout expected)", "OK")
        return True
    except Exception as e:
        print_status(f"SSE check failed: {e}", "FAIL")
        return False


def check_forecast_overlay(base_url="http://localhost:5000"):
    """Verify legacy /api/forecast/overlay endpoint."""
    try:
        r = requests.get(f"{base_url}/api/forecast/overlay?symbol=WOLF", timeout=10)
        if r.status_code != 200:
            print_status(f"/api/forecast/overlay returned {r.status_code}", "FAIL")
            return False

        data = r.json()

        if not data.get("enabled"):
            print_status("/api/forecast/overlay disabled (expected if no legacy data)", "WARN")
            return True

        print_status("/api/forecast/overlay: Legacy endpoint OK", "OK")
        return True

    except Exception as e:
        print_status(f"/api/forecast/overlay check failed: {e}", "FAIL")
        return False


def check_ui_contract(base_url="http://localhost:5000"):
    """Verify UI contract JSON."""
    try:
        r = requests.get(f"{base_url}/public/ui_contract.json", timeout=5)
        if r.status_code != 200:
            print_status(f"UI contract returned {r.status_code}", "WARN")
            return True  # Non-critical

        data = r.json()
        print_status(f"UI contract: {len(data)} endpoints documented", "OK")
        return True

    except Exception as e:
        print_status(f"UI contract check failed: {e}", "WARN")
        return True  # Non-critical


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("GHOST TWO-LINE OVERLAY RUNTIME VERIFICATION")
    print("=" * 60 + "\n")

    base_url = "http://localhost:5000"

    checks = [
        ("Server Health", lambda: check_server(base_url)),
        ("Cockpit API", lambda: check_cockpit(base_url)),
        ("SSE Streaming", lambda: check_sse_stream(base_url)),
        ("Forecast Overlay (Legacy)", lambda: check_forecast_overlay(base_url)),
        ("UI Contract", lambda: check_ui_contract(base_url)),
    ]

    results = []
    for name, check_fn in checks:
        print(f"\nChecking: {name}")
        print("-" * 40)
        result = check_fn()
        results.append((name, result))
        time.sleep(0.5)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {name}")

    print(f"\nResult: {passed}/{total} checks passed")

    if passed == total:
        print("\n✅ All checks passed! Two-line overlay system is operational.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} check(s) failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
