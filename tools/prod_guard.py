#!/usr/bin/env python3
# Production Guard: verify live Railway deployment health and critical endpoints

import json
import os
import sys
import time

import requests

BASE = os.getenv("PROD_BASE_URL", "https://web-production-8e9a0.up.railway.app").rstrip("/")
EXPECTED_MIN_ROUTES = int(os.getenv("EXPECTED_MIN_ROUTES", "260"))
TIMEOUT = float(os.getenv("HTTP_TIMEOUT_S", "10"))


CRITICAL_ENDPOINTS: list[tuple[str, int]] = [
    ("/health", 200),
    ("/api/news", 200),
    ("/api/news/recent", 200),
    ("/api/agent/decisions", 200),
    ("/api/agent/stats", 200),
    ("/api/portfolio", 200),
    ("/api/snapshot", 200),
    ("/api/stage2/forecasts", 200),
    ("/api/stage1/world", 200),
    # aliases added to avoid 404s
    ("/api/market/movers", 200),
    ("/api/predictions/run", 200),
    ("/api/sources/status", 200),
]


def get(url: str) -> requests.Response:
    return requests.get(url, timeout=TIMEOUT)


def main() -> int:
    summary = {
        "base": BASE,
        "ts": int(time.time()),
        "openapi_routes": None,
        "endpoints": [],
        "errors": [],
        "ok": False,
    }

    # OpenAPI route count
    try:
        r = get(f"{BASE}/openapi.json")
        r.raise_for_status()
        data = r.json()
        paths = data.get("paths", {})
        count = len(paths)
        summary["openapi_routes"] = count
        if count < EXPECTED_MIN_ROUTES:
            summary["errors"].append(
                f"route_count_below_threshold: {count} < {EXPECTED_MIN_ROUTES}"
            )
    except Exception as e:
        summary["errors"].append(f"openapi_fetch_failed: {e}")

    # Probe endpoints
    for path, expect in CRITICAL_ENDPOINTS:
        url = f"{BASE}{path}"
        item = {"path": path, "expected": expect, "status": None}
        try:
            res = get(url)
            item["status"] = res.status_code
            if str(res.status_code) != str(expect):
                summary["errors"].append(f"bad_status:{path}:{res.status_code}!={expect}")
        except Exception as e:
            item["status"] = f"ERR:{e}"
            summary["errors"].append(f"exception:{path}:{e}")
        summary["endpoints"].append(item)

    summary["ok"] = len(summary["errors"]) == 0

    # Print human-friendly summary and write JSON artifact
    print("=== PROD GUARD SUMMARY ===")
    print(f"Base: {BASE}")
    print(f"OpenAPI routes: {summary['openapi_routes']}")
    for ep in summary["endpoints"]:
        print(f"{ep['path']}: {ep['status']} (expected {ep['expected']})")
    if summary["errors"]:
        print("Errors:")
        for e in summary["errors"]:
            print(f" - {e}")

    try:
        os.makedirs("audit_out", exist_ok=True)
        with open("audit_out/prod_guard_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
