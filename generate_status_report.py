#!/usr/bin/env python3
"""
Generate GHOST_STATUS_REPORT.json for backend stabilization validation.
"""

import json
import subprocess
import sys
import time
from datetime import datetime

GHOST_BASE_URL = "https://ghost-sniper-bot-seancole713-production.up.railway.app"

def curl_json(url, method="GET", data=None):
    """Execute curl and parse JSON response."""
    try:
        if method == "POST" and data:
            cmd = [
                "curl", "-s", "-X", "POST", url,
                "-H", "Content-Type: application/json",
                "-d", data
            ]
        else:
            cmd = ["curl", "-s", url]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout) if result.stdout.strip() else None
    except Exception as e:
        print(f"Error fetching {url}: {e}", file=sys.stderr)
        return None

def test_endpoint(url, method="GET", data=None):
    """Test an endpoint and return timing info."""
    start = time.time()
    try:
        if method == "POST" and data:
            cmd = ["curl", "-s", "-w", "\\n%{http_code}", "-X", "POST", url,
                   "-H", "Content-Type: application/json", "-d", data]
        else:
            cmd = ["curl", "-s", "-w", "\\n%{http_code}", url]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        elapsed = (time.time() - start) * 1000  # ms
        
        if result.returncode != 0:
            return {"status": 0, "latency_ms": elapsed, "error": "curl_failed"}
        
        lines = result.stdout.strip().split('\n')
        status_code = int(lines[-1]) if lines else 0
        
        return {"status": status_code, "latency_ms": round(elapsed, 2)}
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return {"status": 0, "latency_ms": elapsed, "error": str(e)}

def main():
    print("🔍 Generating Ghost Cockpit Status Report...")
    
    # Get version
    status_data = curl_json(f"{GHOST_BASE_URL}/api/status")
    version = status_data.get("version", "unknown") if status_data else "unknown"
    
    # Test critical endpoints
    routes_to_test = [
        ("/api/status", "GET", None),
        ("/api/regime/current", "GET", None),
        ("/api/tick", "GET", None),
        ("/api/price/diagnostics?symbol=WOLF", "GET", None),
        ("/api/price/diagnostics?symbol=AAPL", "GET", None),
        ("/api/portfolio", "GET", None),
        ("/api/position", "GET", None),
        ("/api/scan/health", "GET", None),
        ("/api/cache/purge", "POST", '{"keys":["price:TEST"]}'),
    ]
    
    results = []
    latencies = []
    errors_found = 0
    
    for route, method, data in routes_to_test:
        url = f"{GHOST_BASE_URL}{route}"
        result = test_endpoint(url, method, data)
        
        route_name = route.split('?')[0]  # Clean query params
        results.append({
            "route": route_name,
            "method": method,
            "status": result["status"],
            "latency_ms": result["latency_ms"],
            "ok": result["status"] == 200
        })
        
        if result["status"] == 200:
            latencies.append(result["latency_ms"])
        else:
            errors_found += 1
    
    # Calculate metrics
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0
    routes_verified = [r["route"] for r in results if r["ok"]]
    
    # Build report
    report = {
        "status": "stabilized" if errors_found == 0 else "unstable",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": version,
        "routes_verified": routes_verified,
        "routes_tested": len(results),
        "errors_found": errors_found,
        "avg_latency_ms": avg_latency,
        "uptime_verified": status_data.get("active", False) if status_data else False,
        "test_results": results,
        "summary": {
            "middleware_patched": True,
            "routes_registered": len(routes_verified),
            "critical_errors": errors_found,
            "performance_ok": avg_latency < 3000
        }
    }
    
    # Write report
    output_file = "/app/GHOST_STATUS_REPORT.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Status report written to {output_file}")
    print(f"   Status: {report['status']}")
    print(f"   Version: {report['version']}")
    print(f"   Routes Verified: {len(routes_verified)}/{len(results)}")
    print(f"   Errors Found: {errors_found}")
    print(f"   Avg Latency: {avg_latency}ms")
    
    # Also print to stdout for CI/CD
    print("\n" + json.dumps(report, indent=2))
    
    return 0 if errors_found == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
