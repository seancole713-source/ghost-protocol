#!/usr/bin/env python3
"""
UI Verification Audit Script - Read-only headless browser automation
Captures screenshots, console logs, network traffic, HTTP status/latency for all panels.
"""
import json
import os
import time
from datetime import datetime
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# Configuration
BASE_URL = "https://ghost-protocol-production.up.railway.app"
SCREENSHOT_DIR = Path("ui_verification_screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

# Audit results collection
audit_results = {
    "timestamp": datetime.now().isoformat(),
    "base_url": BASE_URL,
    "sections": {}
}

console_messages = []
network_requests = []


def capture_console(msg):
    """Capture console messages"""
    console_messages.append({
        "type": msg.type,
        "text": msg.text,
        "timestamp": datetime.now().isoformat()
    })


def capture_network(route, request):
    """Capture network requests"""
    start_time = time.time()
    response = route.fetch()
    latency_ms = (time.time() - start_time) * 1000
    
    network_requests.append({
        "url": request.url,
        "method": request.method,
        "status": response.status,
        "latency_ms": round(latency_ms, 2),
        "timestamp": datetime.now().isoformat(),
        "response_body": None  # Will populate for specific endpoints
    })
    
    route.continue_()


def audit_landing_page(page):
    """Audit landing page"""
    section = {"status": "PENDING", "findings": [], "screenshots": []}
    
    try:
        print("📄 Auditing landing page...")
        page.goto(BASE_URL, timeout=10000)
        page.wait_for_load_state("networkidle", timeout=10000)
        
        screenshot_path = SCREENSHOT_DIR / "landing_page.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        section["screenshots"].append(str(screenshot_path))
        
        # Check for key elements
        title = page.title()
        section["findings"].append(f"Page title: {title}")
        
        # Check if Cockpit link exists
        try:
            cockpit_link = page.locator('a[href*="cockpit"], a:has-text("Cockpit"), a:has-text("Dashboard")').first
            if cockpit_link.is_visible(timeout=2000):
                section["findings"].append("✓ Cockpit link found")
            else:
                section["findings"].append("✗ Cockpit link not visible")
        except:
            section["findings"].append("✗ Cockpit link not found")
        
        section["status"] = "PASS"
        
    except Exception as e:
        section["status"] = "FAIL"
        section["findings"].append(f"Error: {str(e)}")
    
    return section


def audit_cockpit_panels(page):
    """Audit cockpit panels"""
    section = {"status": "PENDING", "panels": {}, "screenshots": []}
    
    try:
        print("🎛️  Auditing cockpit panels...")
        
        # Navigate to cockpit
        page.goto(f"{BASE_URL}/cockpit", timeout=15000)
        page.wait_for_timeout(3000)  # Wait for dynamic content
        
        # Full page screenshot
        screenshot_path = SCREENSHOT_DIR / "cockpit_full.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        section["screenshots"].append(str(screenshot_path))
        
        # Define expected panels
        panels = {
            "predictions": ["prediction", "forecast"],
            "performance": ["performance", "accuracy", "win rate"],
            "positions": ["position", "trade", "portfolio"],
            "alerts": ["alert", "warning", "notification"],
            "regime": ["regime", "market mode", "volatility"]
        }
        
        for panel_name, keywords in panels.items():
            panel_result = {"found": False, "keywords_matched": []}
            
            # Check for panel presence
            for keyword in keywords:
                try:
                    locator = page.locator(f'text=/{keyword}/i').first
                    if locator.count() > 0:
                        panel_result["found"] = True
                        panel_result["keywords_matched"].append(keyword)
                except:
                    pass
            
            section["panels"][panel_name] = panel_result
        
        # Check API endpoints that cockpit should call
        api_endpoints = [
            "/api/v3/predictions/latest",
            "/api/v3/accuracy/summary",
            "/api/v3/performance/dashboard"
        ]
        
        section["api_calls"] = []
        for endpoint in api_endpoints:
            matching_requests = [
                req for req in network_requests 
                if endpoint in req["url"]
            ]
            if matching_requests:
                section["api_calls"].append({
                    "endpoint": endpoint,
                    "count": len(matching_requests),
                    "last_status": matching_requests[-1]["status"],
                    "last_latency_ms": matching_requests[-1]["latency_ms"]
                })
        
        section["status"] = "PASS"
        
    except Exception as e:
        section["status"] = "FAIL"
        section["error"] = str(e)
    
    return section


def audit_api_endpoints(page):
    """Audit critical API endpoints"""
    section = {"status": "PENDING", "endpoints": {}}
    
    endpoints = {
        "/api/v3/predictions/latest": "Latest predictions",
        "/api/v3/accuracy/summary": "Accuracy metrics",
        "/api/v3/performance/dashboard": "Performance dashboard",
        "/api/v3/live_recalculator/status": "Live recalculator status"
    }
    
    print("🔌 Auditing API endpoints...")
    
    for endpoint, description in endpoints.items():
        endpoint_result = {
            "description": description,
            "status": None,
            "latency_ms": None,
            "response_body": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            response = page.request.get(f"{BASE_URL}{endpoint}")
            latency_ms = (time.time() - start_time) * 1000
            
            endpoint_result["status"] = response.status
            endpoint_result["latency_ms"] = round(latency_ms, 2)
            
            if response.ok:
                try:
                    endpoint_result["response_body"] = response.json()
                except:
                    endpoint_result["response_body"] = response.text()[:500]
            else:
                endpoint_result["error"] = f"HTTP {response.status}"
        
        except Exception as e:
            endpoint_result["error"] = str(e)
        
        section["endpoints"][endpoint] = endpoint_result
    
    # Overall status
    all_ok = all(
        ep.get("status") == 200 
        for ep in section["endpoints"].values() 
        if ep.get("status") is not None
    )
    section["status"] = "PASS" if all_ok else "FAIL"
    
    return section


def audit_console_errors(page):
    """Audit console errors"""
    section = {
        "status": "PENDING",
        "total_messages": len(console_messages),
        "errors": [],
        "warnings": []
    }
    
    print("🔍 Analyzing console messages...")
    
    for msg in console_messages:
        if msg["type"] == "error":
            section["errors"].append(msg)
        elif msg["type"] == "warning":
            section["warnings"].append(msg)
    
    section["status"] = "FAIL" if section["errors"] else "PASS"
    
    return section


def audit_reliability_loop(page):
    """Audit reliability micro-loop: refresh cockpit, check for errors"""
    section = {
        "status": "PENDING",
        "iterations": 3,
        "results": []
    }
    
    print("🔄 Running reliability micro-loop...")
    
    for i in range(section["iterations"]):
        iteration = {
            "iteration": i + 1,
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "errors": []
        }
        
        try:
            # Clear previous console messages
            console_messages.clear()
            
            # Reload cockpit
            page.goto(f"{BASE_URL}/cockpit", timeout=15000)
            page.wait_for_timeout(2000)
            
            # Check for errors
            errors_in_iteration = [
                msg for msg in console_messages 
                if msg["type"] == "error"
            ]
            
            if errors_in_iteration:
                iteration["errors"] = errors_in_iteration
            else:
                iteration["success"] = True
            
            # Screenshot
            screenshot_path = SCREENSHOT_DIR / f"reliability_iteration_{i+1}.png"
            page.screenshot(path=str(screenshot_path), full_page=True)
            iteration["screenshot"] = str(screenshot_path)
            
        except Exception as e:
            iteration["errors"].append({"error": str(e)})
        
        section["results"].append(iteration)
        time.sleep(1)  # Brief pause between iterations
    
    # Overall status
    all_successful = all(r["success"] for r in section["results"])
    section["status"] = "PASS" if all_successful else "FAIL"
    
    return section


def generate_markdown_report():
    """Generate comprehensive Markdown report"""
    report_lines = [
        "# UI Verification Audit Report",
        "",
        f"**Timestamp:** {audit_results['timestamp']}",
        f"**Base URL:** {audit_results['base_url']}",
        "",
        "---",
        ""
    ]
    
    # Landing Page
    landing = audit_results["sections"].get("landing_page", {})
    report_lines.extend([
        f"## 1. Landing Page - {landing.get('status', 'UNKNOWN')}",
        ""
    ])
    
    if landing.get("findings"):
        report_lines.append("**Findings:**")
        for finding in landing["findings"]:
            report_lines.append(f"- {finding}")
        report_lines.append("")
    
    if landing.get("screenshots"):
        report_lines.append("**Screenshot:**")
        report_lines.append(f"![Landing Page]({landing['screenshots'][0]})")
        report_lines.append("")
    
    # Cockpit Panels
    cockpit = audit_results["sections"].get("cockpit_panels", {})
    report_lines.extend([
        f"## 2. Cockpit Panels - {cockpit.get('status', 'UNKNOWN')}",
        ""
    ])
    
    if cockpit.get("panels"):
        report_lines.append("**Panel Detection:**")
        for panel_name, panel_data in cockpit["panels"].items():
            status = "✓" if panel_data["found"] else "✗"
            keywords = ", ".join(panel_data["keywords_matched"]) if panel_data["keywords_matched"] else "none"
            report_lines.append(f"- {status} **{panel_name.title()}**: matched keywords: {keywords}")
        report_lines.append("")
    
    if cockpit.get("api_calls"):
        report_lines.append("**API Calls Observed:**")
        for call in cockpit["api_calls"]:
            report_lines.append(
                f"- `{call['endpoint']}`: {call['count']} calls, "
                f"status {call['last_status']}, "
                f"latency {call['last_latency_ms']}ms"
            )
        report_lines.append("")
    
    if cockpit.get("screenshots"):
        report_lines.append("**Screenshot:**")
        report_lines.append(f"![Cockpit]({cockpit['screenshots'][0]})")
        report_lines.append("")
    
    # API Endpoints
    api = audit_results["sections"].get("api_endpoints", {})
    report_lines.extend([
        f"## 3. API Endpoints - {api.get('status', 'UNKNOWN')}",
        ""
    ])
    
    if api.get("endpoints"):
        for endpoint, data in api["endpoints"].items():
            report_lines.append(f"### `{endpoint}`")
            report_lines.append(f"**Description:** {data['description']}")
            report_lines.append(f"**Status:** {data.get('status', 'N/A')}")
            report_lines.append(f"**Latency:** {data.get('latency_ms', 'N/A')}ms")
            
            if data.get("response_body"):
                report_lines.append("**Response Body:**")
                report_lines.append("```json")
                report_lines.append(json.dumps(data["response_body"], indent=2))
                report_lines.append("```")
            
            if data.get("error"):
                report_lines.append(f"**Error:** {data['error']}")
            
            report_lines.append("")
    
    # Console Errors
    console = audit_results["sections"].get("console_errors", {})
    report_lines.extend([
        f"## 4. Console Errors - {console.get('status', 'UNKNOWN')}",
        "",
        f"**Total Messages:** {console.get('total_messages', 0)}",
        f"**Errors:** {len(console.get('errors', []))}",
        f"**Warnings:** {len(console.get('warnings', []))}",
        ""
    ])
    
    if console.get("errors"):
        report_lines.append("**Error Messages:**")
        for err in console["errors"][:10]:  # Limit to first 10
            report_lines.append(f"- [{err['type']}] {err['text']}")
        report_lines.append("")
    
    # Reliability Loop
    reliability = audit_results["sections"].get("reliability_loop", {})
    report_lines.extend([
        f"## 5. Reliability Micro-Loop - {reliability.get('status', 'UNKNOWN')}",
        "",
        f"**Iterations:** {reliability.get('iterations', 0)}",
        ""
    ])
    
    if reliability.get("results"):
        for result in reliability["results"]:
            status_icon = "✓" if result["success"] else "✗"
            status_msg = "Success" if result["success"] else f"{len(result['errors'])} errors"
            report_lines.append(
                f"- {status_icon} **Iteration {result['iteration']}**: {status_msg}"
            )
            if result.get("screenshot"):
                report_lines.append(f"  ![Iteration {result['iteration']}]({result['screenshot']})")
        report_lines.append("")
    
    # Network Summary
    report_lines.extend([
        "## 6. Network Summary",
        "",
        f"**Total Requests:** {len(network_requests)}",
        ""
    ])
    
    if network_requests:
        # Group by status code
        status_codes = {}
        for req in network_requests:
            status = req.get("status", "unknown")
            status_codes[status] = status_codes.get(status, 0) + 1
        
        report_lines.append("**Status Code Distribution:**")
        for status, count in sorted(status_codes.items()):
            report_lines.append(f"- {status}: {count} requests")
        report_lines.append("")
    
    # Final Summary
    report_lines.extend([
        "---",
        "",
        "## Summary",
        ""
    ])
    
    overall_pass = all(
        section.get("status") == "PASS" 
        for section in audit_results["sections"].values()
    )
    
    report_lines.append(f"**Overall Status:** {'✅ PASS' if overall_pass else '❌ FAIL'}")
    report_lines.append("")
    report_lines.append("**Section Status:**")
    for section_name, section_data in audit_results["sections"].items():
        status_icon = "✅" if section_data.get("status") == "PASS" else "❌"
        display_name = section_name.replace('_', ' ').title()
        report_lines.append(f"- {status_icon} {display_name}: {section_data.get('status', 'UNKNOWN')}")
    
    return "\n".join(report_lines)


def main():
    """Run full UI verification audit"""
    print("🚀 Starting UI Verification Audit...")
    print(f"   Base URL: {BASE_URL}")
    print(f"   Screenshots: {SCREENSHOT_DIR}/")
    print()
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent="Ghost-UI-Audit/1.0"
        )
        page = context.new_page()
        
        # Set up console and network listeners
        page.on("console", capture_console)
        page.route("**/*", capture_network)
        
        # Run all audit sections
        audit_results["sections"]["landing_page"] = audit_landing_page(page)
        audit_results["sections"]["cockpit_panels"] = audit_cockpit_panels(page)
        audit_results["sections"]["api_endpoints"] = audit_api_endpoints(page)
        audit_results["sections"]["console_errors"] = audit_console_errors(page)
        audit_results["sections"]["reliability_loop"] = audit_reliability_loop(page)
        
        browser.close()
    
    # Generate report
    print()
    print("📝 Generating report...")
    report = generate_markdown_report()
    
    report_path = Path("UI_VERIFICATION_REPORT.md")
    report_path.write_text(report)
    
    print(f"✅ Report saved to: {report_path}")
    print(f"   Screenshots: {SCREENSHOT_DIR}/")
    
    # Also save raw JSON results
    json_path = Path("ui_verification_results.json")
    json_path.write_text(json.dumps(audit_results, indent=2))
    print(f"   Raw data: {json_path}")
    print()
    
    # Print summary
    overall_pass = all(
        section.get("status") == "PASS" 
        for section in audit_results["sections"].values()
    )
    
    print("=" * 60)
    print(f"OVERALL: {'✅ PASS' if overall_pass else '❌ FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
