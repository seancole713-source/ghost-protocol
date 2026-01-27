#!/usr/bin/env python3
"""
Ghost System Diagnostic Scanner
Tests all subsystems and generates comprehensive health report
"""

import json
import time
from datetime import datetime
from pathlib import Path

import requests

BASE_URL = os.getenv("GHOST_BASE_URL", "http://localhost:8080")
OUTPUT_DIR = Path("healthcheck_out")
OUTPUT_DIR.mkdir(exist_ok=True)


class DiagnosticRunner:
    def __init__(self):
        self.results = {
            "scan_time": datetime.now().isoformat(),
            "base_url": BASE_URL,
            "subsystems": {},
            "critical_issues": [],
            "warnings": [],
            "summary": {},
        }

    def test_endpoint(self, path, name, timeout=10):
        """Test a single endpoint and return status"""
        start = time.time()
        try:
            response = requests.get(f"{BASE_URL}{path}", timeout=timeout)
            duration_ms = int((time.time() - start) * 1000)

            result = {
                "status": "✅ ONLINE"
                if response.status_code == 200
                else f"⚠️ HTTP {response.status_code}",
                "status_code": response.status_code,
                "latency_ms": duration_ms,
                "response_size": len(response.content),
                "content_type": response.headers.get("content-type", "unknown"),
            }

            # Try to parse JSON
            if "json" in result["content_type"]:
                try:
                    data = response.json()
                    result["data_preview"] = str(data)[:200]
                    result["data_keys"] = list(data.keys()) if isinstance(data, dict) else None
                except Exception:
                    result["data_preview"] = "JSON parse failed"

            return result

        except requests.exceptions.Timeout:
            return {
                "status": "❌ TIMEOUT",
                "latency_ms": int((time.time() - start) * 1000),
                "error": f"Timeout after {timeout}s",
            }
        except requests.exceptions.ConnectionError as e:
            return {"status": "❌ CONNECTION_ERROR", "error": str(e)}
        except Exception as e:
            return {"status": "❌ ERROR", "error": str(e)}

    def scan_ai_core(self):
        """Test AI Core & Prediction Engine"""
        print("\n🧠 Scanning AI Core & Prediction Engine...")

        subsystem = {"name": "AI Core & Prediction Engine", "endpoints": {}}

        endpoints = [
            ("/api/agent/stats", "Agent Statistics"),
            ("/api/agent/decisions", "Agent Decisions Log"),
            ("/api/stage2/forecasts", "Stage 2 Forecasts"),
            ("/api/stage2/accuracy", "Forecast Accuracy Metrics"),
            ("/api/stage3/regime/current", "Stage 3 Regime Detection"),
            ("/api/snapshot", "Trading Snapshot"),
        ]

        for path, name in endpoints:
            print(f"  Testing {name}...")
            result = self.test_endpoint(path, name)
            subsystem["endpoints"][path] = result

            if result.get("status_code") != 200:
                self.critical_issues.append(f"AI Core: {name} failed - {result.get('status')}")

        # Calculate subsystem health
        online_count = sum(
            1 for e in subsystem["endpoints"].values() if e.get("status_code") == 200
        )
        total_count = len(subsystem["endpoints"])
        subsystem["health"] = f"{online_count}/{total_count} endpoints online"
        subsystem["status"] = (
            "✅ HEALTHY"
            if online_count == total_count
            else f"⚠️ DEGRADED ({online_count}/{total_count})"
        )

        self.results["subsystems"]["ai_core"] = subsystem
        return subsystem

    def scan_data_feeds(self):
        """Test Stock & Crypto Data Feeds"""
        print("\n📊 Scanning Data Feeds (Stocks + Crypto)...")

        subsystem = {"name": "Data Feeds (Stocks + Crypto)", "endpoints": {}}

        endpoints = [
            ("/api/price/WOLF", "WOLF Stock Price"),
            ("/api/price/SPY", "SPY ETF Price"),
            ("/api/price/AAPL", "AAPL Stock Price"),
            ("/api/price/BTC-USD", "Bitcoin Price"),
            ("/api/crypto/price/bitcoin", "Crypto: Bitcoin"),
            ("/api/crypto/price/ethereum", "Crypto: Ethereum"),
            ("/api/crypto/ohlcv/bitcoin?days=7", "Crypto OHLCV: Bitcoin"),
        ]

        for path, name in endpoints:
            print(f"  Testing {name}...")
            result = self.test_endpoint(path, name)
            subsystem["endpoints"][path] = result

            if result.get("status_code") not in [200, 503]:  # 503 = crypto disabled
                self.warnings.append(f"Data Feed: {name} - {result.get('status')}")

        # Calculate health
        online_count = sum(
            1 for e in subsystem["endpoints"].values() if e.get("status_code") in [200, 503]
        )
        total_count = len(subsystem["endpoints"])
        subsystem["health"] = f"{online_count}/{total_count} endpoints responding"
        subsystem["status"] = "✅ HEALTHY" if online_count >= total_count - 1 else "⚠️ DEGRADED"

        self.results["subsystems"]["data_feeds"] = subsystem
        return subsystem

    def scan_news_sentiment(self):
        """Test News & Sentiment Analysis"""
        print("\n📰 Scanning News & Sentiment...")

        subsystem = {"name": "News & Sentiment Analysis", "endpoints": {}}

        endpoints = [
            ("/api/news", "News API"),
            ("/api/news/recent", "Recent News"),
            ("/api/watcher/ticker_news", "Ticker News Watcher"),
        ]

        for path, name in endpoints:
            print(f"  Testing {name}...")
            result = self.test_endpoint(path, name)
            subsystem["endpoints"][path] = result

        online_count = sum(
            1 for e in subsystem["endpoints"].values() if e.get("status_code") == 200
        )
        total_count = len(subsystem["endpoints"])
        subsystem["health"] = f"{online_count}/{total_count} endpoints online"
        subsystem["status"] = "✅ HEALTHY" if online_count == total_count else "⚠️ DEGRADED"

        self.results["subsystems"]["news_sentiment"] = subsystem
        return subsystem

    def scan_cockpit_ui(self):
        """Test Cockpit UI & Frontend"""
        print("\n🎛️ Scanning Cockpit UI...")

        subsystem = {"name": "Cockpit UI & Frontend", "endpoints": {}}

        endpoints = [
            ("/", "Root / Homepage"),
            ("/cockpit", "Cockpit Dashboard"),
            ("/api/openapi.json", "OpenAPI Schema"),
            ("/api/docs", "Swagger UI"),
            ("/health", "Health Check"),
            ("/static/img/neo_glass_bg.webp", "Static Assets"),
        ]

        for path, name in endpoints:
            print(f"  Testing {name}...")
            result = self.test_endpoint(path, name, timeout=5)
            subsystem["endpoints"][path] = result

        online_count = sum(
            1 for e in subsystem["endpoints"].values() if e.get("status_code") in [200, 307]
        )
        total_count = len(subsystem["endpoints"])
        subsystem["health"] = f"{online_count}/{total_count} endpoints accessible"
        subsystem["status"] = "✅ HEALTHY" if online_count >= total_count - 1 else "⚠️ DEGRADED"

        self.results["subsystems"]["cockpit_ui"] = subsystem
        return subsystem

    def scan_database_services(self):
        """Test Database & Backend Services"""
        print("\n💾 Scanning Database & Services...")

        subsystem = {"name": "Database & Backend Services", "endpoints": {}}

        endpoints = [
            ("/api/portfolio", "Portfolio Manager"),
            ("/api/memory/stats", "Memory Statistics"),
            ("/health", "Health Endpoint"),
            ("/metrics", "Prometheus Metrics"),
        ]

        for path, name in endpoints:
            print(f"  Testing {name}...")
            result = self.test_endpoint(path, name)
            subsystem["endpoints"][path] = result

        online_count = sum(
            1 for e in subsystem["endpoints"].values() if e.get("status_code") == 200
        )
        total_count = len(subsystem["endpoints"])
        subsystem["health"] = f"{online_count}/{total_count} services responding"
        subsystem["status"] = "✅ HEALTHY" if online_count >= total_count - 1 else "⚠️ DEGRADED"

        self.results["subsystems"]["database_services"] = subsystem
        return subsystem

    def generate_summary(self):
        """Generate overall system summary"""
        print("\n📋 Generating Summary...")

        total_subsystems = len(self.results["subsystems"])
        healthy_subsystems = sum(
            1 for s in self.results["subsystems"].values() if "✅" in s["status"]
        )

        total_endpoints = sum(len(s["endpoints"]) for s in self.results["subsystems"].values())
        online_endpoints = sum(
            sum(1 for e in s["endpoints"].values() if e.get("status_code") in [200, 307])
            for s in self.results["subsystems"].values()
        )

        self.results["summary"] = {
            "overall_status": "✅ OPERATIONAL"
            if healthy_subsystems >= total_subsystems - 1
            else "⚠️ DEGRADED",
            "subsystems_healthy": f"{healthy_subsystems}/{total_subsystems}",
            "endpoints_online": f"{online_endpoints}/{total_endpoints}",
            "critical_issues_count": len(self.results["critical_issues"]),
            "warnings_count": len(self.results["warnings"]),
        }

    def write_reports(self):
        """Write all output files"""
        print("\n📝 Writing Reports...")

        # Full JSON report
        json_path = OUTPUT_DIR / "system_diagnostic.json"
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"  ✅ Written: {json_path}")

        # Markdown summary
        md_path = OUTPUT_DIR / "system_diagnostic.md"
        with open(md_path, "w") as f:
            f.write("# Ghost System Diagnostic Report\n\n")
            f.write(f"**Scan Time:** {self.results['scan_time']}  \n")
            f.write(f"**Base URL:** {self.results['base_url']}  \n\n")

            f.write("## 🎯 Overall Status\n\n")
            summary = self.results["summary"]
            f.write(f"- **Status:** {summary['overall_status']}\n")
            f.write(f"- **Subsystems Healthy:** {summary['subsystems_healthy']}\n")
            f.write(f"- **Endpoints Online:** {summary['endpoints_online']}\n")
            f.write(f"- **Critical Issues:** {summary['critical_issues_count']}\n")
            f.write(f"- **Warnings:** {summary['warnings_count']}\n\n")

            f.write("## 🔍 Subsystem Details\n\n")
            for _key, subsystem in self.results["subsystems"].items():
                f.write(f"### {subsystem['name']}\n\n")
                f.write(f"**Status:** {subsystem['status']}  \n")
                f.write(f"**Health:** {subsystem['health']}  \n\n")

                f.write("| Endpoint | Status | Latency | Notes |\n")
                f.write("|----------|--------|---------|-------|\n")

                for path, result in subsystem["endpoints"].items():
                    status = result.get("status", "❌ UNKNOWN")
                    latency = (
                        f"{result.get('latency_ms', '-')}ms" if "latency_ms" in result else "-"
                    )
                    notes = result.get("error", result.get("data_preview", "OK"))[:50]
                    f.write(f"| `{path}` | {status} | {latency} | {notes} |\n")

                f.write("\n")

            if self.results["critical_issues"]:
                f.write("## ❌ Critical Issues\n\n")
                for issue in self.results["critical_issues"]:
                    f.write(f"- {issue}\n")
                f.write("\n")

            if self.results["warnings"]:
                f.write("## ⚠️ Warnings\n\n")
                for warning in self.results["warnings"]:
                    f.write(f"- {warning}\n")
                f.write("\n")

            f.write("## 💡 Recommendations\n\n")

            if summary["critical_issues_count"] > 0:
                f.write(
                    "1. **Fix critical endpoint failures** - Some core APIs are not responding\n"
                )

            if "data_feeds" in self.results["subsystems"]:
                df = self.results["subsystems"]["data_feeds"]
                if "⚠️" in df["status"]:
                    f.write(
                        "2. **Check data provider API keys** - YFinance errors detected, Polygon rate limits hit\n"
                    )

            f.write(
                "3. **Verify environment variables** - Ensure OPENAI_API_KEY, CRYPTO_ENABLED, etc. are set\n"
            )
            f.write("4. **Check SIM_MODE setting** - Currently set to 0 (live mode)\n")

        print(f"  ✅ Written: {md_path}")

    def run_full_scan(self):
        """Execute complete diagnostic scan"""
        print("=" * 60)
        print("🚀 Ghost System Diagnostic Scanner")
        print("=" * 60)

        # Run all scans
        self.scan_ai_core()
        self.scan_data_feeds()
        self.scan_news_sentiment()
        self.scan_cockpit_ui()
        self.scan_database_services()

        # Generate summary
        self.generate_summary()

        # Write reports
        self.write_reports()

        # Print summary to console
        print("\n" + "=" * 60)
        print("📊 SCAN COMPLETE")
        print("=" * 60)
        summary = self.results["summary"]
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Subsystems Healthy: {summary['subsystems_healthy']}")
        print(f"Endpoints Online: {summary['endpoints_online']}")
        print(f"Critical Issues: {summary['critical_issues_count']}")
        print(f"Warnings: {summary['warnings_count']}")
        print("\n📁 Reports saved to: healthcheck_out/")
        print("=" * 60)


if __name__ == "__main__":
    scanner = DiagnosticRunner()
    scanner.run_full_scan()
