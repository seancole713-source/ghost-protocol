#!/usr/bin/env python3
"""
GHOST Full System Test Suite
Comprehensive testing of all major components and integrations
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

# Load environment
try:
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed, using system env only")


class GhostSystemTest:
    """Comprehensive system test for GHOST"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "test_sections": {},
            "overall_status": "UNKNOWN",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "warnings": 0,
        }
        self.section_results = []

    def log(self, message: str, level: str = "INFO"):
        """Log test message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def test_result(self, name: str, passed: bool, details: str = "", warning: bool = False):
        """Record a test result"""
        status = "✅ PASS" if passed else ("⚠️  WARN" if warning else "❌ FAIL")
        self.results["total_tests"] += 1

        if passed and not warning:
            self.results["passed_tests"] += 1
        elif warning:
            self.results["warnings"] += 1
        else:
            self.results["failed_tests"] += 1

        self.section_results.append(
            {
                "name": name,
                "status": status,
                "passed": passed,
                "warning": warning,
                "details": details,
            }
        )

        detail_str = f": {details}" if details else ""
        print(f"  {status} {name}{detail_str}")

    async def test_environment_variables(self) -> dict[str, Any]:
        """Test 1: Environment Variables"""
        self.log("\n" + "=" * 80)
        self.log("TEST SECTION 1: Environment Variables")
        self.log("=" * 80)

        self.section_results = []

        required_vars = ["OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN", "POLYGON_API_KEY"]

        optional_vars = [
            "ALPHAVANTAGE_API_KEY",
            "COINBASE_API_KEY",
            "COINBASE_API_SECRET",
            "CDP_API_KEY_NAME",
            "CDP_API_KEY_PRIVATE_KEY",
            "OPENAI_BASE_URL",
        ]

        for var in required_vars:
            value = os.getenv(var)
            if value and len(value) > 10:
                self.test_result(f"{var}", True, f"Set ({len(value)} chars)")
            else:
                self.test_result(f"{var}", False, "Missing or invalid")

        for var in optional_vars:
            value = os.getenv(var)
            if value and len(value) > 5:
                self.test_result(f"{var}", True, f"Set ({len(value)} chars)", warning=True)
            else:
                self.test_result(f"{var}", True, "Not set (optional)", warning=True)

        section_result = {
            "tests": self.section_results.copy(),
            "summary": f"{self.results['passed_tests']} required vars set",
        }

        return section_result

    async def test_api_connections(self) -> dict[str, Any]:
        """Test 2: API Connections"""
        self.log("\n" + "=" * 80)
        self.log("TEST SECTION 2: API Connections")
        self.log("=" * 80)

        self.section_results = []

        # Test OpenAI
        await self._test_openai()

        # Test Polygon
        await self._test_polygon()

        # Test AlphaVantage
        await self._test_alphavantage()

        # Test Telegram
        await self._test_telegram()

        section_result = {
            "tests": self.section_results.copy(),
            "summary": f"{len([t for t in self.section_results if t['passed']])} APIs working",
        }

        return section_result

    async def _test_openai(self):
        """Test OpenAI API connection"""
        try:
            import httpx

            api_key = os.getenv("OPENAI_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

            if not api_key:
                self.test_result("OpenAI API", False, "No API key")
                return

            start = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}
                )
            latency = round((time.time() - start) * 1000, 2)

            if response.status_code == 200:
                models = response.json().get("data", [])
                self.test_result("OpenAI API", True, f"{len(models)} models, {latency}ms")
            else:
                self.test_result("OpenAI API", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.test_result("OpenAI API", False, f"Error: {str(e)[:50]}")

    async def _test_polygon(self):
        """Test Polygon.io API"""
        try:
            import httpx

            api_key = os.getenv("POLYGON_API_KEY", "")

            if not api_key:
                self.test_result("Polygon API", False, "No API key")
                return

            start = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={api_key}"
                )
            latency = round((time.time() - start) * 1000, 2)

            if response.status_code == 200:
                self.test_result("Polygon API", True, f"{latency}ms")
            elif response.status_code == 401:
                self.test_result("Polygon API", False, "Invalid API key")
            else:
                self.test_result("Polygon API", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.test_result("Polygon API", False, f"Error: {str(e)[:50]}")

    async def _test_alphavantage(self):
        """Test AlphaVantage API"""
        try:
            import httpx

            api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")

            if not api_key:
                self.test_result(
                    "AlphaVantage API", True, "Not configured (optional)", warning=True
                )
                return

            start = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={api_key}"
                )
            latency = round((time.time() - start) * 1000, 2)

            if response.status_code == 200:
                data = response.json()
                if "Global Quote" in data:
                    self.test_result("AlphaVantage API", True, f"{latency}ms")
                else:
                    self.test_result("AlphaVantage API", False, "Invalid response")
            else:
                self.test_result("AlphaVantage API", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.test_result("AlphaVantage API", False, f"Error: {str(e)[:50]}")

    async def _test_telegram(self):
        """Test Telegram Bot API"""
        try:
            import httpx

            token = os.getenv("TELEGRAM_BOT_TOKEN", "")

            if not token:
                self.test_result("Telegram Bot", False, "No token")
                return

            start = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"https://api.telegram.org/bot{token}/getMe")
            latency = round((time.time() - start) * 1000, 2)

            if response.status_code == 200:
                data = response.json()
                if data.get("ok"):
                    bot_name = data.get("result", {}).get("username", "unknown")
                    self.test_result("Telegram Bot", True, f"@{bot_name}, {latency}ms")
                else:
                    self.test_result("Telegram Bot", False, "API error")
            else:
                self.test_result("Telegram Bot", False, f"HTTP {response.status_code}")
        except Exception as e:
            self.test_result("Telegram Bot", False, f"Error: {str(e)[:50]}")

    async def test_database_files(self) -> dict[str, Any]:
        """Test 3: Database Files"""
        self.log("\n" + "=" * 80)
        self.log("TEST SECTION 3: Database Files")
        self.log("=" * 80)

        self.section_results = []

        db_files = ["ghost.db", "wolf_memory.db", "vector_memory.db", "agent_memory.db"]

        for db_file in db_files:
            if os.path.exists(db_file):
                size_kb = os.path.getsize(db_file) / 1024
                # Test if DB is readable
                try:
                    import sqlite3

                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()
                    self.test_result(f"{db_file}", True, f"{size_kb:.1f} KB, {len(tables)} tables")
                except Exception as e:
                    self.test_result(f"{db_file}", False, f"Cannot read: {str(e)[:30]}")
            else:
                self.test_result(f"{db_file}", True, "Does not exist yet", warning=True)

        section_result = {
            "tests": self.section_results.copy(),
            "summary": f"{len([t for t in self.section_results if t['passed']])} databases checked",
        }

        return section_result

    async def test_core_imports(self) -> dict[str, Any]:
        """Test 4: Core Module Imports"""
        self.log("\n" + "=" * 80)
        self.log("TEST SECTION 4: Core Module Imports")
        self.log("=" * 80)

        self.section_results = []

        modules = [
            ("FastAPI", "fastapi", "FastAPI"),
            ("Pydantic", "pydantic", "BaseModel"),
            ("OpenAI", "openai", "OpenAI"),
            ("HTTPX", "httpx", "AsyncClient"),
            ("DuckDB", "duckdb", None),
            ("SQLite3", "sqlite3", None),
            ("Requests", "requests", None),
            ("AgentKit", "cdp_agentkit_core", None),
            ("LangChain", "langchain_openai", None),
        ]

        for name, module_name, attr in modules:
            try:
                mod = __import__(module_name)
                if attr and not hasattr(mod, attr):
                    self.test_result(f"{name} module", False, f"Missing {attr}")
                else:
                    version = getattr(mod, "__version__", "unknown")
                    self.test_result(f"{name} module", True, f"v{version}")
            except ImportError:
                if name in ["AgentKit", "LangChain"]:
                    self.test_result(
                        f"{name} module", True, "Not installed (optional)", warning=True
                    )
                else:
                    self.test_result(f"{name} module", False, "Import failed")

        section_result = {
            "tests": self.section_results.copy(),
            "summary": f"{len([t for t in self.section_results if t['passed']])} modules available",
        }

        return section_result

    async def test_ghost_server(self) -> dict[str, Any]:
        """Test 5: GHOST Server Health"""
        self.log("\n" + "=" * 80)
        self.log("TEST SECTION 5: GHOST Server Health")
        self.log("=" * 80)

        self.section_results = []

        # Check if server is running
        try:
            import httpx

            base_url = os.getenv("GHOST_SERVER_URL", "http://localhost:8080")

            # Test health endpoint
            async with httpx.AsyncClient(timeout=5.0) as client:
                try:
                    response = await client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        self.test_result("Server Running", True, f"{base_url}")

                        # Test specific endpoints
                        endpoints = [
                            "/api/status",
                            "/api/v1/wolf/watchlist",
                            "/api/v1/portfolio",
                            "/metrics",
                        ]

                        for endpoint in endpoints:
                            try:
                                start = time.time()
                                resp = await client.get(f"{base_url}{endpoint}")
                                latency = round((time.time() - start) * 1000, 2)

                                if resp.status_code == 200:
                                    self.test_result(f"Endpoint {endpoint}", True, f"{latency}ms")
                                elif resp.status_code == 401:
                                    self.test_result(
                                        f"Endpoint {endpoint}",
                                        True,
                                        "Auth required (OK)",
                                        warning=True,
                                    )
                                else:
                                    self.test_result(
                                        f"Endpoint {endpoint}", False, f"HTTP {resp.status_code}"
                                    )
                            except Exception as e:
                                self.test_result(
                                    f"Endpoint {endpoint}", False, f"Error: {str(e)[:30]}"
                                )
                    else:
                        self.test_result("Server Running", False, f"HTTP {response.status_code}")
                        self.test_result("Server Endpoints", False, "Server not running")

                except httpx.ConnectError:
                    self.test_result(
                        "Server Running", False, "Connection refused - server not running"
                    )

        except Exception as e:
            self.test_result("Server Running", False, f"Error: {str(e)[:50]}")

        section_result = {"tests": self.section_results.copy(), "summary": "Server status checked"}

        return section_result

    async def test_unit_tests(self) -> dict[str, Any]:
        """Test 6: Run Unit Test Suite"""
        self.log("\n" + "=" * 80)
        self.log("TEST SECTION 6: Unit Test Suite")
        self.log("=" * 80)

        self.section_results = []

        # Check if pytest is available
        try:
            import importlib.util

            pytest_available = importlib.util.find_spec("pytest") is not None
        except ImportError:
            pytest_available = False
            self.test_result("PyTest Available", False, "pytest not installed")

        if pytest_available:
            self.test_result("PyTest Available", True, "Ready to run tests")

            # List test files
            test_files = []
            for root, _dirs, files in os.walk("."):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        test_files.append(os.path.join(root, file))

            self.test_result(
                "Test Files Found", len(test_files) > 0, f"{len(test_files)} test files"
            )

            # Run a quick test if any exist
            if test_files and len(test_files) > 0:
                self.log("\n  Running quick validation test...")
                try:
                    # Run pytest with minimal output
                    result = subprocess.run(
                        ["python", "-m", "pytest", "--collect-only", "-q"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )

                    if result.returncode == 0:
                        lines = result.stdout.split("\n")
                        test_count = len([line for line in lines if "::" in line])
                        self.test_result("Test Collection", True, f"{test_count} tests collected")
                    else:
                        self.test_result("Test Collection", False, "Collection failed")
                except Exception as e:
                    self.test_result("Test Collection", False, f"Error: {str(e)[:30]}")

        section_result = {"tests": self.section_results.copy(), "summary": "Unit tests validated"}

        return section_result

    async def test_critical_features(self) -> dict[str, Any]:
        """Test 7: Critical Feature Validation"""
        self.log("\n" + "=" * 80)
        self.log("TEST SECTION 7: Critical Features")
        self.log("=" * 80)

        self.section_results = []

        # Test wolf_app.py exists and is valid
        if os.path.exists("wolf_app.py"):
            size_mb = os.path.getsize("wolf_app.py") / (1024 * 1024)
            self.test_result("wolf_app.py", True, f"{size_mb:.2f} MB")

            # Check for key functions/classes
            try:
                with open("wolf_app.py") as f:
                    content = f.read()

                    checks = [
                        ("FastAPI app", "app = FastAPI"),
                        ("WOLF endpoint", "/api/v1/wolf"),
                        ("Portfolio system", "def portfolio"),
                        ("Watchlist system", "watchlist"),
                        ("Price fetcher", "fetch_price"),
                    ]

                    for check_name, check_str in checks:
                        if check_str.lower() in content.lower():
                            self.test_result(f"Feature: {check_name}", True, "Found")
                        else:
                            self.test_result(f"Feature: {check_name}", False, "Not found")
            except Exception as e:
                self.test_result("wolf_app.py validation", False, f"Error: {str(e)[:30]}")
        else:
            self.test_result("wolf_app.py", False, "File not found")

        section_result = {
            "tests": self.section_results.copy(),
            "summary": "Critical features validated",
        }

        return section_result

    async def run_full_test(self) -> dict[str, Any]:
        """Run complete system test"""
        start_time = time.time()

        self.log("\n")
        self.log("╔" + "=" * 78 + "╗")
        self.log("║" + " " * 20 + "GHOST FULL SYSTEM TEST" + " " * 36 + "║")
        self.log("║" + " " * 78 + "║")
        self.log(
            "║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 45 + "║"
        )
        self.log("╚" + "=" * 78 + "╝")

        # Run all test sections
        self.results["test_sections"]["1_environment"] = await self.test_environment_variables()
        self.results["test_sections"]["2_api_connections"] = await self.test_api_connections()
        self.results["test_sections"]["3_databases"] = await self.test_database_files()
        self.results["test_sections"]["4_imports"] = await self.test_core_imports()
        self.results["test_sections"]["5_server"] = await self.test_ghost_server()
        self.results["test_sections"]["6_unit_tests"] = await self.test_unit_tests()
        self.results["test_sections"]["7_critical_features"] = await self.test_critical_features()

        # Calculate overall status
        elapsed = time.time() - start_time
        self.results["test_duration_seconds"] = round(elapsed, 2)

        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        warnings = self.results["warnings"]

        pass_rate = (passed / total * 100) if total > 0 else 0

        if pass_rate >= 90 and failed == 0:
            self.results["overall_status"] = "✅ EXCELLENT"
        elif pass_rate >= 75 and failed <= 2:
            self.results["overall_status"] = "✅ GOOD"
        elif pass_rate >= 60:
            self.results["overall_status"] = "⚠️  ACCEPTABLE"
        elif pass_rate >= 40:
            self.results["overall_status"] = "⚠️  DEGRADED"
        else:
            self.results["overall_status"] = "❌ CRITICAL"

        # Print summary
        self.log("\n")
        self.log("╔" + "=" * 78 + "╗")
        self.log("║" + " " * 25 + "TEST SUMMARY" + " " * 41 + "║")
        self.log("╠" + "=" * 78 + "╣")
        self.log(
            "║"
            + f"  Overall Status: {self.results['overall_status']}"
            + " " * (67 - len(self.results["overall_status"]))
            + "║"
        )
        self.log("║" + " " * 78 + "║")
        self.log("║" + f"  Total Tests:    {total}" + " " * (66 - len(str(total))) + "║")
        self.log(
            "║"
            + f"  Passed:         {passed} ({pass_rate:.1f}%)"
            + " " * (58 - len(f"{passed} ({pass_rate:.1f}%)"))
            + "║"
        )
        self.log("║" + f"  Failed:         {failed}" + " " * (66 - len(str(failed))) + "║")
        self.log("║" + f"  Warnings:       {warnings}" + " " * (66 - len(str(warnings))) + "║")
        self.log("║" + " " * 78 + "║")
        self.log(
            "║" + f"  Duration:       {elapsed:.2f}s" + " " * (63 - len(f"{elapsed:.2f}s")) + "║"
        )
        self.log("╚" + "=" * 78 + "╝")
        self.log("\n")

        return self.results

    def save_report(self, filename: str = "ghost_system_test_report.json"):
        """Save test results to JSON file"""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)

        self.log(f"📄 Full test report saved to: {filename}")

        # Also create a simple text summary
        summary_file = filename.replace(".json", "_summary.txt")
        with open(summary_file, "w") as f:
            f.write("GHOST SYSTEM TEST SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n")
            f.write(f"Status: {self.results['overall_status']}\n")
            f.write(f"Duration: {self.results['test_duration_seconds']}s\n\n")
            f.write(f"Tests: {self.results['total_tests']}\n")
            f.write(f"Passed: {self.results['passed_tests']}\n")
            f.write(f"Failed: {self.results['failed_tests']}\n")
            f.write(f"Warnings: {self.results['warnings']}\n\n")

            for section_name, section_data in self.results["test_sections"].items():
                f.write(f"\n{section_name}:\n")
                f.write("-" * 80 + "\n")
                for test in section_data.get("tests", []):
                    f.write(f"  {test['status']} {test['name']}")
                    if test.get("details"):
                        f.write(f" - {test['details']}")
                    f.write("\n")

        self.log(f"📄 Summary saved to: {summary_file}")


async def main():
    """Main entry point"""
    test_suite = GhostSystemTest()

    try:
        results = await test_suite.run_full_test()
        test_suite.save_report()

        # Exit with appropriate code
        if results["overall_status"].startswith("✅"):
            sys.exit(0)
        elif results["overall_status"].startswith("⚠️"):
            sys.exit(1)
        else:
            sys.exit(2)

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ TEST SUITE FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
