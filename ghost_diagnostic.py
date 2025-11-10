#!/usr/bin/env python3
"""
Ghost Protocol Diagnostic - Complete System Health Check
Tests all APIs, connections, and services before auto-recovery
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Any

# Load environment
try:
    from dotenv import load_dotenv

    load_dotenv("secrets.env")
except ImportError:
    print("⚠️  python-dotenv not installed, using system env only")


class GhostDiagnostic:
    """Comprehensive diagnostic suite for Ghost system"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "api_connections": {},
            "agentkit_status": {},
            "data_feeds": {},
            "telegram": {},
            "ui_sync": {},
            "vector_memory": {},
            "gpt_reasoning": {},
            "overall_health": "UNKNOWN",
        }

    def check_env_var(self, key: str, required: bool = True) -> tuple[str, str]:
        """Check if environment variable exists and is valid"""
        value = os.getenv(key, "")

        if not value:
            status = "❌ MISSING" if required else "⚠️  NOT SET"
            return (status, "")

        # Mask sensitive values
        if "KEY" in key or "TOKEN" in key:
            masked = f"***{value[-4:]}" if len(value) > 4 else "***"
            return ("✅ SET", masked)

        return ("✅ SET", value)

    async def test_openai_connection(self) -> dict[str, Any]:
        """Test OpenAI API connection"""
        import httpx

        api_key = os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        if not api_key:
            return {"status": "❌ FAILED", "error": "OPENAI_API_KEY not set", "latency_ms": None}

        try:
            start = datetime.now()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}
                )

            latency = (datetime.now() - start).total_seconds() * 1000

            if response.status_code == 200:
                models = response.json().get("data", [])
                model_names = [m.get("id") for m in models[:5]]

                return {
                    "status": "✅ ACTIVE",
                    "latency_ms": round(latency, 2),
                    "models_available": len(models),
                    "sample_models": model_names,
                    "base_url": base_url,
                }
            elif response.status_code == 401:
                return {
                    "status": "❌ AUTH_FAILED",
                    "error": "Invalid API key",
                    "status_code": 401,
                    "latency_ms": round(latency, 2),
                }
            else:
                return {
                    "status": "⚠️  ERROR",
                    "error": f"HTTP {response.status_code}",
                    "latency_ms": round(latency, 2),
                }

        except httpx.TimeoutException:
            return {"status": "❌ TIMEOUT", "error": "Request timed out after 10s"}
        except Exception as e:
            return {"status": "❌ FAILED", "error": str(e)}

    async def test_polygon_api(self) -> dict[str, Any]:
        """Test Polygon.io API connection"""
        import httpx

        api_key = os.getenv("POLYGON_API_KEY", "")

        if not api_key:
            return {"status": "❌ FAILED", "error": "POLYGON_API_KEY not set"}

        try:
            start = datetime.now()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://api.polygon.io/v2/aggs/ticker/WOLF/prev?apiKey={api_key}"
                )

            latency = (datetime.now() - start).total_seconds() * 1000

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])

                return {
                    "status": "✅ ACTIVE",
                    "latency_ms": round(latency, 2),
                    "sample_data": results[0] if results else None,
                }
            elif response.status_code == 401 or response.status_code == 403:
                return {
                    "status": "❌ AUTH_FAILED",
                    "error": "Invalid API key or unauthorized",
                    "status_code": response.status_code,
                }
            else:
                return {
                    "status": "⚠️  ERROR",
                    "error": f"HTTP {response.status_code}",
                    "latency_ms": round(latency, 2),
                }

        except Exception as e:
            return {"status": "❌ FAILED", "error": str(e)}

    async def test_alphavantage_api(self) -> dict[str, Any]:
        """Test AlphaVantage API connection"""
        import httpx

        api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")

        if not api_key:
            return {"status": "❌ FAILED", "error": "ALPHAVANTAGE_API_KEY not set"}

        try:
            start = datetime.now()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=WOLF&apikey={api_key}"
                )

            latency = (datetime.now() - start).total_seconds() * 1000

            if response.status_code == 200:
                data = response.json()

                if "Global Quote" in data and data["Global Quote"]:
                    return {
                        "status": "✅ ACTIVE",
                        "latency_ms": round(latency, 2),
                        "sample_quote": data["Global Quote"],
                    }
                elif "Note" in data:
                    return {
                        "status": "⚠️  RATE_LIMITED",
                        "error": "API call frequency exceeded",
                        "message": data["Note"],
                    }
                else:
                    return {"status": "⚠️  NO_DATA", "response": data}
            else:
                return {"status": "⚠️  ERROR", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"status": "❌ FAILED", "error": str(e)}

    async def test_telegram_bot(self) -> dict[str, Any]:
        """Test Telegram bot connection"""
        import httpx

        bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

        if not bot_token:
            return {"status": "❌ FAILED", "error": "TELEGRAM_BOT_TOKEN not set"}

        if not chat_id:
            return {"status": "❌ FAILED", "error": "TELEGRAM_CHAT_ID not set"}

        try:
            # Test getMe endpoint
            start = datetime.now()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"https://api.telegram.org/bot{bot_token}/getMe")

            latency = (datetime.now() - start).total_seconds() * 1000

            if response.status_code == 200:
                data = response.json()

                if data.get("ok"):
                    bot_info = data.get("result", {})
                    return {
                        "status": "✅ ACTIVE",
                        "latency_ms": round(latency, 2),
                        "bot_username": bot_info.get("username"),
                        "bot_name": bot_info.get("first_name"),
                        "chat_id_configured": chat_id,
                    }
                else:
                    return {
                        "status": "❌ FAILED",
                        "error": data.get("description", "Unknown error"),
                    }
            elif response.status_code == 401:
                return {"status": "❌ AUTH_FAILED", "error": "Invalid bot token"}
            else:
                return {"status": "⚠️  ERROR", "error": f"HTTP {response.status_code}"}

        except Exception as e:
            return {"status": "❌ FAILED", "error": str(e)}

    async def test_ghost_server(self) -> dict[str, Any]:
        """Test Ghost server health endpoints"""
        import httpx

        base_url = "http://localhost:5000"

        results = {"server_running": False, "endpoints": {}}

        endpoints_to_test = ["/api/health", "/api/portfolio", "/api/price/WOLF", "/agent/health"]

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                for endpoint in endpoints_to_test:
                    try:
                        start = datetime.now()
                        response = await client.get(f"{base_url}{endpoint}")
                        latency = (datetime.now() - start).total_seconds() * 1000

                        results["endpoints"][endpoint] = {
                            "status": "✅" if response.status_code == 200 else "⚠️",
                            "status_code": response.status_code,
                            "latency_ms": round(latency, 2),
                        }

                        results["server_running"] = True

                    except Exception as e:
                        results["endpoints"][endpoint] = {"status": "❌", "error": str(e)}

            return results

        except Exception as e:
            return {"server_running": False, "error": str(e)}

    def check_database_files(self) -> dict[str, Any]:
        """Check if required database files exist"""
        files_to_check = ["data/wolf.db", "data/ghost_agent.db", "ghost_state.json"]

        results = {}

        for filepath in files_to_check:
            if os.path.exists(filepath):
                size_bytes = os.path.getsize(filepath)
                size_kb = round(size_bytes / 1024, 2)

                results[filepath] = {"status": "✅ EXISTS", "size_kb": size_kb}
            else:
                results[filepath] = {"status": "❌ MISSING"}

        return results

    async def run_full_diagnostic(self) -> dict[str, Any]:
        """Run complete diagnostic suite"""

        print("=" * 80)
        print("🔍 GHOST PROTOCOL DIAGNOSTIC - FULL SYSTEM SCAN")
        print("=" * 80)
        print()

        # ========== STEP 1: Environment Variables ==========
        print("📋 STEP 1: Environment Variables")
        print("-" * 80)

        required_vars = [
            "OPENAI_API_KEY",
            "OPENAI_AGENT_API_KEY",
            "POLYGON_API_KEY",
            "ALPHAVANTAGE_API_KEY",
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_CHAT_ID",
            "GHOST_API_TOKEN",
        ]

        optional_vars = [
            "OPENAI_ORG_ID",
            "VECTOR_DB_URL",
            "VECTOR_DB_API_KEY",
            "AGENTKIT_ENABLED",
            "AGENTS_ENABLED",
            "CACHE_MODE",
            "SIM_MODE",
        ]

        env_results = {}

        for var in required_vars:
            status, value = self.check_env_var(var, required=True)
            env_results[var] = {"status": status, "value": value}
            print(f"  {status} {var}: {value}")

        print()
        print("Optional Variables:")
        for var in optional_vars:
            status, value = self.check_env_var(var, required=False)
            env_results[var] = {"status": status, "value": value}
            print(f"  {status} {var}: {value}")

        self.results["env_vars"] = env_results

        print()
        print()

        # ========== STEP 2: API Connections ==========
        print("🌐 STEP 2: API Connections")
        print("-" * 80)

        print("Testing OpenAI API...")
        openai_result = await self.test_openai_connection()
        self.results["api_connections"]["openai"] = openai_result
        print(f"  {openai_result['status']} OpenAI: {openai_result.get('latency_ms', 'N/A')}ms")
        if "models_available" in openai_result:
            print(f"     Models: {openai_result['models_available']} available")
        if "error" in openai_result:
            print(f"     Error: {openai_result['error']}")

        print()
        print("Testing Polygon.io API...")
        polygon_result = await self.test_polygon_api()
        self.results["api_connections"]["polygon"] = polygon_result
        print(f"  {polygon_result['status']} Polygon: {polygon_result.get('latency_ms', 'N/A')}ms")
        if "error" in polygon_result:
            print(f"     Error: {polygon_result['error']}")

        print()
        print("Testing AlphaVantage API...")
        alphavantage_result = await self.test_alphavantage_api()
        self.results["api_connections"]["alphavantage"] = alphavantage_result
        print(
            f"  {alphavantage_result['status']} AlphaVantage: {alphavantage_result.get('latency_ms', 'N/A')}ms"
        )
        if "error" in alphavantage_result:
            print(f"     Error: {alphavantage_result['error']}")

        print()
        print("Testing Telegram Bot...")
        telegram_result = await self.test_telegram_bot()
        self.results["telegram"] = telegram_result
        print(f"  {telegram_result['status']} Telegram")
        if "bot_username" in telegram_result:
            print(f"     Bot: @{telegram_result['bot_username']}")
        if "error" in telegram_result:
            print(f"     Error: {telegram_result['error']}")

        print()
        print()

        # ========== STEP 3: Ghost Server ==========
        print("🖥️  STEP 3: Ghost Server Health")
        print("-" * 80)

        server_result = await self.test_ghost_server()
        self.results["ghost_server"] = server_result

        if server_result["server_running"]:
            print("  ✅ Server is RUNNING")
            print()
            print("  Endpoint Status:")
            for endpoint, status in server_result["endpoints"].items():
                print(f"    {status['status']} {endpoint}: {status.get('latency_ms', 'N/A')}ms")
        else:
            print("  ❌ Server is NOT RUNNING")
            print(f"     Error: {server_result.get('error', 'Could not connect')}")

        print()
        print()

        # ========== STEP 4: Database Files ==========
        print("💾 STEP 4: Database Files")
        print("-" * 80)

        db_results = self.check_database_files()
        self.results["databases"] = db_results

        for filepath, result in db_results.items():
            print(f"  {result['status']} {filepath}")
            if "size_kb" in result:
                print(f"     Size: {result['size_kb']} KB")

        print()
        print()

        # ========== Overall Assessment ==========
        print("=" * 80)
        print("📊 OVERALL ASSESSMENT")
        print("=" * 80)

        # Calculate health score
        critical_checks = [
            ("OpenAI", self.results["api_connections"]["openai"]["status"].startswith("✅")),
            ("Telegram", self.results["telegram"]["status"].startswith("✅")),
            ("Server", self.results["ghost_server"]["server_running"]),
        ]

        important_checks = [
            ("Polygon", self.results["api_connections"]["polygon"]["status"].startswith("✅")),
            (
                "AlphaVantage",
                not self.results["api_connections"]["alphavantage"]["status"].startswith("❌"),
            ),
        ]

        critical_passed = sum(1 for _, passed in critical_checks if passed)
        important_passed = sum(1 for _, passed in important_checks if passed)

        if critical_passed == len(critical_checks) and important_passed >= 1:
            overall_health = "✅ HEALTHY"
        elif critical_passed >= 2:
            overall_health = "⚠️  DEGRADED"
        else:
            overall_health = "❌ CRITICAL"

        self.results["overall_health"] = overall_health

        print(f"\nSystem Health: {overall_health}")
        print(f"Critical Services: {critical_passed}/{len(critical_checks)} online")
        print(f"Data Feeds: {important_passed}/{len(important_checks)} working")

        print()
        print("=" * 80)

        return self.results

    def generate_report(self, output_file: str = "ghost_diagnostic_report.json"):
        """Save diagnostic results to JSON file"""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Full report saved to: {output_file}")


async def main():
    """Main diagnostic entry point"""
    diagnostic = GhostDiagnostic()

    try:
        results = await diagnostic.run_full_diagnostic()
        diagnostic.generate_report()

        # Exit code based on health
        if results["overall_health"] == "✅ HEALTHY":
            sys.exit(0)
        elif results["overall_health"] == "⚠️  DEGRADED":
            sys.exit(1)
        else:
            sys.exit(2)

    except Exception as e:
        print(f"\n❌ DIAGNOSTIC FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
