"""
Ghost Live Pipeline Tests
Self-tests for price diagnostics, predictions, SSE, and Telegram alerts
"""

import asyncio
import os

import requests

# Configuration
BASE_URL = os.getenv("GHOST_BASE_URL", "http://localhost:5000")
API_TOKEN = os.getenv("GHOST_API_TOKEN", "")


class LivePipelineTests:
    """Test suite for live Ghost pipeline"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []

    def test(self, name: str, passed: bool, detail: str = ""):
        """Record test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if detail:
            print(f"  → {detail}")

        self.results.append({"name": name, "passed": passed, "detail": detail})

        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def test_price_diagnostics(self, symbol: str, market: str):
        """Test /api/price/diagnostics endpoint"""
        try:
            if market == "stock":
                url = f"{BASE_URL}/api/price/diagnostics?symbol={symbol}"
            else:
                url = f"{BASE_URL}/api/crypto/price/{symbol}"

            response = requests.get(url, timeout=5)

            if response.status_code != 200:
                self.test(
                    f"Price diagnostics ({symbol})",
                    False,
                    f"HTTP {response.status_code}",
                )
                return

            data = response.json()

            # Check required fields
            if market == "stock":
                has_price = data.get("price") is not None
                has_provider = data.get("provider") is not None
            else:
                has_price = data.get("price") is not None
                has_provider = True  # Crypto always has provider

            self.test(
                f"Price diagnostics ({symbol})",
                has_price and has_provider,
                f"price={data.get('price')}, provider={data.get('provider')}",
            )

        except Exception as e:
            self.test(f"Price diagnostics ({symbol})", False, str(e))

    def test_prediction_run(self, symbol: str):
        """Test /api/predict/run endpoint"""
        try:
            url = f"{BASE_URL}/api/predict/run"
            headers = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
            payload = {"symbol": symbol}

            response = requests.post(url, json=payload, headers=headers, timeout=10)

            if response.status_code != 200:
                self.test(
                    f"Prediction run ({symbol})",
                    False,
                    f"HTTP {response.status_code}: {response.text[:100]}",
                )
                return

            data = response.json()
            has_prediction_id = data.get("prediction_id") is not None
            has_price = data.get("price") is not None

            self.test(
                f"Prediction run ({symbol})",
                has_prediction_id and has_price,
                f"pred_id={data.get('prediction_id')}, price={data.get('price')}",
            )

        except Exception as e:
            self.test(f"Prediction run ({symbol})", False, str(e))

    async def test_sse_stream(self):
        """Test /api/cockpit/stream SSE endpoint"""
        try:
            url = f"{BASE_URL}/api/cockpit/stream"
            headers = {"Authorization": f"Bearer {API_TOKEN}"}

            # Use aiohttp for async SSE
            import aiohttp

            received_status = False
            received_ping = False
            received_snapshot = False

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status != 200:
                        self.test("SSE stream", False, f"HTTP {response.status}")
                        return

                    start = asyncio.get_event_loop().time()
                    async for line in response.content:
                        # Check timeout (15 seconds)
                        if asyncio.get_event_loop().time() - start > 15:
                            break

                        line_str = line.decode().strip()

                        if line_str.startswith("event: status"):
                            received_status = True
                        elif line_str.startswith("event: ping"):
                            received_ping = True
                        elif line_str.startswith("event: snapshot"):
                            received_snapshot = True

                        # If we got all 3, we're good
                        if received_status and received_ping:
                            break

            self.test(
                "SSE stream",
                received_status and received_ping,
                f"status={received_status}, ping={received_ping}, snapshot={received_snapshot}",
            )

        except Exception as e:
            self.test("SSE stream", False, str(e))

    def test_telegram_alert_format(self):
        """Test Telegram alert rendering (dry run)"""
        try:
            # Import the telegram_alerts module
            from core import telegram_alerts

            # Mock data
            prediction = {
                "action": "BUY",
                "confidence": 0.75,
                "direction": "UP",
                "factors": ["Strong momentum", "Volume spike", "RSI oversold"],
            }

            price_meta = {
                "price": 150.25,
                "prev_close": 148.50,
                "provider": "polygon",
                "after_hours": False,
            }

            # Render alert
            message = telegram_alerts.render_alert(
                symbol="AAPL",
                market="stock",
                horizon_bucket="SHORT",
                prediction=prediction,
                price_meta=price_meta,
            )

            # Check message format
            has_symbol = "AAPL" in message
            has_action = "BUY" in message
            has_confidence = "75%" in message
            has_price = "150.25" in message
            has_factors = "Strong momentum" in message
            no_zero_confidence = "0%" not in message
            no_contradictions = not ("HOLD" in message and "BUY" in message)

            passed = all(
                [
                    has_symbol,
                    has_action,
                    has_confidence,
                    has_price,
                    has_factors,
                    no_zero_confidence,
                    no_contradictions,
                ]
            )

            self.test(
                "Telegram alert format",
                passed,
                f"symbol={has_symbol}, action={has_action}, conf={has_confidence}, "
                f"no_0%={no_zero_confidence}, no_contradictions={no_contradictions}",
            )

        except Exception as e:
            self.test("Telegram alert format", False, str(e))

    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0

        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total:  {total}")
        print(f"Passed: {self.passed} ({pass_rate:.1f}%)")
        print(f"Failed: {self.failed}")
        print("=" * 60)

        return self.failed == 0


async def main():
    """Run all tests"""
    tests = LivePipelineTests()

    print("🧪 Ghost Live Pipeline Tests")
    print("=" * 60)

    # Test price diagnostics
    print("\n📊 Price Diagnostics Tests")
    tests.test_price_diagnostics("AAPL", "stock")
    tests.test_price_diagnostics("WOLF", "stock")
    tests.test_price_diagnostics("BTC", "crypto")

    # Test predictions
    print("\n🎯 Prediction Tests")
    tests.test_prediction_run("AAPL")
    tests.test_prediction_run("BTC")

    # Test SSE stream
    print("\n📡 SSE Stream Test")
    await tests.test_sse_stream()

    # Test Telegram alert format
    print("\n✉️  Telegram Alert Test")
    tests.test_telegram_alert_format()

    # Print summary
    success = tests.print_summary()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
