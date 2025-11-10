"""
Polygon.io Live Data Integration
Real-time quotes, volume, corporate events, short interest
Budget: ~$29/month Starter Plan
"""

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class RealtimeQuote:
    """Real-time stock quote"""

    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: int
    change_pct: float
    prev_close: float


@dataclass
class CorporateEvent:
    """Corporate event (earnings, filings, etc.)"""

    symbol: str
    event_type: str  # earnings, filing, dividend, split
    date: int
    description: str
    metadata: dict[str, Any]


class PolygonClient:
    """
    Polygon.io API client for real-time market data
    Requires API key in environment: POLYGON_API_KEY
    """

    BASE_URL = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            LOGGER.warning("No Polygon API key found - using free tier with delays")

        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {self.api_key}" if self.api_key else ""}
        )

    def get_realtime_quote(self, symbol: str) -> RealtimeQuote | None:
        """Get real-time quote for symbol"""
        try:
            # Get last trade
            url = f"{self.BASE_URL}/v2/last/trade/{symbol}"
            params = {"apiKey": self.api_key} if self.api_key else {}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data.get("status") != "OK":
                LOGGER.error(f"Polygon API error: {data}")
                return None

            results = data.get("results", {})

            # Get previous close
            prev_close_url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/prev"
            prev_response = self.session.get(prev_close_url, params=params, timeout=10)
            prev_data = prev_response.json()

            prev_close = 0.0
            if prev_data.get("results"):
                prev_close = prev_data["results"][0].get("c", 0.0)

            price = results.get("p", 0.0)
            change_pct = ((price - prev_close) / prev_close * 100) if prev_close > 0 else 0.0

            return RealtimeQuote(
                symbol=symbol,
                price=price,
                bid=results.get("bid", price),
                ask=results.get("ask", price),
                volume=results.get("s", 0),
                timestamp=results.get("t", int(time.time() * 1000)) // 1000,
                change_pct=change_pct,
                prev_close=prev_close,
            )

        except Exception as e:
            LOGGER.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_bulk_quotes(self, symbols: list[str]) -> dict[str, RealtimeQuote]:
        """Get real-time quotes for multiple symbols"""
        quotes = {}

        for symbol in symbols:
            quote = self.get_realtime_quote(symbol)
            if quote:
                quotes[symbol] = quote
            time.sleep(0.1)  # Rate limiting (12 req/sec limit on free tier)

        return quotes

    def get_daily_volume(self, symbol: str, days: int = 20) -> list[int]:
        """Get historical daily volume for averaging"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 5)  # Extra buffer

            url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {"apiKey": self.api_key} if self.api_key else {}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            volumes = [r.get("v", 0) for r in results[-days:]]
            return volumes

        except Exception as e:
            LOGGER.error(f"Error fetching volume for {symbol}: {e}")
            return []

    def get_short_interest(self, symbol: str) -> dict[str, Any] | None:
        """Get short interest data (Starter plan feature)"""
        try:
            # Note: Short interest endpoint may require higher tier
            # This is a placeholder for the data structure
            url = f"{self.BASE_URL}/v2/market/short-interest/{symbol}"
            params = {"apiKey": self.api_key} if self.api_key else {}

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return {
                    "symbol": symbol,
                    "short_interest": data.get("short_interest", 0),
                    "short_percent": data.get("short_percent_of_float", 0.0),
                    "days_to_cover": data.get("days_to_cover", 0.0),
                    "updated": data.get("settlement_date", ""),
                }
            else:
                LOGGER.warning(f"Short interest not available for {symbol}")
                return None

        except Exception as e:
            LOGGER.error(f"Error fetching short interest for {symbol}: {e}")
            return None

    def get_corporate_events(
        self, symbol: str | None = None, event_type: str | None = None, days_ahead: int = 30
    ) -> list[CorporateEvent]:
        """Get upcoming corporate events"""
        events = []

        try:
            # Earnings calendar
            if not event_type or event_type == "earnings":
                earnings = self._get_earnings_calendar(symbol, days_ahead)
                events.extend(earnings)

            # Dividends
            if not event_type or event_type == "dividend":
                dividends = self._get_dividends(symbol, days_ahead)
                events.extend(dividends)

            return sorted(events, key=lambda x: x.date)

        except Exception as e:
            LOGGER.error(f"Error fetching corporate events: {e}")
            return []

    def _get_earnings_calendar(self, symbol: str | None, days_ahead: int) -> list[CorporateEvent]:
        """Get earnings calendar"""
        events = []

        try:
            url = f"{self.BASE_URL}/v2/reference/earnings"
            params = {"apiKey": self.api_key} if self.api_key else {}

            if symbol:
                params["ticker"] = symbol

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                cutoff = int(time.time()) + (days_ahead * 86400)

                for result in data.get("results", []):
                    report_date = result.get("report_date", "")
                    if not report_date:
                        continue

                    # Parse date
                    dt = datetime.strptime(report_date, "%Y-%m-%d")
                    timestamp = int(dt.timestamp())

                    if timestamp > cutoff:
                        continue

                    events.append(
                        CorporateEvent(
                            symbol=result.get("ticker", ""),
                            event_type="earnings",
                            date=timestamp,
                            description=f"Earnings report: Q{result.get('quarter', '?')} {result.get('year', '')}",
                            metadata={
                                "quarter": result.get("quarter"),
                                "year": result.get("year"),
                                "consensus_eps": result.get("consensus_eps"),
                                "actual_eps": result.get("actual_eps"),
                            },
                        )
                    )

        except Exception as e:
            LOGGER.error(f"Error fetching earnings: {e}")

        return events

    def _get_dividends(self, symbol: str | None, days_ahead: int) -> list[CorporateEvent]:
        """Get dividend calendar"""
        events = []

        try:
            if not symbol:
                return events  # Dividends require specific symbol

            url = f"{self.BASE_URL}/v2/reference/dividends/{symbol}"
            params = {"apiKey": self.api_key} if self.api_key else {}

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                cutoff = int(time.time()) + (days_ahead * 86400)

                for result in data.get("results", []):
                    ex_date = result.get("ex_dividend_date", "")
                    if not ex_date:
                        continue

                    dt = datetime.strptime(ex_date, "%Y-%m-%d")
                    timestamp = int(dt.timestamp())

                    if timestamp > cutoff:
                        continue

                    events.append(
                        CorporateEvent(
                            symbol=symbol,
                            event_type="dividend",
                            date=timestamp,
                            description=f"Dividend: ${result.get('amount', 0):.2f} per share",
                            metadata={
                                "amount": result.get("amount"),
                                "ex_date": ex_date,
                                "payment_date": result.get("payment_date"),
                                "record_date": result.get("record_date"),
                            },
                        )
                    )

        except Exception as e:
            LOGGER.error(f"Error fetching dividends: {e}")

        return events

    def get_stock_splits(self, symbol: str, days_back: int = 90) -> list[dict[str, Any]]:
        """Get recent stock splits"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            url = f"{self.BASE_URL}/v2/reference/splits/{symbol}"
            params = {"from": start_date.strftime("%Y-%m-%d"), "to": end_date.strftime("%Y-%m-%d")}

            if self.api_key:
                params["apiKey"] = self.api_key

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return data.get("results", [])

            return []

        except Exception as e:
            LOGGER.error(f"Error fetching splits for {symbol}: {e}")
            return []

    def get_market_status(self) -> dict[str, Any]:
        """Get market open/close status"""
        try:
            url = f"{self.BASE_URL}/v1/marketstatus/now"
            params = {"apiKey": self.api_key} if self.api_key else {}

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            return {
                "market": data.get("market", "unknown"),
                "server_time": data.get("serverTime", ""),
                "exchanges": {
                    "nyse": data.get("exchanges", {}).get("nyse", "unknown"),
                    "nasdaq": data.get("exchanges", {}).get("nasdaq", "unknown"),
                },
                "currencies": data.get("currencies", {}),
                "is_open": data.get("market") == "open",
            }

        except Exception as e:
            LOGGER.error(f"Error fetching market status: {e}")
            return {"market": "unknown", "is_open": False}

    def search_ticker(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for ticker symbols"""
        try:
            url = f"{self.BASE_URL}/v3/reference/tickers"
            params = {"search": query, "limit": limit, "active": "true"}

            if self.api_key:
                params["apiKey"] = self.api_key

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = []

            for ticker in data.get("results", []):
                results.append(
                    {
                        "ticker": ticker.get("ticker"),
                        "name": ticker.get("name"),
                        "market": ticker.get("market"),
                        "type": ticker.get("type"),
                        "primary_exchange": ticker.get("primary_exchange"),
                        "active": ticker.get("active"),
                    }
                )

            return results

        except Exception as e:
            LOGGER.error(f"Error searching tickers: {e}")
            return []


# Singleton instance
_polygon_client: PolygonClient | None = None


def get_polygon_client() -> PolygonClient:
    """Get singleton Polygon client"""
    global _polygon_client
    if _polygon_client is None:
        _polygon_client = PolygonClient()
    return _polygon_client
