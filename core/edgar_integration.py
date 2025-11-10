"""
SEC EDGAR Integration - Free Corporate Filings Data
Real-time 8-K, 10-K, 10-Q, 13F filings with auto-ticker linking
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class SECFiling:
    """SEC filing document"""

    accession_number: str
    filing_type: str  # 8-K, 10-K, 10-Q, 13F, etc.
    company_name: str
    cik: str
    ticker: str | None
    filing_date: int
    report_date: int | None
    description: str
    url: str
    items: list[str]  # For 8-K: Item 2.02, etc.
    sentiment_score: float  # Calculated from filing text
    urgency: str  # "low", "medium", "high", "critical"


class EDGARClient:
    """
    SEC EDGAR API client (100% free)
    Rate limit: 10 requests/second with User-Agent
    """

    BASE_URL = "https://data.sec.gov"

    # User-Agent required by SEC
    USER_AGENT = "GHOST Trading Platform info@ghosttrading.ai"

    # 8-K Item codes and their importance
    CRITICAL_8K_ITEMS = {
        "1.01": "Entry into Material Agreement",
        "1.02": "Termination of Material Agreement",
        "2.01": "Completion of Acquisition or Disposition",
        "2.02": "Results of Operations and Financial Condition",
        "2.03": "Creation of Direct Financial Obligation",
        "2.04": "Triggering Events That Accelerate Direct Financial Obligation",
        "3.01": "Notice of Delisting",
        "4.01": "Changes in Accountant",
        "4.02": "Non-Reliance on Previously Issued Financial Statements",
        "5.02": "Departure of Directors or Officers; Election of Directors",
        "8.01": "Other Events",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": self.USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        )

    def get_recent_filings(
        self, filing_type: str | None = None, hours_back: int = 24, limit: int = 100
    ) -> list[SECFiling]:
        """
        Get recent filings from RSS feed
        Types: 8-K (breaking news), 10-K (annual), 10-Q (quarterly), 13F (holdings)
        """
        filings = []

        try:
            # SEC provides atom feeds for recent filings
            if filing_type:
                url = f"{self.BASE_URL}/cgi-bin/browse-edgar?action=getcurrent&type={filing_type}&company=&dateb=&owner=exclude&start=0&count={limit}&output=atom"
            else:
                url = f"{self.BASE_URL}/cgi-bin/browse-edgar?action=getcurrent&company=&dateb=&owner=exclude&start=0&count={limit}&output=atom"

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            # Parse atom XML
            root = ET.fromstring(response.content)

            # Namespace
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            cutoff = int(time.time()) - (hours_back * 3600)

            for entry in root.findall("atom:entry", ns):
                # Extract basic info
                title_elem = entry.find("atom:title", ns)
                if title_elem is None or title_elem.text is None:
                    continue

                title = title_elem.text

                # Parse title: "8-K - COMPANY NAME (CIK)"
                match = re.match(r"([\w\-/]+)\s*-\s*(.+?)\s*\((\d+)\)", title)
                if not match:
                    continue

                form_type = match.group(1).strip()
                company_name = match.group(2).strip()
                cik = match.group(3).strip()

                # Get filing date
                updated_elem = entry.find("atom:updated", ns)
                if updated_elem is None or updated_elem.text is None:
                    continue

                filing_date = datetime.fromisoformat(updated_elem.text.replace("Z", "+00:00"))
                filing_timestamp = int(filing_date.timestamp())

                if filing_timestamp < cutoff:
                    continue

                # Get URL
                link_elem = entry.find("atom:link", ns)
                filing_url = link_elem.get("href", "") if link_elem is not None else ""

                # Get accession number from URL
                accession_match = re.search(r"/([0-9\-]+)\.txt", filing_url)
                accession_number = accession_match.group(1) if accession_match else ""

                # Get summary
                summary_elem = entry.find("atom:summary", ns)
                description = (
                    summary_elem.text if summary_elem is not None and summary_elem.text else ""
                )

                # For 8-K, extract Item numbers
                items = []
                urgency = "medium"

                if form_type == "8-K":
                    items = self._extract_8k_items(description)
                    urgency = self._assess_8k_urgency(items)
                elif form_type in ["10-K", "10-Q"]:
                    urgency = "high"
                elif form_type == "13F-HR":
                    urgency = "medium"

                # Get ticker (requires CIK lookup)
                ticker = self._cik_to_ticker(cik)

                # Calculate sentiment (simple keyword analysis)
                sentiment = self._calculate_filing_sentiment(description, form_type)

                filing = SECFiling(
                    accession_number=accession_number,
                    filing_type=form_type,
                    company_name=company_name,
                    cik=cik,
                    ticker=ticker,
                    filing_date=filing_timestamp,
                    report_date=None,  # Would need to parse document
                    description=description[:500],  # Truncate
                    url=filing_url,
                    items=items,
                    sentiment_score=sentiment,
                    urgency=urgency,
                )

                filings.append(filing)

            LOGGER.info(f"Fetched {len(filings)} recent filings")
            return filings

        except Exception as e:
            LOGGER.error(f"Error fetching SEC filings: {e}")
            return []

    def get_company_filings(
        self, ticker_or_cik: str, filing_type: str | None = None, limit: int = 20
    ) -> list[SECFiling]:
        """Get filings for specific company"""
        try:
            # If ticker, convert to CIK
            if not ticker_or_cik.isdigit():
                cik = self._ticker_to_cik(ticker_or_cik)
                if not cik:
                    LOGGER.warning(f"Could not find CIK for {ticker_or_cik}")
                    return []
            else:
                cik = ticker_or_cik

            # Pad CIK to 10 digits
            cik_padded = cik.zfill(10)

            # Get company submissions
            url = f"{self.BASE_URL}/submissions/CIK{cik_padded}.json"

            response = self.session.get(url, timeout=15)
            response.raise_for_status()

            data = response.json()

            filings = []
            recent_filings = data.get("filings", {}).get("recent", {})

            forms = recent_filings.get("form", [])
            filing_dates = recent_filings.get("filingDate", [])
            accession_numbers = recent_filings.get("accessionNumber", [])
            primary_docs = recent_filings.get("primaryDocument", [])

            for i in range(min(len(forms), limit)):
                form_type = forms[i]

                if filing_type and form_type != filing_type:
                    continue

                filing_date_str = filing_dates[i]
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d")
                filing_timestamp = int(filing_date.timestamp())

                accession = accession_numbers[i]
                doc = primary_docs[i]

                # Build URL
                accession_clean = accession.replace("-", "")
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_clean}/{doc}"
                )

                filing = SECFiling(
                    accession_number=accession,
                    filing_type=form_type,
                    company_name=data.get("name", ""),
                    cik=cik,
                    ticker=data.get("tickers", [None])[0] if data.get("tickers") else None,
                    filing_date=filing_timestamp,
                    report_date=None,
                    description=f"{form_type} filing",
                    url=filing_url,
                    items=[],
                    sentiment_score=0.0,
                    urgency="medium",
                )

                filings.append(filing)

            return filings

        except Exception as e:
            LOGGER.error(f"Error fetching company filings for {ticker_or_cik}: {e}")
            return []

    def _extract_8k_items(self, description: str) -> list[str]:
        """Extract Item numbers from 8-K description"""
        items = []

        # Look for patterns like "Item 2.02", "Items 1.01 and 8.01"
        pattern = r"Item(?:s)?\s+([\d\.]+(?:,?\s+(?:and\s+)?[\d\.]+)*)"
        matches = re.findall(pattern, description, re.IGNORECASE)

        for match in matches:
            # Split multiple items
            item_nums = re.findall(r"[\d\.]+", match)
            items.extend(item_nums)

        return list(set(items))  # Unique items

    def _assess_8k_urgency(self, items: list[str]) -> str:
        """Assess urgency based on 8-K items"""
        critical_items = ["1.02", "2.01", "2.04", "3.01", "4.02"]
        high_items = ["1.01", "2.02", "2.03", "5.02"]

        for item in items:
            if item in critical_items:
                return "critical"

        for item in items:
            if item in high_items:
                return "high"

        return "medium"

    def _calculate_filing_sentiment(self, text: str, form_type: str) -> float:
        """Simple sentiment analysis of filing text"""
        positive_words = [
            "growth",
            "increased",
            "profit",
            "revenue",
            "improved",
            "strong",
            "exceeded",
            "positive",
            "gain",
            "acquisition",
            "expanded",
            "launched",
            "innovative",
            "successful",
        ]

        negative_words = [
            "loss",
            "decreased",
            "declined",
            "impairment",
            "litigation",
            "investigation",
            "restructuring",
            "bankruptcy",
            "default",
            "terminated",
            "delisting",
            "restatement",
            "warning",
            "risk",
        ]

        text_lower = text.lower()

        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        sentiment = (pos_count - neg_count) / total
        return sentiment

    def _ticker_to_cik(self, ticker: str) -> str | None:
        """Convert ticker to CIK (requires ticker mappings)"""
        try:
            # SEC provides company tickers JSON
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            ticker_upper = ticker.upper()
            for _key, company in data.items():
                if company.get("ticker") == ticker_upper:
                    return str(company.get("cik_str"))

            return None

        except Exception as e:
            LOGGER.error(f"Error converting ticker {ticker} to CIK: {e}")
            return None

    def _cik_to_ticker(self, cik: str) -> str | None:
        """Convert CIK to ticker"""
        try:
            # SEC provides company tickers JSON
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            cik_int = int(cik)
            for _key, company in data.items():
                if company.get("cik_str") == cik_int:
                    return company.get("ticker")

            return None

        except Exception as e:
            LOGGER.error(f"Error converting CIK {cik} to ticker: {e}")
            return None

    def search_companies(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for companies by name"""
        try:
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()

            query_lower = query.lower()
            results = []

            for _key, company in data.items():
                name = company.get("title", "").lower()
                ticker = company.get("ticker", "")

                if query_lower in name or query_lower in ticker.lower():
                    results.append(
                        {
                            "cik": company.get("cik_str"),
                            "ticker": ticker,
                            "name": company.get("title"),
                            "exchange": company.get("exchange", ""),
                        }
                    )

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            LOGGER.error(f"Error searching companies: {e}")
            return []

    def get_insider_transactions(
        self, ticker_or_cik: str, days_back: int = 90
    ) -> list[dict[str, Any]]:
        """Get Form 4 insider transactions"""
        try:
            # Get CIK
            if not ticker_or_cik.isdigit():
                cik = self._ticker_to_cik(ticker_or_cik)
                if not cik:
                    return []
            else:
                cik = ticker_or_cik

            # Get company filings
            filings = self.get_company_filings(cik, filing_type="4", limit=50)

            cutoff = int(time.time()) - (days_back * 86400)

            transactions = []
            for filing in filings:
                if filing.filing_date < cutoff:
                    continue

                transactions.append(
                    {
                        "ticker": filing.ticker,
                        "company": filing.company_name,
                        "filing_date": filing.filing_date,
                        "url": filing.url,
                        "type": "insider_transaction",
                    }
                )

            return transactions

        except Exception as e:
            LOGGER.error(f"Error fetching insider transactions: {e}")
            return []


# Singleton instance
_edgar_client: EDGARClient | None = None


def get_edgar_client() -> EDGARClient:
    """Get singleton EDGAR client"""
    global _edgar_client
    if _edgar_client is None:
        _edgar_client = EDGARClient()
    return _edgar_client
