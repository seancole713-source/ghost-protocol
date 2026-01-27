"""
Ghost Intel - Full Stock History Module

Provides comprehensive historical data for stocks:
1. Full price history (IPO to now)
2. Fundamentals (P/E, market cap, earnings, revenue)
3. 52-week context (high, low, % from each)
4. Volume analysis (current vs average)
5. Key technical levels
6. Earnings history and patterns

This data feeds into Intel rules for better predictions.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from functools import lru_cache
import time

LOGGER = logging.getLogger(__name__)

@dataclass
class StockFundamentals:
    """Stock fundamental data"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    market_cap: float
    pe_ratio: Optional[float]
    forward_pe: Optional[float]
    peg_ratio: Optional[float]
    price_to_book: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    profit_margin: Optional[float]
    revenue_growth: Optional[float]
    earnings_growth: Optional[float]
    fifty_two_week_high: float
    fifty_two_week_low: float
    fifty_day_avg: float
    two_hundred_day_avg: float
    avg_volume: float
    shares_outstanding: float
    float_shares: Optional[float]
    short_ratio: Optional[float]
    short_percent: Optional[float]
    insider_ownership: Optional[float]
    institution_ownership: Optional[float]

@dataclass 
class StockHistory:
    """Complete stock history context"""
    symbol: str
    fundamentals: Optional[StockFundamentals]
    
    # Price context
    current_price: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    pct_from_52w_high: float  # Negative = below high
    pct_from_52w_low: float   # Positive = above low
    all_time_high: Optional[float]
    all_time_low: Optional[float]
    ipo_date: Optional[str]
    years_trading: Optional[float]
    
    # Volume context
    current_volume: float
    avg_volume_10d: float
    avg_volume_90d: float
    volume_ratio: float  # Current / avg
    
    # Moving averages
    sma_20: float
    sma_50: float
    sma_200: float
    above_sma_20: bool
    above_sma_50: bool
    above_sma_200: bool
    
    # Technical levels
    support_level: Optional[float]
    resistance_level: Optional[float]
    pivot_point: Optional[float]
    
    # Momentum
    rsi_14: float
    macd_signal: str  # "bullish", "bearish", "neutral"
    trend_direction: str  # "uptrend", "downtrend", "sideways"
    
    # Earnings context
    next_earnings_date: Optional[str]
    days_to_earnings: Optional[int]
    last_earnings_surprise: Optional[float]  # % beat/miss
    earnings_trend: str  # "beats", "misses", "mixed"
    
    # Historical patterns
    avg_daily_range_pct: float
    max_drawdown_1y: float
    best_month: Optional[str]
    worst_month: Optional[str]
    
    # Flags
    is_near_52w_high: bool  # Within 5%
    is_near_52w_low: bool   # Within 5%
    is_oversold: bool       # RSI < 30
    is_overbought: bool     # RSI > 70
    is_high_volume: bool    # > 2x avg
    is_low_volume: bool     # < 0.5x avg


class StockHistoryProvider:
    """Fetches and caches comprehensive stock history"""
    
    def __init__(self):
        self._cache: Dict[str, Tuple[StockHistory, float]] = {}
        self._cache_ttl = 300  # 5 minutes
        self._fundamentals_cache: Dict[str, Tuple[StockFundamentals, float]] = {}
        self._fundamentals_ttl = 3600  # 1 hour for fundamentals
        self._failed_symbols: Dict[str, float] = {}  # Track failed fetches to avoid hammering
        self._failed_ttl = 60  # Retry failed symbols after 60 seconds
    
    def get_stock_history(self, symbol: str) -> Optional[StockHistory]:
        """Get comprehensive stock history with caching"""
        symbol = symbol.upper()
        
        # Check cache
        if symbol in self._cache:
            cached, timestamp = self._cache[symbol]
            if time.time() - timestamp < self._cache_ttl:
                return cached
        
        # Check if recently failed (avoid hammering API)
        if symbol in self._failed_symbols:
            if time.time() - self._failed_symbols[symbol] < self._failed_ttl:
                LOGGER.debug(f"Skipping {symbol} - recently failed")
                return None
        
        try:
            history = self._fetch_stock_history(symbol)
            if history:
                self._cache[symbol] = (history, time.time())
                # Clear from failed list if it was there
                self._failed_symbols.pop(symbol, None)
            else:
                self._failed_symbols[symbol] = time.time()
            return history
        except Exception as e:
            LOGGER.error(f"Error fetching history for {symbol}: {e}")
            self._failed_symbols[symbol] = time.time()
            return None
    
    def _fetch_stock_history(self, symbol: str) -> Optional[StockHistory]:
        """Fetch all historical data for a stock"""
        try:
            import yfinance as yf
            import pandas as pd
            import numpy as np
            import os
            
            # Set longer timeout and retry
            ticker = yf.Ticker(symbol)
            
            # Try to get history - be graceful about failures
            hist_max = None
            hist_1y = None
            hist_3mo = None
            
            try:
                hist_1y = ticker.history(period="1y", timeout=10)
            except Exception as e:
                LOGGER.debug(f"1y history failed for {symbol}: {e}")
                # Try shorter period
                try:
                    hist_1y = ticker.history(period="6mo", timeout=10)
                except:
                    pass
            
            if hist_1y is None or hist_1y.empty:
                # Try Polygon API as fallback
                return self._fetch_from_polygon(symbol)
            
            try:
                hist_3mo = ticker.history(period="3mo", timeout=10)
            except:
                hist_3mo = hist_1y.tail(63) if not hist_1y.empty else pd.DataFrame()
            
            try:
                hist_max = ticker.history(period="max", timeout=15)
            except:
                hist_max = hist_1y  # Use 1y as fallback
            
            # Get info (fundamentals) - wrap in try/except
            info = {}
            try:
                info = ticker.info or {}
            except:
                pass
            
            # Current price
            current_price = float(hist_1y['Close'].iloc[-1])
            
            # 52-week high/low
            fifty_two_week_high = float(hist_1y['High'].max())
            fifty_two_week_low = float(hist_1y['Low'].min())
            
            pct_from_52w_high = ((current_price - fifty_two_week_high) / fifty_two_week_high) * 100
            pct_from_52w_low = ((current_price - fifty_two_week_low) / fifty_two_week_low) * 100
            
            # All-time high/low
            all_time_high = float(hist_max['High'].max()) if not hist_max.empty else None
            all_time_low = float(hist_max['Low'].min()) if not hist_max.empty else None
            
            # IPO date and years trading
            ipo_date = None
            years_trading = None
            if not hist_max.empty:
                first_date = hist_max.index[0]
                ipo_date = first_date.strftime("%Y-%m-%d")
                years_trading = (datetime.now() - first_date.to_pydatetime().replace(tzinfo=None)).days / 365.25
            
            # Volume analysis
            current_volume = float(hist_1y['Volume'].iloc[-1])
            avg_volume_10d = float(hist_1y['Volume'].tail(10).mean())
            avg_volume_90d = float(hist_3mo['Volume'].mean()) if not hist_3mo.empty else avg_volume_10d
            volume_ratio = current_volume / avg_volume_90d if avg_volume_90d > 0 else 1.0
            
            # Moving averages
            close = hist_1y['Close']
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma_20
            sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma_50
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_diff = float((macd - signal).iloc[-1])
            macd_signal = "bullish" if macd_diff > 0 else "bearish" if macd_diff < 0 else "neutral"
            
            # Trend direction
            if current_price > sma_50 > sma_200:
                trend_direction = "uptrend"
            elif current_price < sma_50 < sma_200:
                trend_direction = "downtrend"
            else:
                trend_direction = "sideways"
            
            # Support/Resistance (simple pivot points)
            recent_high = float(hist_3mo['High'].max())
            recent_low = float(hist_3mo['Low'].min())
            pivot_point = (recent_high + recent_low + current_price) / 3
            support_level = (2 * pivot_point) - recent_high
            resistance_level = (2 * pivot_point) - recent_low
            
            # Average daily range
            daily_range = (hist_1y['High'] - hist_1y['Low']) / hist_1y['Close'] * 100
            avg_daily_range_pct = float(daily_range.mean())
            
            # Max drawdown (1 year)
            rolling_max = hist_1y['Close'].cummax()
            drawdown = (hist_1y['Close'] - rolling_max) / rolling_max * 100
            max_drawdown_1y = float(drawdown.min())
            
            # Monthly performance (find best/worst month)
            best_month = None
            worst_month = None
            try:
                monthly = hist_1y['Close'].resample('M').last().pct_change() * 100
                if not monthly.empty:
                    best_idx = monthly.idxmax()
                    worst_idx = monthly.idxmin()
                    best_month = best_idx.strftime("%B") if pd.notna(best_idx) else None
                    worst_month = worst_idx.strftime("%B") if pd.notna(worst_idx) else None
            except:
                pass
            
            # Earnings info
            next_earnings_date = None
            days_to_earnings = None
            try:
                calendar = ticker.calendar
                if calendar is not None and 'Earnings Date' in calendar:
                    earnings_dates = calendar['Earnings Date']
                    if earnings_dates:
                        next_date = earnings_dates[0] if isinstance(earnings_dates, list) else earnings_dates
                        if hasattr(next_date, 'strftime'):
                            next_earnings_date = next_date.strftime("%Y-%m-%d")
                            days_to_earnings = (next_date - datetime.now()).days
            except:
                pass
            
            # Earnings history
            last_earnings_surprise = info.get('earningsQuarterlyGrowth')
            earnings_trend = "mixed"  # Default
            
            # Build fundamentals
            fundamentals = self._build_fundamentals(symbol, info)
            
            # Build history object
            return StockHistory(
                symbol=symbol,
                fundamentals=fundamentals,
                current_price=current_price,
                fifty_two_week_high=fifty_two_week_high,
                fifty_two_week_low=fifty_two_week_low,
                pct_from_52w_high=pct_from_52w_high,
                pct_from_52w_low=pct_from_52w_low,
                all_time_high=all_time_high,
                all_time_low=all_time_low,
                ipo_date=ipo_date,
                years_trading=years_trading,
                current_volume=current_volume,
                avg_volume_10d=avg_volume_10d,
                avg_volume_90d=avg_volume_90d,
                volume_ratio=volume_ratio,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                above_sma_20=current_price > sma_20,
                above_sma_50=current_price > sma_50,
                above_sma_200=current_price > sma_200,
                support_level=support_level,
                resistance_level=resistance_level,
                pivot_point=pivot_point,
                rsi_14=rsi_14,
                macd_signal=macd_signal,
                trend_direction=trend_direction,
                next_earnings_date=next_earnings_date,
                days_to_earnings=days_to_earnings,
                last_earnings_surprise=last_earnings_surprise,
                earnings_trend=earnings_trend,
                avg_daily_range_pct=avg_daily_range_pct,
                max_drawdown_1y=max_drawdown_1y,
                best_month=best_month,
                worst_month=worst_month,
                is_near_52w_high=pct_from_52w_high >= -5,
                is_near_52w_low=pct_from_52w_low <= 5,
                is_oversold=rsi_14 < 30,
                is_overbought=rsi_14 > 70,
                is_high_volume=volume_ratio >= 2.0,
                is_low_volume=volume_ratio <= 0.5,
            )
            
        except Exception as e:
            LOGGER.error(f"Error building history for {symbol}: {e}", exc_info=True)
            return None
    
    def _build_fundamentals(self, symbol: str, info: Dict) -> Optional[StockFundamentals]:
        """Build fundamentals from yfinance info"""
        try:
            return StockFundamentals(
                symbol=symbol,
                company_name=info.get('longName', info.get('shortName', symbol)),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=info.get('marketCap', 0),
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                price_to_book=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                beta=info.get('beta'),
                profit_margin=info.get('profitMargins'),
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),
                fifty_two_week_high=info.get('fiftyTwoWeekHigh', 0),
                fifty_two_week_low=info.get('fiftyTwoWeekLow', 0),
                fifty_day_avg=info.get('fiftyDayAverage', 0),
                two_hundred_day_avg=info.get('twoHundredDayAverage', 0),
                avg_volume=info.get('averageVolume', 0),
                shares_outstanding=info.get('sharesOutstanding', 0),
                float_shares=info.get('floatShares'),
                short_ratio=info.get('shortRatio'),
                short_percent=info.get('shortPercentOfFloat'),
                insider_ownership=info.get('heldPercentInsiders'),
                institution_ownership=info.get('heldPercentInstitutions'),
            )
        except Exception as e:
            LOGGER.error(f"Error building fundamentals for {symbol}: {e}")
            return None
    
    def _fetch_from_polygon(self, symbol: str) -> Optional[StockHistory]:
        """Fallback: Fetch from Polygon API if yfinance fails"""
        import os
        import requests
        from datetime import datetime, timedelta
        import pandas as pd
        import numpy as np
        
        polygon_key = os.getenv("POLYGON_API_KEY")
        if not polygon_key:
            LOGGER.debug(f"No Polygon API key for {symbol} fallback")
            return None
        
        try:
            # Get 1 year of daily data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {
                "apiKey": polygon_key,
                "adjusted": "true",
                "sort": "asc",
                "limit": 500
            }
            
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                LOGGER.debug(f"Polygon failed for {symbol}: {resp.status_code}")
                return None
            
            data = resp.json()
            results = data.get("results", [])
            
            if not results:
                return None
            
            # Convert to dataframe
            df = pd.DataFrame(results)
            df['Date'] = pd.to_datetime(df['t'], unit='ms')
            df.set_index('Date', inplace=True)
            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
            
            current_price = float(df['Close'].iloc[-1])
            fifty_two_week_high = float(df['High'].max())
            fifty_two_week_low = float(df['Low'].min())
            
            pct_from_52w_high = ((current_price - fifty_two_week_high) / fifty_two_week_high) * 100
            pct_from_52w_low = ((current_price - fifty_two_week_low) / fifty_two_week_low) * 100
            
            # Calculate basic technicals
            close = df['Close']
            volume = df['Volume']
            
            # SMAs
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else sma_20
            sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma_50
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
            
            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_diff = float((macd - signal).iloc[-1])
            macd_signal_str = "bullish" if macd_diff > 0 else "bearish" if macd_diff < 0 else "neutral"
            
            # Volume
            current_volume = float(volume.iloc[-1])
            avg_volume_90d = float(volume.tail(90).mean())
            volume_ratio = current_volume / avg_volume_90d if avg_volume_90d > 0 else 1.0
            
            # Trend
            if current_price > sma_50 > sma_200:
                trend_direction = "uptrend"
            elif current_price < sma_50 < sma_200:
                trend_direction = "downtrend"
            else:
                trend_direction = "sideways"
            
            # Support/Resistance
            recent_high = float(df.tail(63)['High'].max())
            recent_low = float(df.tail(63)['Low'].min())
            pivot_point = (recent_high + recent_low + current_price) / 3
            support_level = (2 * pivot_point) - recent_high
            resistance_level = (2 * pivot_point) - recent_low
            
            # Daily range
            daily_range = (df['High'] - df['Low']) / df['Close'] * 100
            avg_daily_range_pct = float(daily_range.mean())
            
            # Max drawdown
            rolling_max = df['Close'].cummax()
            drawdown = (df['Close'] - rolling_max) / rolling_max * 100
            max_drawdown_1y = float(drawdown.min())
            
            LOGGER.info(f"📊 Polygon fallback succeeded for {symbol}")
            
            return StockHistory(
                symbol=symbol,
                fundamentals=None,  # Polygon doesn't have fundamentals in free tier
                current_price=current_price,
                fifty_two_week_high=fifty_two_week_high,
                fifty_two_week_low=fifty_two_week_low,
                pct_from_52w_high=pct_from_52w_high,
                pct_from_52w_low=pct_from_52w_low,
                all_time_high=fifty_two_week_high,  # Using 52w as proxy
                all_time_low=fifty_two_week_low,
                ipo_date=None,
                years_trading=None,
                current_volume=current_volume,
                avg_volume_10d=float(volume.tail(10).mean()),
                avg_volume_90d=avg_volume_90d,
                volume_ratio=volume_ratio,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                above_sma_20=current_price > sma_20,
                above_sma_50=current_price > sma_50,
                above_sma_200=current_price > sma_200,
                support_level=support_level,
                resistance_level=resistance_level,
                pivot_point=pivot_point,
                rsi_14=rsi_14,
                macd_signal=macd_signal_str,
                trend_direction=trend_direction,
                next_earnings_date=None,
                days_to_earnings=None,
                last_earnings_surprise=None,
                earnings_trend="unknown",
                avg_daily_range_pct=avg_daily_range_pct,
                max_drawdown_1y=max_drawdown_1y,
                best_month=None,
                worst_month=None,
                is_near_52w_high=pct_from_52w_high >= -5,
                is_near_52w_low=pct_from_52w_low <= 5,
                is_oversold=rsi_14 < 30,
                is_overbought=rsi_14 > 70,
                is_high_volume=volume_ratio >= 2.0,
                is_low_volume=volume_ratio <= 0.5,
            )
            
        except Exception as e:
            LOGGER.debug(f"Polygon fallback failed for {symbol}: {e}")
            return None
    
    def get_fundamentals(self, symbol: str) -> Optional[StockFundamentals]:
        """Get just fundamentals (cached longer)"""
        symbol = symbol.upper()
        
        # Check cache
        if symbol in self._fundamentals_cache:
            cached, timestamp = self._fundamentals_cache[symbol]
            if time.time() - timestamp < self._fundamentals_ttl:
                return cached
        
        history = self.get_stock_history(symbol)
        if history and history.fundamentals:
            self._fundamentals_cache[symbol] = (history.fundamentals, time.time())
            return history.fundamentals
        return None


# Global instance
_provider: Optional[StockHistoryProvider] = None

def get_stock_history_provider() -> StockHistoryProvider:
    """Get or create the global stock history provider"""
    global _provider
    if _provider is None:
        _provider = StockHistoryProvider()
    return _provider


def get_stock_context(symbol: str) -> Dict[str, Any]:
    """
    Get comprehensive stock context for Intel integration.
    Returns a dict that can be merged into intel_context.
    """
    provider = get_stock_history_provider()
    history = provider.get_stock_history(symbol)
    
    if not history:
        return {}
    
    context = {
        # Price context
        "current_price": history.current_price,
        "fifty_two_week_high": history.fifty_two_week_high,
        "fifty_two_week_low": history.fifty_two_week_low,
        "pct_from_52w_high": history.pct_from_52w_high,
        "pct_from_52w_low": history.pct_from_52w_low,
        "all_time_high": history.all_time_high,
        "years_trading": history.years_trading,
        
        # Volume
        "volume_ratio": history.volume_ratio,
        "relative_volume": history.volume_ratio,  # Alias for existing code
        "avg_volume": history.avg_volume_90d,
        
        # Technical
        "rsi": history.rsi_14,
        "rsi_14": history.rsi_14,
        "trend": history.trend_direction,
        "macd_signal": history.macd_signal,
        "above_sma_200": history.above_sma_200,
        "above_sma_50": history.above_sma_50,
        
        # Support/Resistance
        "support": history.support_level,
        "resistance": history.resistance_level,
        "pivot": history.pivot_point,
        
        # Flags
        "is_near_52w_high": history.is_near_52w_high,
        "is_near_52w_low": history.is_near_52w_low,
        "is_oversold": history.is_oversold,
        "is_overbought": history.is_overbought,
        "is_high_volume": history.is_high_volume,
        
        # Earnings
        "days_to_earnings": history.days_to_earnings,
        "earnings_date": history.next_earnings_date,
        
        # Volatility
        "avg_daily_range_pct": history.avg_daily_range_pct,
        "volatility_pct": history.avg_daily_range_pct,  # Alias
        "max_drawdown_1y": history.max_drawdown_1y,
    }
    
    # Add fundamentals if available
    if history.fundamentals:
        f = history.fundamentals
        context.update({
            "pe_ratio": f.pe_ratio,
            "forward_pe": f.forward_pe,
            "market_cap": f.market_cap,
            "sector": f.sector,
            "industry": f.industry,
            "beta": f.beta,
            "dividend_yield": f.dividend_yield,
            "short_percent": f.short_percent,
            "insider_ownership": f.insider_ownership,
            "institution_ownership": f.institution_ownership,
        })
    
    return context


def print_stock_summary(symbol: str):
    """Print a comprehensive stock summary"""
    provider = get_stock_history_provider()
    history = provider.get_stock_history(symbol)
    
    if not history:
        print(f"❌ Could not fetch data for {symbol}")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 GHOST INTEL: {symbol} FULL HISTORY")
    print(f"{'='*60}")
    
    # Fundamentals
    if history.fundamentals:
        f = history.fundamentals
        print(f"\n🏢 {f.company_name}")
        print(f"   Sector: {f.sector} | Industry: {f.industry}")
        print(f"   Market Cap: ${f.market_cap/1e9:.1f}B" if f.market_cap else "")
        print(f"   P/E: {f.pe_ratio:.1f}" if f.pe_ratio else "   P/E: N/A")
        print(f"   Beta: {f.beta:.2f}" if f.beta else "")
    
    # Price context
    print(f"\n💰 PRICE CONTEXT")
    print(f"   Current: ${history.current_price:.2f}")
    print(f"   52W High: ${history.fifty_two_week_high:.2f} ({history.pct_from_52w_high:+.1f}%)")
    print(f"   52W Low: ${history.fifty_two_week_low:.2f} ({history.pct_from_52w_low:+.1f}%)")
    if history.all_time_high:
        print(f"   All-Time High: ${history.all_time_high:.2f}")
    if history.years_trading:
        print(f"   Trading Since: {history.ipo_date} ({history.years_trading:.1f} years)")
    
    # Technical
    print(f"\n📈 TECHNICALS")
    print(f"   RSI(14): {history.rsi_14:.1f} {'🔴 OVERBOUGHT' if history.is_overbought else '🟢 OVERSOLD' if history.is_oversold else ''}")
    print(f"   MACD: {history.macd_signal.upper()}")
    print(f"   Trend: {history.trend_direction.upper()}")
    print(f"   SMA 20: ${history.sma_20:.2f} {'✅' if history.above_sma_20 else '❌'}")
    print(f"   SMA 50: ${history.sma_50:.2f} {'✅' if history.above_sma_50 else '❌'}")
    print(f"   SMA 200: ${history.sma_200:.2f} {'✅' if history.above_sma_200 else '❌'}")
    
    # Volume
    print(f"\n📊 VOLUME")
    print(f"   Current: {history.current_volume:,.0f}")
    print(f"   Avg (90d): {history.avg_volume_90d:,.0f}")
    print(f"   Ratio: {history.volume_ratio:.2f}x {'🔥 HIGH' if history.is_high_volume else '💤 LOW' if history.is_low_volume else ''}")
    
    # Key levels
    print(f"\n🎯 KEY LEVELS")
    print(f"   Support: ${history.support_level:.2f}")
    print(f"   Pivot: ${history.pivot_point:.2f}")
    print(f"   Resistance: ${history.resistance_level:.2f}")
    
    # Earnings
    if history.next_earnings_date:
        print(f"\n📅 EARNINGS")
        print(f"   Next: {history.next_earnings_date} ({history.days_to_earnings} days)")
    
    # Volatility
    print(f"\n⚡ VOLATILITY")
    print(f"   Avg Daily Range: {history.avg_daily_range_pct:.2f}%")
    print(f"   Max Drawdown (1Y): {history.max_drawdown_1y:.1f}%")
    
    # Flags
    print(f"\n🚩 FLAGS")
    flags = []
    if history.is_near_52w_high: flags.append("Near 52W High")
    if history.is_near_52w_low: flags.append("Near 52W Low")
    if history.is_oversold: flags.append("Oversold")
    if history.is_overbought: flags.append("Overbought")
    if history.is_high_volume: flags.append("High Volume")
    if history.is_low_volume: flags.append("Low Volume")
    print(f"   {', '.join(flags) if flags else 'None'}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Test
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    print_stock_summary(symbol)
