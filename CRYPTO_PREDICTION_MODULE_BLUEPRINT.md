# 🔮 GHOST CRYPTO PREDICTION MODULE — BLUEPRINT

**Parallel Architecture to Stock Module**\
**Created**: October 12, 2025\
**Status**: 📋 Design Phase

______________________________________________________________________

## 🎯 EXECUTIVE SUMMARY

This blueprint outlines a **parallel crypto prediction system**that mirrors the
existing stock prediction architecture while leveraging crypto-specific data sources,
patterns, and behaviors.**Key Objectives:**1. ✅**Parallel Independence**: Separate from stock module, shared core AI

1. ✅ **24/7 Operation**: No market hours constraints
2. ✅ **Multi-Asset**: BTC, ETH, SOL, and configurable altcoins
3. ✅ **Real-Time Data**: Sub-minute price updates
4. ✅ **Cross-Asset Intelligence**: Learn from correlations between crypto and stocks

______________________________________________________________________

## 📊 ARCHITECTURE OVERVIEW

```text
┌─────────────────────────────────────────────────────────────────┐
│                     GHOST CORE ENGINE                            │
│  (Shared: AI Memory, Learning Loop, Prediction Engine)          │
└───────────────┬─────────────────────────┬───────────────────────┘
                │                         │
    ┌───────────▼──────────┐  ┌──────────▼────────────┐
    │   STOCK MODULE       │  │   CRYPTO MODULE       │
    │   (Existing)         │  │   (New - Parallel)    │
    ├──────────────────────┤  ├───────────────────────┤
    │ • Yahoo Finance      │  │ • CoinGecko           │
    │ • AlphaVantage       │  │ • Binance             │
    │ • Polygon            │  │ • Coinbase            │
    │ • yfinance           │  │ • Kraken              │
    │                      │  │ • CryptoCompare       │
    │ Market Hours: M-F    │  │ Market Hours: 24/7    │
    │ Update: 15-60min     │  │ Update: 1-5min        │
    └──────────────────────┘  └───────────────────────┘

```text

______________________________________________________________________

## 🏗️ MODULE STRUCTURE

### **File Organization**```text

/workspaces/GHOST/
├── wolf_app.py                    # Main FastAPI app (add crypto routes)
├── core/
│   ├── crypto/                    # 🆕 New crypto module
│   │   ├── __init__.py
│   │   ├── crypto_providers.py   # Price fetchers (CoinGecko, Binance, etc.)
│   │   ├── crypto_predictor.py   # Crypto-specific prediction logic
│   │   ├── crypto_portfolio.py   # Crypto portfolio management
│   │   ├── crypto_metrics.py     # Crypto-specific metrics (volatility, etc.)
│   │   ├── crypto_news.py        # Crypto news aggregation
│   │   └── defi_integrations.py  # DeFi protocol data (optional)
│   │
│   ├── prediction_engine.py      # Shared prediction core
│   ├── ai_memory.py               # Shared AI memory
│   ├── learning_loop.py           # Shared learning system
│   └── accuracy_tracker.py       # Shared accuracy tracking
│
├── data/
│   ├── crypto/                    # 🆕 Crypto data directory
│   │   ├── forecasts/             # Per-asset forecast grids
│   │   ├── history/               # Historical price data
│   │   └── watchlist.json         # Tracked crypto assets
│   │
│   └── forecast_WOLF.json         # Stock forecasts (existing)
│
└── static/
    └── crypto_dashboard.html      # 🆕 Crypto UI panel

```text

______________________________________________________________________

## 🔌 DATA PROVIDERS

###**1. CoinGecko (Primary - Free Tier)**```python

# core/crypto/crypto_providers.py

class CoinGeckoProvider:
    """
    Free tier: 50 calls/min
    Endpoint: <<<<<https://api.coingecko.com/api/v3/simple/price>>>>>
    """

    BASE_URL = "<<<<<https://api.coingecko.com/api/v3">>>>>

    async def get_price(self, coin_id: str) -> dict:
        """
        coin_id examples: 'bitcoin', 'ethereum', 'solana'

        Returns:
        {
            'price': 43251.50,
            'price_change_24h': 1250.30,
            'price_change_percentage_24h': 2.98,
            'market_cap': 845000000000,
            'total_volume': 32000000000,
            'last_updated': 1728741600
        }
        """
        url = f"{self.BASE_URL}/simple/price"
        params = {
            'ids': coin_id,
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_market_cap': 'true',
            'include_24hr_vol': 'true',
            'include_last_updated_at': 'true'
        }

        # Implementation similar to _fetch_price_alphavantage

        pass

    async def get_historical(self, coin_id: str, days: int = 7) -> list:
        """Get historical prices for pattern analysis"""
        url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'hourly'  # hourly for 1-90 days
        }
        pass

    async def get_trending(self) -> list:
        """Get trending coins (useful for watchlist expansion)"""
        url = f"{self.BASE_URL}/search/trending"
        pass

```text

###**2. Binance (Secondary - WebSocket Real-Time)**```python

class BinanceProvider:
    """
    Real-time price updates via WebSocket
    No API key needed for public data
    """

    WS_URL = "wss://stream.binance.com:9443/ws"
    REST_URL = "<<<<<https://api.binance.com/api/v3">>>>>

    async def get_price(self, symbol: str) -> dict:
        """
        symbol: 'BTCUSDT', 'ETHUSDT', etc.
        """
        url = f"{self.REST_URL}/ticker/price"
        params = {'symbol': symbol}
        pass

    async def subscribe_websocket(self, symbols: list[str], callback):
        """
        Subscribe to real-time price updates
        Updates every second
        """
        stream = "/".join([f"{s.lower()}usdt@ticker" for s in symbols])
        url = f"{self.WS_URL}/{stream}"

        # WebSocket implementation

        pass

```text

###**3. Coinbase (Tertiary - High Reliability)**```python

class CoinbaseProvider:
    """
    Backup provider with high uptime
    Public API (no key needed for spot prices)
    """

    BASE_URL = "<<<<<https://api.coinbase.com/v2">>>>>

    async def get_price(self, currency_code: str) -> dict:
        """
        currency_code: 'BTC', 'ETH', 'SOL'
        """
        url = f"{self.BASE_URL}/prices/{currency_code}-USD/spot"
        pass

```text

###**4. CryptoCompare (Quaternary - Historical Data)**```python

class CryptoCompareProvider:
    """
    Excellent for historical data and OHLCV
    Free tier: 100k calls/month
    """

    BASE_URL = "<<<<<https://min-api.cryptocompare.com/data">>>>>

    async def get_ohlcv(self, symbol: str, hours: int = 48) -> list:
        """Get hourly OHLCV data for technical analysis"""
        url = f"{self.BASE_URL}/v2/histohour"
        params = {
            'fsym': symbol,
            'tsym': 'USD',
            'limit': hours
        }
        pass

```text

###**Provider Chain Strategy**```python

# core/crypto/crypto_providers.py

async def get_crypto_price_quorum(symbol: str) -> dict:
    """
    Similar to get_wolf_price() but for crypto

    Chain:

    1. CoinGecko (primary, free, reliable)
    2. Binance (secondary, real-time)
    3. Coinbase (tertiary, high uptime)
    4. Cache fallback (if all fail)


    Returns:
    {
        'symbol': 'BTC',
        'price': 43251.50,
        'provider': 'coingecko',
        'confidence': 0.95,
        'quorum_size': 3,
        'timestamp': 1728741600
    }
    """

    providers = [
        ('coingecko', CoinGeckoProvider()),
        ('binance', BinanceProvider()),
        ('coinbase', CoinbaseProvider())
    ]

    results = []
    for name, provider in providers:
        try:
            price_data = await provider.get_price(symbol)
            if price_data and price_data.get('price'):
                results.append((name, price_data['price']))
        except Exception as e:
            LOGGER.warning(f"Crypto provider {name} failed: {e}")

    # Quorum logic: require 2+ agreeing providers within 1% spread

    if len(results) >= 2:
        prices = [r[1] for r in results]
        median_price = sorted(prices)[len(prices) // 2]
        spread = (max(prices) - min(prices)) / median_price

        if spread < 0.01:  # 1% max deviation
            return {
                'symbol': symbol,
                'price': median_price,
                'provider': results[0][0],
                'confidence': 0.95,
                'quorum_size': len(results),
                'spread': spread,
                'timestamp': time.time()
            }

    # Fallback to cache if quorum fails

    cached = _get_crypto_cache(symbol)
    if cached:
        return cached

    return None

```text

______________________________________________________________________

## 🧠 PREDICTION ENGINE

###**Crypto-Specific Prediction Logic**

```python

# core/crypto/crypto_predictor.py

class CryptoPredictionEngine:
    """
    Crypto-specific prediction with 24/7 operation
    """

    def __init__(self):
        self.volatility_threshold = 0.05  # 5% moves are normal in crypto
        self.update_interval = 300  # 5-minute updates
        self.horizon_hours = 24  # 24h forecasts (crypto moves faster)

    async def generate_prediction(self, symbol: str) -> dict:
        """
        Generate 24h crypto prediction

        Key differences from stock prediction:

        1. No market hours constraints
        2. Higher volatility acceptance
        3. Faster update cycles
        4. Include on-chain metrics (optional)


        """

        # 1. Fetch current price with quorum

        price_data = await get_crypto_price_quorum(symbol)
        if not price_data:
            raise ValueError(f"Unable to fetch price for {symbol}")

        current_price = price_data['price']

        # 2. Get historical data (7 days for pattern detection)

        history = await self._get_historical_prices(symbol, days=7)

        # 3. Calculate crypto-specific metrics

        metrics = self._calculate_crypto_metrics(history)

        # - Volatility (higher than stocks)

        # - Momentum (faster reversals)

        # - Volume trends (24h volume vs 7d average)

        # - Market cap changes

        # 4. Generate forecast points (48 points @ 30min intervals = 24h)

        forecast_points = self._generate_forecast_grid(
            current_price=current_price,
            metrics=metrics,
            horizon_hours=24,
            step_minutes=30
        )

        # 5. Determine direction and confidence

        direction, confidence = self._analyze_direction(metrics, history)

        # 6. Store prediction in shared AI memory

        prediction_id = await self._store_prediction(
            symbol=symbol,
            forecast_points=forecast_points,
            direction=direction,
            confidence=confidence,
            metrics=metrics
        )

        return {
            'prediction_id': prediction_id,
            'symbol': symbol,
            'current_price': current_price,
            'direction': direction,
            'confidence': confidence,
            'horizon_hours': 24,
            'volatility': metrics['volatility'],
            'timestamp': time.time()
        }

    def _calculate_crypto_metrics(self, history: list) -> dict:
        """
        Crypto-specific technical indicators
        """
        prices = [h['price'] for h in history]
        volumes = [h.get('volume', 0) for h in history]

        return {
            'volatility': self._calculate_volatility(prices),
            'momentum': self._calculate_momentum(prices),
            'volume_trend': self._analyze_volume(volumes),
            'support_resistance': self._find_support_resistance(prices),
            'rsi': self._calculate_rsi(prices),
            'bollinger_bands': self._calculate_bollinger(prices)
        }

    def _calculate_volatility(self, prices: list) -> float:
        """
        Standard deviation of returns
        Crypto typically 2-5x more volatile than stocks
        """
        import numpy as np
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns))

    def _generate_forecast_grid(
        self,
        current_price: float,
        metrics: dict,
        horizon_hours: int,
        step_minutes: int
    ) -> list:
        """
        Generate forecast points with crypto-adjusted confidence bands

        Crypto bands are wider due to higher volatility:

        - Stock: ±2% per day
        - Crypto: ±5% per day


        """
        import numpy as np

        volatility = metrics['volatility']
        momentum = metrics.get('momentum', 0)

        points = []
        num_steps = (horizon_hours * 60) // step_minutes

        for i in range(num_steps + 1):
            t = time.time() + (i *step_minutes* 60)

            # Base forecast with momentum

            hours_ahead = (i * step_minutes) / 60
            price = current_price *(1 + momentum* hours_ahead / 24)

            # Confidence bands (wider than stocks)

            band_width = volatility *np.sqrt(hours_ahead / 24)* current_price

            points.append({
                't': t,
                'p': round(price, 2),
                'p_low': round(price - band_width, 2),
                'p_high': round(price + band_width, 2),
                'confidence': max(0.5, 0.9 - (hours_ahead / 24) * 0.3)
            })

        return points

```text

______________________________________________________________________

## 📡 API ENDPOINTS

### **New Crypto Routes in wolf_app.py**

```python

# wolf_app.py

# ═══════════════════════════════════════════════════════════════

# CRYPTO PREDICTION MODULE

# ═══════════════════════════════════════════════════════════════

from core.crypto.crypto_providers import get_crypto_price_quorum
from core.crypto.crypto_predictor import CryptoPredictionEngine

crypto_predictor = CryptoPredictionEngine()

@APP.get("/api/crypto/price/{symbol}")
async def api_crypto_price(symbol: str, force: int = 0):
    """
    Get current crypto price with provider quorum

    Example: GET /api/crypto/price/BTC
    """
    symbol = symbol.upper()

    if force == 1:

        # Clear cache

        _crypto_cache_clear(symbol)

    price_data = await get_crypto_price_quorum(symbol)

    if not price_data:
        raise HTTPException(404, f"Unable to fetch price for {symbol}")

    return {
        "symbol": symbol,
        "price": price_data['price'],
        "provider": price_data['provider'],
        "confidence": price_data.get('confidence', 0.8),
        "quorum_size": price_data.get('quorum_size', 1),
        "timestamp": int(price_data['timestamp']),
        "24h_change": price_data.get('change_24h'),
        "market_cap": price_data.get('market_cap')
    }


@APP.post("/api/crypto/predict/run")
async def api_crypto_predict_run(
    body: PredictRunRequest,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """
    Generate 24h crypto prediction

    Example: POST /api/crypto/predict/run
    Body: {"symbol": "BTC"}
    """
    try:
        _require_bearer(
            f"Bearer {credentials.credentials}"
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    symbol = body.symbol.upper().strip()
    if not symbol:
        raise HTTPException(400, "symbol required")

    try:
        prediction = await crypto_predictor.generate_prediction(symbol)

        # Update metrics

        try:
            PROM_CRYPTO_PREDICT_RUNS.labels(symbol=symbol).inc()
        except Exception:
            pass

        return {
            "ok": True,
            "prediction_id": prediction['prediction_id'],
            "symbol": symbol,
            "current_price": prediction['current_price'],
            "direction": prediction['direction'],
            "confidence": prediction['confidence'],
            "horizon_h": prediction['horizon_hours'],
            "volatility": prediction['volatility'],
            "run_at": int(prediction['timestamp'] * 1000)
        }

    except Exception as e:
        LOGGER.error(f"Crypto prediction failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"Crypto prediction failed: {str(e)[:200]}")


@APP.get("/api/crypto/predict/series")
async def api_crypto_predict_series(
    symbol: str,
    since_hours: int = 24,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """
    Get crypto prediction series for chart overlay

    Returns forecast + actual prices aligned
    """
    try:
        _require_bearer(
            f"Bearer {credentials.credentials}"
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    symbol = symbol.upper().strip()

    # Get latest prediction from AI memory

    pred = await crypto_predictor.get_latest_prediction(symbol)

    if not pred:
        return {
            "symbol": symbol,
            "last_prediction": None,
            "forecast": [],
            "actual": []
        }

    # Get forecast points

    forecast = await crypto_predictor.get_forecast_points(pred['id'])

    # Get actual prices collected since prediction

    actual = await crypto_predictor.get_actual_points(pred['id'])

    return {
        "symbol": symbol,
        "last_prediction": {
            "id": pred['id'],
            "run_at": int(pred['timestamp'] * 1000),
            "horizon_h": pred['horizon_hours'],
            "confidence": pred['confidence'],
            "direction": pred['direction']
        },
        "forecast": forecast,
        "actual": actual,
        "accuracy": await crypto_predictor.calculate_accuracy(pred['id'])
    }


@APP.get("/api/crypto/watchlist")
async def api_crypto_watchlist():
    """
    Get crypto watchlist with live prices

    Default watchlist: BTC, ETH, SOL, BNB, ADA
    """
    watchlist = _get_crypto_watchlist()

    results = []
    for symbol in watchlist:
        try:
            price_data = await get_crypto_price_quorum(symbol)
            if price_data:
                results.append({
                    "symbol": symbol,
                    "price": price_data['price'],
                    "change_24h": price_data.get('change_24h'),
                    "change_24h_pct": price_data.get('change_24h_pct'),
                    "market_cap": price_data.get('market_cap'),
                    "provider": price_data['provider']
                })
        except Exception as e:
            LOGGER.warning(f"Failed to fetch {symbol}: {e}")

    return {
        "watchlist": results,
        "timestamp": int(time.time())
    }


@APP.get("/api/crypto/portfolio")
async def api_crypto_portfolio():
    """
    Get crypto portfolio positions

    Separate from stock portfolio
    """
    from core.crypto.crypto_portfolio import get_crypto_portfolio

    portfolio = get_crypto_portfolio()

    positions = []
    total_value = 0.0

    for pos in portfolio.get_positions():
        price_data = await get_crypto_price_quorum(pos['symbol'])
        current_price = price_data['price'] if price_data else 0

        market_value = pos['qty'] * current_price
        pnl = market_value - (pos['qty'] * pos['avg_cost'])
        pnl_pct = (pnl / (pos['qty'] *pos['avg_cost'])* 100) if pos['avg_cost'] > 0 else 0

        positions.append({
            "symbol": pos['symbol'],
            "qty": pos['qty'],
            "avg_cost": pos['avg_cost'],
            "current_price": current_price,
            "market_value": market_value,
            "pnl": pnl,
            "pnl_pct": pnl_pct
        })

        total_value += market_value

    return {
        "positions": positions,
        "total_value": total_value,
        "timestamp": int(time.time())
    }

```text

______________________________________________________________________

## 📊 PROMETHEUS METRICS

```python

# wolf_app.py (add to existing metrics section around line 4081)

# Crypto Prediction Metrics (parallel to stock metrics)

PROM_CRYPTO_PREDICT_RUNS = Counter(
    "ghost_crypto_predict_runs_total",
    "Total crypto prediction runs by symbol",
    labelnames=("symbol",)
)

PROM_CRYPTO_PREDICT_OUTCOMES = Counter(
    "ghost_crypto_predict_outcomes_total",
    "Total crypto prediction outcomes by symbol and hit status",
    labelnames=("symbol", "hit")
)

PROM_CRYPTO_PRICE_FETCH = Histogram(
    "ghost_crypto_price_fetch_seconds",
    "Crypto price fetch latency by provider",
    labelnames=("provider",)
)

PROM_CRYPTO_VOLATILITY = Gauge(
    "ghost_crypto_volatility",
    "Current volatility metric for crypto assets",
    labelnames=("symbol",)
)

PROM_CRYPTO_PORTFOLIO_VALUE = Gauge(
    "ghost_crypto_portfolio_value_usd",
    "Total crypto portfolio value in USD"
)

```text

______________________________________________________________________

## 🎨 UI INTEGRATION

### **Crypto Dashboard Panel**```html

<!-- static/crypto_dashboard.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Ghost Crypto Dashboard</title>
    <style>
        body {
            background: #0a0a0a;
            color: #00ff00;
            font-family: 'Courier New', monospace;
        }
        .crypto-card {
            background: #1a1a1a;
            border: 1px solid #00ff00;
            padding: 20px;
            margin: 10px;
            border-radius: 5px;
        }
        .price-up { color: #00ff00; }
        .price-down { color: #ff0000; }
        .chart-container {
            height: 400px;
            margin: 20px 0;
        }
    </style>
    <script src="<<<<<https://cdn.jsdelivr.net/npm/chart.js"></script>>>>>>
</head>
<body>
    <h1>🔮 GHOST CRYPTO PREDICTION DASHBOARD</h1>

    <div id="watchlist">
        <!-- Crypto watchlist cards -->
    </div>

    <div class="chart-container">
        <canvas id="cryptoChart"></canvas>
    </div>

    <div id="portfolio">
        <!-- Crypto portfolio positions -->
    </div>

    <script>
        // Real-time crypto price updates
        async function updateCryptoPrices() {
            const response = await fetch('/api/crypto/watchlist');
            const data = await response.json();

            const watchlistDiv = document.getElementById('watchlist');
            watchlistDiv.innerHTML = data.watchlist.map(crypto => `
                <div class="crypto-card">
                    <h2>${crypto.symbol}</h2>
                    <p class="${crypto.change_24h_pct >= 0 ? 'price-up' : 'price-down'}">
                        $${crypto.price.toFixed(2)}
                        (${crypto.change_24h_pct >= 0 ? '+' : ''}${crypto.change_24h_pct.toFixed(2)}%)
                    </p>
                    <p>Market Cap: $${(crypto.market_cap / 1e9).toFixed(2)}B</p>
                    <button onclick="generatePrediction('${crypto.symbol}')">
                        Generate Prediction
                    </button>
                </div>
            `).join('');
        }

        async function generatePrediction(symbol) {
            const response = await fetch('/api/crypto/predict/run', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${localStorage.getItem('ghost_token')}`
                },
                body: JSON.stringify({ symbol })
            });

            const prediction = await response.json();
            console.log('Prediction generated:', prediction);

            // Load prediction chart
            loadPredictionChart(symbol);
        }

        async function loadPredictionChart(symbol) {
            const response = await fetch(`/api/crypto/predict/series?symbol=${symbol}`);
            const data = await response.json();

            const ctx = document.getElementById('cryptoChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Forecast',
                            data: data.forecast.map(p => ({ x: p.ts, y: p.price })),
                            borderColor: '#00ff00',
                            fill: false
                        },
                        {
                            label: 'Actual',
                            data: data.actual.map(p => ({ x: p.ts, y: p.price })),
                            borderColor: '#0088ff',
                            fill: false
                        }
                    ]
                },
                options: {
                    scales: {
                        x: { type: 'time' },
                        y: { beginAtZero: false }
                    }
                }
            });
        }

        // Update prices every 30 seconds
        setInterval(updateCryptoPrices, 30000);
        updateCryptoPrices();
    </script>
</body>
</html>

```text

###**Integration in Main Wolf App**```python

# wolf_app.py (add route to serve crypto dashboard)

@APP.get("/crypto")
async def crypto_dashboard():
    """Serve crypto prediction dashboard"""
    return FileResponse("static/crypto_dashboard.html")

```text

______________________________________________________________________

## 🗄️ DATABASE SCHEMA

###**Crypto Tables (Parallel to Stock Tables)**```python

# core/crypto/crypto_predictor.py

def _init_crypto_tables():
    """
    Initialize crypto-specific database tables
    Parallel to existing stock prediction tables
    """
    conn = sqlite3.connect("ai_memory.db")
    c = conn.cursor()

    # Crypto predictions table

    c.execute("""
        CREATE TABLE IF NOT EXISTS crypto_predictions (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            run_at REAL NOT NULL,
            horizon_h INTEGER NOT NULL,
            method TEXT,
            confidence REAL,
            direction TEXT,
            volatility REAL,
            market_cap REAL,
            volume_24h REAL,
            tag TEXT,
            created_at REAL NOT NULL
        )
    """)

    # Crypto forecast points

    c.execute("""
        CREATE TABLE IF NOT EXISTS crypto_forecast_points (
            prediction_id TEXT NOT NULL,
            ts REAL NOT NULL,
            price REAL NOT NULL,
            price_low REAL,
            price_high REAL,
            confidence REAL,
            FOREIGN KEY (prediction_id) REFERENCES crypto_predictions(id)
        )
    """)

    # Crypto actual prices (for accuracy tracking)

    c.execute("""
        CREATE TABLE IF NOT EXISTS crypto_actual_points (
            prediction_id TEXT NOT NULL,
            ts REAL NOT NULL,
            price REAL NOT NULL,
            provider TEXT,
            FOREIGN KEY (prediction_id) REFERENCES crypto_predictions(id)
        )
    """)

    # Crypto portfolio positions

    c.execute("""
        CREATE TABLE IF NOT EXISTS crypto_positions (
            symbol TEXT PRIMARY KEY,
            qty REAL NOT NULL,
            avg_cost REAL NOT NULL,
            last_updated REAL NOT NULL
        )
    """)

    # Crypto price cache

    c.execute("""
        CREATE TABLE IF NOT EXISTS crypto_price_cache (
            symbol TEXT PRIMARY KEY,
            price REAL NOT NULL,
            provider TEXT,
            timestamp REAL NOT NULL,
            change_24h REAL,
            change_24h_pct REAL,
            market_cap REAL,
            volume_24h REAL
        )
    """)

    conn.commit()
    conn.close()

```text

______________________________________________________________________

## 🔄 BACKGROUND JOBS

###**Crypto Price Updater (24/7 Operation)**

```python

# core/crypto/crypto_jobs.py

import asyncio
import threading

class CryptoBackgroundJobs:
    """
    Background jobs for crypto module
    Run continuously (no market hours)
    """

    def __init__(self):
        self.running = False
        self.update_interval = 300  # 5 minutes
        self.watchlist = ['BTC', 'ETH', 'SOL', 'BNB', 'ADA']

    async def start(self):
        """Start all background jobs"""
        self.running = True

        tasks = [
            self.price_update_loop(),
            self.prediction_reconcile_loop(),
            self.portfolio_sync_loop()
        ]

        await asyncio.gather(*tasks)

    async def price_update_loop(self):
        """Update watchlist prices every 5 minutes"""
        while self.running:
            try:
                for symbol in self.watchlist:
                    price_data = await get_crypto_price_quorum(symbol)
                    if price_data:
                        _update_crypto_cache(symbol, price_data)

                        # Update Prometheus metrics

                        PROM_CRYPTO_VOLATILITY.labels(symbol=symbol).set(
                            price_data.get('volatility', 0)
                        )

                await asyncio.sleep(self.update_interval)
            except Exception as e:
                LOGGER.error(f"Crypto price update failed: {e}")
                await asyncio.sleep(60)

    async def prediction_reconcile_loop(self):
        """
        Collect actual prices for active predictions
        Similar to stock prediction reconciler
        """
        while self.running:
            try:
                active_preds = _get_active_crypto_predictions()

                for pred in active_preds:
                    symbol = pred['symbol']
                    price_data = await get_crypto_price_quorum(symbol)

                    if price_data:
                        _store_crypto_actual_point(
                            prediction_id=pred['id'],
                            ts=time.time(),
                            price=price_data['price'],
                            provider=price_data['provider']
                        )

                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                LOGGER.error(f"Crypto reconcile failed: {e}")
                await asyncio.sleep(60)

    async def portfolio_sync_loop(self):
        """Sync crypto portfolio values"""
        while self.running:
            try:
                positions = _get_crypto_positions()
                total_value = 0

                for pos in positions:
                    price_data = await get_crypto_price_quorum(pos['symbol'])
                    if price_data:
                        value = pos['qty'] * price_data['price']
                        total_value += value

                PROM_CRYPTO_PORTFOLIO_VALUE.set(total_value)

                await asyncio.sleep(600)  # Every 10 minutes
            except Exception as e:
                LOGGER.error(f"Crypto portfolio sync failed: {e}")
                await asyncio.sleep(60)

    def stop(self):
        """Stop all background jobs"""
        self.running = False

# Global instance

crypto_jobs = CryptoBackgroundJobs()

# Start jobs on module init

def start_crypto_jobs():
    """Start crypto background jobs in separate thread"""
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(crypto_jobs.start())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

```text

______________________________________________________________________

## 🧪 TESTING PLAN

### **Phase 1: Provider Testing**```bash

# Test CoinGecko provider

curl "<<<<<http://localhost:5000/api/crypto/price/BTC">>>>>

# Test with force refresh

curl "<<<<<http://localhost:5000/api/crypto/price/ETH?force=1">>>>>

# Test watchlist

curl "<<<<<http://localhost:5000/api/crypto/watchlist">>>>>

```text

###**Phase 2: Prediction Testing**```bash

# Generate BTC prediction

curl -X POST "<<<<<http://localhost:5000/api/crypto/predict/run">>>>> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${GHOST_API_TOKEN}" \
  -d '{"symbol":"BTC"}'

# Get prediction series

curl "<<<<<http://localhost:5000/api/crypto/predict/series?symbol=BTC">>>>> \
  -H "Authorization: Bearer ${GHOST_API_TOKEN}"

```text

###**Phase 3: Integration Testing**```python

# test_crypto_module.py

import pytest
from core.crypto.crypto_providers import get_crypto_price_quorum
from core.crypto.crypto_predictor import CryptoPredictionEngine

@pytest.mark.asyncio
async def test_crypto_price_quorum():
    """Test provider quorum for BTC"""
    result = await get_crypto_price_quorum('BTC')

    assert result is not None
    assert result['symbol'] == 'BTC'
    assert result['price'] > 0
    assert result['quorum_size'] >= 2
    assert result['provider'] in ['coingecko', 'binance', 'coinbase']

@pytest.mark.asyncio
async def test_crypto_prediction():
    """Test 24h crypto prediction"""
    engine = CryptoPredictionEngine()
    pred = await engine.generate_prediction('ETH')

    assert pred['symbol'] == 'ETH'
    assert pred['horizon_hours'] == 24
    assert 0.5 <= pred['confidence'] <= 1.0
    assert pred['direction'] in ['UP', 'DOWN', 'FLAT']

@pytest.mark.asyncio
async def test_crypto_volatility_higher_than_stocks():
    """Verify crypto has higher volatility metrics"""
    engine = CryptoPredictionEngine()

    # Get 7-day history

    btc_history = await engine._get_historical_prices('BTC', days=7)
    btc_metrics = engine._calculate_crypto_metrics(btc_history)

    # Crypto volatility should be > 0.02 (2% daily)

    assert btc_metrics['volatility'] > 0.02

```text

______________________________________________________________________

## 📈 PERFORMANCE TARGETS

| Metric | Target | Rationale | |--------|--------|-----------| |**Price Fetch
Latency**| < 500ms | CoinGecko + Binance are fast | |**Prediction Generation**| < 2s
| 24h forecasts (half of stock 48h) | |**Update Frequency**| 5 minutes | Crypto moves
faster than stocks | |**Quorum Requirement**| ≥2 providers | Same reliability as stock
module | |**Forecast Accuracy**| > 65% direction | Lower than stocks due to volatility
| |**Cache TTL**| 2 minutes | Shorter than stocks (5 min) | |**Concurrent Requests**| 100/min | Same as stock module |

______________________________________________________________________

## 🚀 ROLLOUT PLAN

###**Phase 1: Foundation (Week 1)**- [ ] Create `core/crypto/` module structure

- [ ] Implement CoinGecko provider (primary)
- [ ] Implement Binance provider (secondary)
- [ ] Implement provider quorum logic
- [ ] Add crypto price cache
- [ ] Test provider chain thoroughly


###**Phase 2: Prediction Engine (Week 2)**- [ ] Implement `CryptoPredictionEngine`

- [ ] Add crypto-specific metrics (volatility, momentum, RSI)
- [ ] Create 24h forecast generation
- [ ] Add database tables for crypto predictions
- [ ] Integrate with shared AI memory
- [ ] Test prediction accuracy


###**Phase 3: API Integration (Week 3)**- [ ] Add crypto API endpoints to `wolf_app.py`

- [ ] Implement `/api/crypto/price/{symbol}`
- [ ] Implement `/api/crypto/predict/run`
- [ ] Implement `/api/crypto/predict/series`
- [ ] Implement `/api/crypto/watchlist`
- [ ] Implement `/api/crypto/portfolio`
- [ ] Add Prometheus metrics
- [ ] Test all endpoints


###**Phase 4: Background Jobs (Week 4)**- [ ] Implement crypto price updater (5-min loop)

- [ ] Implement prediction reconciler
- [ ] Implement portfolio sync
- [ ] Add job health monitoring
- [ ] Test 24/7 operation
- [ ] Monitor memory/CPU usage


###**Phase 5: UI & Polish (Week 5)**- [ ] Create crypto dashboard HTML

- [ ] Add Chart.js integration
- [ ] Add real-time price updates (WebSocket optional)
- [ ] Add prediction visualization
- [ ] Style matching Ghost theme
- [ ] Mobile responsive design


###**Phase 6: Testing & Launch (Week 6)**- [ ] Comprehensive integration tests

- [ ] Load testing (100 req/min)
- [ ] 24h continuous operation test
- [ ] Accuracy tracking validation
- [ ] Documentation
- [ ] Deploy to Railway
- [ ] Monitor for 1 week


______________________________________________________________________

## 🔮 FUTURE ENHANCEMENTS

###**Phase 7: Advanced Features (Future)**1.**On-Chain Metrics**- Whale wallet movements

   - Exchange inflows/outflows
   - Network activity (transactions/day)
   - Hash rate (for BTC)


1.**DeFi Integration**- Staking APY tracking

   - Liquidity pool volumes
   - DEX trading pairs
   - Gas fees optimization


1.**Cross-Asset Correlation**- BTC vs SPY correlation

   - Crypto sector rotation
   - Risk-on/risk-off signals
   - Macro event detection


1.**Social Sentiment**- Twitter/X crypto mentions

   - Reddit r/cryptocurrency sentiment
   - Fear & Greed Index
   - Google Trends


1.**Advanced ML Models**- LSTM for time series

   - Transformer models (attention mechanism)
   - Ensemble methods
   - Reinforcement learning


______________________________________________________________________

## 📊 SUCCESS METRICS

###**Launch Criteria**- ✅ All 3 price providers operational

- ✅ Prediction accuracy > 60% in testing
- ✅ API response times < 500ms (p95)
- ✅ 24/7 background jobs stable
- ✅ Zero memory leaks in 48h test
- ✅ UI responsive and functional
- ✅ Documentation complete


###**6-Month Goals**- 🎯 Prediction accuracy > 70% direction

- 🎯 95%+ provider uptime
- 🎯 500+ successful predictions
- 🎯 5+ crypto assets supported
- 🎯 Real-time WebSocket updates
- 🎯 DeFi protocol integration
- 🎯 Cross-asset correlation analysis


______________________________________________________________________

## 💰 COST ANALYSIS

###**Free Tier Capabilities**- ✅**CoinGecko**: 50 calls/min (sufficient)

- ✅ **Binance**: Unlimited public data
- ✅ **Coinbase**: Unlimited spot prices
- ✅ **Total Cost**: $0/month


### **Paid Tier Upgrades (Optional)**-**CoinGecko Pro**: $129/month (500 calls/min)

- **CryptoCompare**: $0/month free tier (100k calls/month)
- **Messari**: $29/month (on-chain metrics)
- **Nansen**: $100+/month (whale tracking)


**Recommendation**: Start with free tier, upgrade only if rate limits hit.

______________________________________________________________________

## 🔐 SECURITY CONSIDERATIONS

1. **API Key Management**- Store in environment variables
   - Never commit to git
   - Rotate quarterly


1.**Rate Limiting**- Implement per-endpoint limits

   - Prevent abuse of free tiers
   - Queue requests intelligently


1.**Data Validation**- Validate all price data

   - Detect outliers/bad data
   - Cross-reference multiple sources


1.**Error Handling**- Graceful degradation

   - Fallback to cache
   - Log all failures


______________________________________________________________________

## 📚 DOCUMENTATION

###**Required Documentation**1.**API Reference**: Crypto endpoints with examples

1. **Integration Guide**: How to add new crypto assets
2. **Provider Guide**: Adding new price providers
3. **Troubleshooting**: Common issues and solutions
4. **Performance Tuning**: Optimization tips


______________________________________________________________________

## ✅ CHECKLIST

### **Implementation Checklist**- [ ] Create `core/crypto/` directory structure

- [ ] Implement CoinGecko provider
- [ ] Implement Binance provider
- [ ] Implement Coinbase provider
- [ ] Implement provider quorum logic
- [ ] Create `CryptoPredictionEngine` class
- [ ] Add database tables
- [ ] Add API endpoints
- [ ] Add Prometheus metrics
- [ ] Implement background jobs
- [ ] Create crypto dashboard UI
- [ ] Write integration tests
- [ ] Write documentation
- [ ] Deploy to Railway
- [ ] Monitor for 1 week


______________________________________________________________________

## 🎯 SUMMARY

This blueprint provides a**complete, production-ready architecture**for adding crypto
prediction capabilities to GHOST, fully parallel to the existing stock module.**Key Advantages:**- ✅**Independent**:
Separate from stock code, no conflicts

- ✅ **Scalable**: Easy to add new crypto assets
- ✅ **Reliable**: Multi-provider quorum system
- ✅ **24/7**: No market hours constraints
- ✅ **Cost-Effective**: 100% free tier initially
- ✅ **Intelligent**: Shares AI memory with stock predictions


**Estimated Timeline**: 6 weeks from start to production **Estimated Effort**: 40-60
hours of development **Risk Level**: Low (parallel implementation, no stock module
changes)

______________________________________________________________________

**Ready to begin implementation? Let me know and I'll start with Phase 1!** 🚀
