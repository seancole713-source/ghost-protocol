#!/usr/bin/env python3
"""
🔍 GHOST SCOUT - Find the NEXT BIG DEAL

Think like a video game scout finding new talent:
- Scan ALL assets for bullish potential
- Track which ones are MAKING MONEY
- The goal: Find the next #1 money maker

EVERY asset is competing to prove they can MAKE MONEY.
Losses are BAD. Profits are WINS.

This is survival of the fittest - only the TOP 10 money makers
get to be in Ghost's predictions.

NEW FEATURES:
- Dynamic mover detection (catches 10%+ daily gainers)
- News sentiment integration (real ✅ indicator)
- Flexible hold periods (not just 48hr)
"""

import os
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import random

# ── #111-116: Centralized Symbol Registry (replaces 4 duplicate lists) ──
from core.symbol_registry import (
    ALL_STOCKS as _REGISTRY_STOCKS,
    ALL_CRYPTO as _REGISTRY_CRYPTO,
    get_coingecko_id,
    get_all_coingecko_ids,
    is_crypto,
)

# ── Brain v3: 25 cognitive abilities ──
from core.ghost_brain import GhostBrain, BrainDecision
from core.brain_data import BrainContext, load_brain_context, build_context_from_accuracy_data

LOGGER = logging.getLogger("ghost.scout")

# Direction mapping: Scout uses BUY/SELL, Brain uses UP/DOWN
_DIR_TO_BRAIN = {"BUY": "UP", "SELL": "DOWN", "HOLD": "HOLD"}
_DIR_FROM_BRAIN = {"UP": "BUY", "DOWN": "SELL"}


# ALL ASSETS IN THE GAME - Everyone competes!
# Source of truth: core/symbol_registry.py — NO duplicate lists here
ALL_STOCKS = list(_REGISTRY_STOCKS)
ALL_CRYPTO = list(_REGISTRY_CRYPTO)


def fetch_daily_movers(min_gain_pct: float = 5.0) -> List[Dict]:
    """
    🚀 DYNAMIC MOVER DETECTION
    
    Fetch today's biggest gainers that Ghost might be missing.
    This catches stocks like Nextpower +16%, Seagate +15% that
    aren't in our static list.
    
    Uses Yahoo Finance screener API.
    
    Returns: List of {symbol, name, change_pct, price}
    """
    movers = []
    
    try:
        # Yahoo Finance day gainers
        url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
        params = {
            "scrIds": "day_gainers",
            "count": 25
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
            
            for q in quotes:
                change_pct = q.get("regularMarketChangePercent", 0)
                if change_pct >= min_gain_pct:
                    movers.append({
                        "symbol": q.get("symbol", ""),
                        "name": q.get("shortName", q.get("symbol", "")),
                        "change_pct": round(change_pct, 2),
                        "price": q.get("regularMarketPrice", 0),
                        "volume": q.get("regularMarketVolume", 0)
                    })
            
            LOGGER.info(f"🚀 [MOVERS] Found {len(movers)} stocks up {min_gain_pct}%+ today")
    except Exception as e:
        LOGGER.error(f"🚀 [MOVERS] Fetch error: {e}")
    
    return movers


def get_news_sentiment_for_symbol(symbol: str) -> Dict:
    """
    📰 Get news sentiment for a symbol.
    
    Returns sentiment data that can influence predictions.
    Used to determine if ✅ indicator should show.
    """
    try:
        from core.news_sentiment import fetch_news_sentiment
        news_data = fetch_news_sentiment(symbol, limit=5)
        
        return {
            "has_news": news_data.get("article_count", 0) > 0,
            "sentiment_score": news_data.get("sentiment_score", 0),
            "sentiment_label": news_data.get("sentiment_label", "NEUTRAL"),
            "article_count": news_data.get("article_count", 0),
            "news_influenced": abs(news_data.get("sentiment_score", 0)) > 0.2  # Strong sentiment
        }
    except Exception as e:
        LOGGER.debug(f"News fetch failed for {symbol}: {e}")
        return {
            "has_news": False,
            "sentiment_score": 0,
            "sentiment_label": "NEUTRAL",
            "article_count": 0,
            "news_influenced": False
        }


class GhostScout:
    """
    🔍 The Scout finds MONEY MAKERS
    
    Every day, the scout:
    1. Looks at ALL assets (static + dynamic movers!)
    2. Evaluates bullish potential with NEWS SENTIMENT
    3. Records predictions for EVERYONE
    4. Later: See who MADE MONEY
    
    The ones who make money = TOP 10
    The ones who lose money = Stay benched
    
    NO BLACKLIST. Everyone gets a fair shot.
    Prove yourself through PROFITS.
    
    NEW: Dynamic mover detection catches 10%+ gainers!
    NEW: News sentiment integration for ✅ indicator!
    """
    
    def __init__(self, include_dynamic_movers: bool = True):
        self.stocks = ALL_STOCKS[:]
        self.crypto = ALL_CRYPTO[:]
        self.include_dynamic_movers = include_dynamic_movers
        self.dynamic_movers_added = []
        
        # #41: Reusable HTTP session — keeps TCP connections alive across calls
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        
        # Add dynamic movers to stock list
        if include_dynamic_movers:
            self._add_dynamic_movers()
        
        LOGGER.info(f"🔍 [SCOUT] Ready to find money makers!")
        LOGGER.info(f"   {len(self.stocks)} stocks competing")
        LOGGER.info(f"   {len(self.crypto)} crypto competing")
        if self.dynamic_movers_added:
            LOGGER.info(f"   🚀 Dynamic movers added: {self.dynamic_movers_added}")
    
    # ══════════════════════════════════════════════════════════════
    # BRAIN v3 INTEGRATION — Replaces hardcoded BUY/0.55 defaults
    # ══════════════════════════════════════════════════════════════

    def _load_brain_context_sync(self) -> Optional[BrainContext]:
        """
        Load BrainContext from PostgreSQL (sync bridge for async loader).

        Falls back gracefully:
          1. Full async load via load_brain_context() (rich data)
          2. Empty BrainContext (Brain still works with basic data)
          3. None (Brain disabled entirely)
        """
        import asyncio

        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            LOGGER.debug("[SCOUT] No DATABASE_URL — Brain will use basic mode")
            return BrainContext()  # Empty context, Brain passes through

        # Gather live market data for Brain's regime/F&G/cross-asset abilities
        market_data = self._gather_market_data()

        try:
            # Prefer existing event loop (if inside async runtime)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Already in async context — use thread bridge
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    ctx = pool.submit(
                        asyncio.run,
                        load_brain_context(db_url, market_data=market_data),
                    ).result(timeout=30)
            else:
                ctx = asyncio.run(
                    load_brain_context(db_url, market_data=market_data)
                )

            LOGGER.info(
                f"🧠 [SCOUT] Brain context loaded: {len(ctx.symbols)} symbols, "
                f"regime={ctx.market_regime}, F&G={ctx.fear_greed_index}"
            )
            return ctx

        except Exception as exc:
            LOGGER.warning(f"🧠 [SCOUT] Brain context load failed: {exc} — using basic mode")
            return BrainContext()

    def _gather_market_data(self) -> dict:
        """Gather live market data for BrainContext (VIX, Fear & Greed, cross-asset)."""
        data = {}
        try:
            # Fear & Greed Index
            resp = self._session.get(
                "https://api.alternative.me/fng/?limit=1", timeout=5
            )
            if resp.status_code == 200:
                fg = resp.json().get("data", [{}])[0]
                data["fear_greed"] = int(fg.get("value", 50))
                classification = fg.get("value_classification", "Neutral").lower()
                if "extreme fear" in classification:
                    data["regime"] = "panic"
                elif "fear" in classification:
                    data["regime"] = "fear"
                elif "extreme greed" in classification:
                    data["regime"] = "elevated"
                elif "greed" in classification:
                    data["regime"] = "calm"
                else:
                    data["regime"] = "neutral"
        except Exception as e:
            LOGGER.warning(f"[SCOUT] Fear & Greed API failed — Brain regime/F&G abilities disabled: {e}")

        try:
            # BTC 24h change for cross-asset signal
            resp = self._session.get(
                "https://api.coingecko.com/api/v3/simple/price"
                "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true",
                timeout=5,
            )
            if resp.status_code == 200:
                cg = resp.json()
                data["btc_24h"] = cg.get("bitcoin", {}).get("usd_24h_change", 0.0)
                data["eth_24h"] = cg.get("ethereum", {}).get("usd_24h_change", 0.0)
        except Exception as e:
            LOGGER.warning(f"[SCOUT] CoinGecko cross-asset API failed — Brain cross-asset ability disabled: {e}")

        return data

    def _apply_brain_analysis(
        self, raw_predictions: dict
    ) -> dict:
        """
        Run Brain v3's 25 cognitive abilities on all raw predictions.

        Args:
            raw_predictions: {symbol: {"direction": "BUY"/"SELL", "confidence": float, ...}}

        Returns:
            {symbol: BrainDecision} — Brain's verdicts with adjusted direction,
            confidence, tier, and action (SEND/EXCLUDE/INVERT).
        """
        if not raw_predictions:
            return {}

        # Load rich context from DB (accuracy history, streaks, calibration, etc.)
        brain_context = self._load_brain_context_sync()

        # Build predictions dict in Brain's format: {symbol: {direction: UP/DOWN, confidence: float}}
        brain_input = {}
        for sym, pred in raw_predictions.items():
            scout_dir = pred.get("direction", "BUY")
            brain_dir = _DIR_TO_BRAIN.get(scout_dir, "UP")
            # Skip HOLD predictions — no conviction, don't feed to brain
            if brain_dir == "HOLD":
                LOGGER.debug(f"[BRAIN] Skipping {sym}: HOLD direction — no conviction")
                continue
            brain_input[sym] = {
                "direction": brain_dir,
                "confidence": pred.get("confidence", 0.55),
            }

        # Run Brain v3 batch analysis (25 abilities: invert, exclude, boost, calibrate, etc.)
        brain = GhostBrain()
        decisions = brain.analyze_batch(
            predictions=brain_input,
            context=brain_context,
        )

        # Log Brain summary
        stats = brain._cycle_stats
        LOGGER.info(
            f"🧠 [BRAIN] Batch analysis complete: "
            f"{stats['analyzed']} analyzed, {stats['sent']} sent, "
            f"{stats['excluded']} excluded, {stats['inverted']} inverted, "
            f"{stats['boosted']} boosted, {stats['penalized']} penalized"
        )

        return decisions

    def _apply_decision_to_prediction(
        self, pred: dict, decision: BrainDecision
    ) -> dict:
        """
        Apply Brain's decision to a raw prediction dict.

        Overwrites direction and confidence with Brain's analysis.
        Recalculates target_price based on adjusted confidence.
        Stores Brain metadata for downstream visibility.
        """
        entry = pred["entry_price"]

        # Map Brain direction back to Scout format
        new_dir = _DIR_FROM_BRAIN.get(decision.direction, pred["direction"])
        new_conf = decision.confidence

        pred["direction"] = new_dir
        pred["confidence"] = new_conf

        # Recalculate target with Brain's adjusted confidence
        if new_dir == "BUY":
            pred["target_price"] = entry * (1 + (new_conf * 0.08))
        else:
            pred["target_price"] = entry * (1 - (new_conf * 0.08))

        # Store Brain metadata for visibility in cockpit/notifications
        pred["brain_tier"] = decision.tier
        pred["brain_action"] = decision.action
        pred["brain_accuracy"] = decision.brain_accuracy
        pred["brain_inverted"] = decision.inverted
        pred["brain_reasons"] = decision.reasons[:5]  # Top 5 reasons
        pred["brain_expected_value"] = decision.expected_value

        return pred

    def _add_dynamic_movers(self):
        """Add today's biggest gainers to the scout list with BULLISH bias"""
        try:
            movers = fetch_daily_movers(min_gain_pct=5.0)  # 5%+ gainers
            for mover in movers[:20]:  # Max 20 dynamic adds
                symbol = mover.get("symbol", "").replace(".US", "").split(".")[0]
                if symbol and symbol not in self.stocks:
                    self.stocks.append(symbol)
                    self.dynamic_movers_added.append(f"{symbol} (+{mover['change_pct']}%)")
                    # Track that this is a BIG GAINER - should be BUY not SELL!
                    self._bullish_movers = getattr(self, '_bullish_movers', set())
                    self._bullish_movers.add(symbol)
            
            if self.dynamic_movers_added:
                LOGGER.info(f"🚀 [SCOUT] Added {len(self.dynamic_movers_added)} dynamic movers to watchlist!")
        except Exception as e:
            LOGGER.error(f"🚀 [SCOUT] Dynamic mover fetch failed: {e}")
    
    def scout_all(self, use_news: bool = True) -> Dict:
        """
        Run a full scouting cycle with Brain v3 analysis.

        Three-phase pipeline (sequential version of scout_all_fast):
          Phase 1: Collect raw predictions for every asset
          Phase 2: Brain v3 batch analysis (25 cognitive abilities)
          Phase 3: Apply Brain decisions and record trades

        Args:
            use_news: If True, fetch news sentiment for each symbol (slower but accurate ✅)
        """
        from core.money_game_engine import get_money_game

        game = get_money_game()
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "stocks_scouted": 0,
            "crypto_scouted": 0,
            "news_influenced_count": 0,
            "dynamic_movers_found": len(self.dynamic_movers_added),
            "trades_recorded": [],
            "brain_excluded": 0,
            "brain_inverted": 0,
            "brain_boosted": 0,
        }

        try:
            import wolf_app
            latest_predictions = wolf_app._LATEST_PREDICTIONS
        except ImportError:
            latest_predictions = {}

        LOGGER.info("🔍 [SCOUT] Starting full scouting run...")

        # ═══════════════════════════════════════════════════════
        # PHASE 1: Collect raw predictions
        # ═══════════════════════════════════════════════════════
        raw_predictions = {}  # {symbol: pred_dict}
        symbol_types = {}     # {symbol: asset_type}

        for symbol in self.stocks:
            try:
                prediction = self._make_prediction(symbol, "stock", use_news=use_news)
                if prediction:
                    raw_predictions[symbol] = prediction
                    symbol_types[symbol] = "stock"
            except Exception as e:
                LOGGER.error(f"🔍 [SCOUT] Error scouting {symbol}: {e}")

        for symbol in self.crypto:
            try:
                prediction = self._make_prediction(symbol, "crypto", use_news=use_news)
                if prediction:
                    raw_predictions[symbol] = prediction
                    symbol_types[symbol] = "crypto"
            except Exception as e:
                LOGGER.error(f"🔍 [SCOUT] Error scouting {symbol}: {e}")

        LOGGER.info(f"🔍 [SCOUT] Phase 1 complete: {len(raw_predictions)} raw predictions")

        # ═══════════════════════════════════════════════════════
        # PHASE 2: Brain v3 batch analysis
        # ═══════════════════════════════════════════════════════
        brain_decisions = self._apply_brain_analysis(raw_predictions)

        # ═══════════════════════════════════════════════════════
        # PHASE 3: Apply Brain decisions and record trades
        # ═══════════════════════════════════════════════════════
        for sym, pred in raw_predictions.items():
            atype = symbol_types[sym]
            decision = brain_decisions.get(sym)

            if decision:
                if decision.action == "EXCLUDE":
                    results["brain_excluded"] += 1
                    continue
                pred = self._apply_decision_to_prediction(pred, decision)
                if decision.inverted:
                    results["brain_inverted"] += 1
                if decision.tier in ("🟢HOT", "🔥FIRE"):
                    results["brain_boosted"] += 1

            trade_id = game.record_trade(
                symbol=sym,
                asset_type=atype,
                direction=pred["direction"],
                entry_price=pred["entry_price"],
                target_price=pred["target_price"],
                confidence=pred["confidence"],
            )
            if trade_id > 0:
                if atype == "stock":
                    results["stocks_scouted"] += 1
                else:
                    results["crypto_scouted"] += 1
                if pred.get("news_influenced"):
                    results["news_influenced_count"] += 1
                results["trades_recorded"].append({
                    "trade_id": trade_id,
                    "symbol": sym,
                    "direction": pred["direction"],
                    "brain_tier": pred.get("brain_tier", "⚪NEUTRAL"),
                    "news_influenced": pred.get("news_influenced", False),
                })
                if latest_predictions is not None:
                    latest_predictions[sym] = {
                        "symbol": sym,
                        "direction": pred["direction"],
                        "confidence": pred["confidence"],
                        "price": pred["entry_price"],
                        "current_price": pred["entry_price"],
                        "entry_price": pred["entry_price"],
                        "target_price": pred["target_price"],
                        "asset_type": atype,
                        "run_at": time.time(),
                        "source": "money_game_scout",
                        "news_influenced": pred.get("news_influenced", False),
                        "sentiment_score": pred.get("sentiment_score", 0),
                        "sentiment_label": pred.get("sentiment_label", "NEUTRAL"),
                        "hold_hours": pred.get("hold_hours", 48),
                        "hold_reason": pred.get("hold_reason", "default"),
                        "brain_tier": pred.get("brain_tier", "⚪NEUTRAL"),
                        "brain_action": pred.get("brain_action", "SEND"),
                        "brain_accuracy": pred.get("brain_accuracy", 0.0),
                        "brain_inverted": pred.get("brain_inverted", False),
                    }

        LOGGER.info(f"🔍 [SCOUT] Scouting complete!")
        LOGGER.info(f"   Stocks: {results['stocks_scouted']}")
        LOGGER.info(f"   Crypto: {results['crypto_scouted']}")
        LOGGER.info(f"   📰 News-influenced: {results['news_influenced_count']}")
        LOGGER.info(f"   🧠 Brain: {results['brain_excluded']} excluded, "
                    f"{results['brain_inverted']} inverted, "
                    f"{results['brain_boosted']} boosted")
        if self.dynamic_movers_added:
            LOGGER.info(f"   🚀 Dynamic movers: {len(self.dynamic_movers_added)}")
        if latest_predictions:
            LOGGER.info(f"   Predictions in memory: {len(latest_predictions)}")

        return results

    # ── #41-43: Concurrent scout with ThreadPoolExecutor ──────────────
    def scout_all_fast(self, use_news: bool = True, max_workers: int = 10) -> Dict:
        """
        Run a full scouting cycle with PARALLEL price fetches + BRAIN v3 analysis.

        Three-phase pipeline:
          Phase 1: Parallel raw prediction collection (ThreadPoolExecutor)
          Phase 2: Brain v3 batch analysis (25 cognitive abilities)
          Phase 3: Apply Brain decisions and record trades

        Brain can INVERT direction, EXCLUDE weak symbols, BOOST confidence
        for proven performers, apply calibration, circuit breaker, etc.

        Args:
            use_news: If True, fetch news sentiment (slower but accurate).
            max_workers: Max concurrent HTTP requests (default 10,
                         keeps CoinGecko/Polygon rate limits happy).
        """
        from core.money_game_engine import get_money_game

        game = get_money_game()
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "stocks_scouted": 0,
            "crypto_scouted": 0,
            "news_influenced_count": 0,
            "dynamic_movers_found": len(self.dynamic_movers_added),
            "trades_recorded": [],
            "brain_excluded": 0,
            "brain_inverted": 0,
            "brain_boosted": 0,
            "mode": "fast_parallel_brain_v3",
        }

        try:
            import wolf_app
            latest_predictions = wolf_app._LATEST_PREDICTIONS
        except ImportError:
            latest_predictions = {}

        LOGGER.info(f"⚡ [SCOUT-FAST] Starting parallel scouting ({max_workers} workers)...")

        # ═══════════════════════════════════════════════════════
        # PHASE 1: Collect raw predictions in parallel
        # ═══════════════════════════════════════════════════════
        work = [(s, "stock") for s in self.stocks] + [(s, "crypto") for s in self.crypto]
        raw_predictions = {}  # {symbol: {direction, confidence, entry_price, ...}}
        symbol_types = {}     # {symbol: asset_type}

        def _scout_one(item):
            """Scout a single symbol — runs in a thread."""
            sym, atype = item
            try:
                pred = self._make_prediction(sym, atype, use_news=use_news)
                if pred:
                    return (sym, atype, pred, None)
            except Exception as exc:
                return (sym, atype, None, exc)
            return (sym, atype, None, None)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_scout_one, w): w for w in work}
            for future in as_completed(futures):
                sym, atype, pred, exc = future.result()
                if exc:
                    LOGGER.error(f"⚡ [SCOUT-FAST] Error scouting {sym}: {exc}")
                    continue
                if not pred:
                    continue
                raw_predictions[sym] = pred
                symbol_types[sym] = atype

        LOGGER.info(f"⚡ [SCOUT-FAST] Phase 1 complete: {len(raw_predictions)} raw predictions")

        # ═══════════════════════════════════════════════════════
        # PHASE 2: Brain v3 batch analysis (25 cognitive abilities)
        # ═══════════════════════════════════════════════════════
        brain_decisions = self._apply_brain_analysis(raw_predictions)

        # ═══════════════════════════════════════════════════════
        # PHASE 3: Apply Brain decisions and record trades
        # ═══════════════════════════════════════════════════════
        for sym, pred in raw_predictions.items():
            atype = symbol_types[sym]
            decision = brain_decisions.get(sym)

            if decision:
                # Brain says EXCLUDE → skip this symbol entirely
                if decision.action == "EXCLUDE":
                    results["brain_excluded"] += 1
                    LOGGER.debug(f"🧠 [BRAIN] EXCLUDED {sym}: {decision.reasons[:2]}")
                    continue

                # Apply Brain's direction, confidence, and metadata
                pred = self._apply_decision_to_prediction(pred, decision)

                if decision.inverted:
                    results["brain_inverted"] += 1
                if decision.tier in ("🟢HOT", "🔥FIRE"):
                    results["brain_boosted"] += 1

            # Record trade with Brain-enhanced prediction
            trade_id = game.record_trade(
                symbol=sym,
                asset_type=atype,
                direction=pred["direction"],
                entry_price=pred["entry_price"],
                target_price=pred["target_price"],
                confidence=pred["confidence"],
            )
            if trade_id <= 0:
                continue

            if atype == "stock":
                results["stocks_scouted"] += 1
            else:
                results["crypto_scouted"] += 1
            if pred.get("news_influenced"):
                results["news_influenced_count"] += 1
            results["trades_recorded"].append({
                "trade_id": trade_id,
                "symbol": sym,
                "direction": pred["direction"],
                "brain_tier": pred.get("brain_tier", "⚪NEUTRAL"),
                "news_influenced": pred.get("news_influenced", False),
            })

            if latest_predictions is not None:
                latest_predictions[sym] = {
                    "symbol": sym,
                    "direction": pred["direction"],
                    "confidence": pred["confidence"],
                    "price": pred["entry_price"],
                    "current_price": pred["entry_price"],
                    "entry_price": pred["entry_price"],
                    "target_price": pred["target_price"],
                    "asset_type": atype,
                    "run_at": time.time(),
                    "source": "money_game_scout_fast",
                    "news_influenced": pred.get("news_influenced", False),
                    "sentiment_score": pred.get("sentiment_score", 0),
                    "sentiment_label": pred.get("sentiment_label", "NEUTRAL"),
                    "hold_hours": pred.get("hold_hours", 48),
                    "hold_reason": pred.get("hold_reason", "default"),
                    # Brain v3 metadata — visible in cockpit
                    "brain_tier": pred.get("brain_tier", "⚪NEUTRAL"),
                    "brain_action": pred.get("brain_action", "SEND"),
                    "brain_accuracy": pred.get("brain_accuracy", 0.0),
                    "brain_inverted": pred.get("brain_inverted", False),
                }

        LOGGER.info(f"⚡ [SCOUT-FAST] Scouting complete!")
        LOGGER.info(f"   Stocks: {results['stocks_scouted']}")
        LOGGER.info(f"   Crypto: {results['crypto_scouted']}")
        LOGGER.info(f"   📰 News-influenced: {results['news_influenced_count']}")
        LOGGER.info(f"   🧠 Brain: {results['brain_excluded']} excluded, "
                    f"{results['brain_inverted']} inverted, "
                    f"{results['brain_boosted']} boosted")
        if latest_predictions:
            LOGGER.info(f"   Predictions in memory: {len(latest_predictions)}")

        return results
    
    def _make_prediction(self, symbol: str, asset_type: str, use_news: bool = True) -> Optional[Dict]:
        """
        Make a prediction for a symbol.

        Full intelligence pipeline:
        1. Technical analysis (SMA + RSI)
        2. News sentiment (Alpha Vantage)
        3. 🧠 Intelligence Hub — 20 systems aggregated
        4. Dynamic hold period calculation
        """
        try:
            # Get current price
            current_price = self._get_current_price(symbol, asset_type)
            if not current_price:
                return None

            # Fetch price history (needed by technical + intelligence hub)
            price_history = self._fetch_price_history(symbol, asset_type)

            # Get base prediction from the engine
            prediction = self._get_prediction_from_engine(symbol, asset_type, current_price)
            if not prediction:
                return None

            # Add news sentiment if enabled (this makes ✅ real!)
            if use_news:
                news_data = get_news_sentiment_for_symbol(symbol)
                prediction["news_influenced"] = news_data.get("news_influenced", False)
                prediction["sentiment_score"] = news_data.get("sentiment_score", 0)
                prediction["sentiment_label"] = news_data.get("sentiment_label", "NEUTRAL")

                # Boost confidence if news agrees with direction
                if news_data["news_influenced"]:
                    sentiment = news_data["sentiment_score"]
                    if (prediction["direction"] == "BUY" and sentiment > 0) or \
                       (prediction["direction"] == "SELL" and sentiment < 0):
                        prediction["confidence"] = min(0.85, prediction["confidence"] * 1.15)
                        LOGGER.info(f"📰 [NEWS] {symbol}: News confirms {prediction['direction']} (sentiment: {sentiment:.2f})")

            # ═══════════════════════════════════════════════════════
            # 🧠 INTELLIGENCE HUB — 20 systems aggregated
            # ═══════════════════════════════════════════════════════
            try:
                from core.intelligence_hub import get_intelligence_hub
                hub = get_intelligence_hub()
                report = hub.analyze(
                    symbol=symbol,
                    direction=prediction["direction"],
                    confidence=prediction["confidence"],
                    entry_price=current_price,
                    asset_type=asset_type,
                    price_history=price_history,
                )

                # Store intelligence metadata on prediction
                prediction["intel_active_systems"] = report.active_systems
                prediction["intel_total_systems"] = report.total_systems
                prediction["intel_news_risk"] = report.news_risk
                prediction["intel_direction_adj"] = report.direction_adjustment
                prediction["intel_confidence_adj"] = report.confidence_adjustment
                prediction["intel_trust_boost"] = report.trust_boost

                # Apply direction adjustment
                if report.should_block:
                    LOGGER.info(f"🛑 [HUB] {symbol}: BLOCKED — {report.block_reason}")
                    return None

                if report.direction_adjustment == "FLIP":
                    old_dir = prediction["direction"]
                    prediction["direction"] = "SELL" if old_dir == "BUY" else "BUY"
                    LOGGER.info(f"🔄 [HUB] {symbol}: Direction FLIPPED {old_dir} → {prediction['direction']}")
                elif report.direction_adjustment == "WEAKEN":
                    LOGGER.info(f"⚠️ [HUB] {symbol}: Direction WEAKENED (signals disagree)")

                # Apply confidence adjustment
                old_conf = prediction["confidence"]
                prediction["confidence"] += report.confidence_adjustment
                prediction["confidence"] += report.trust_boost
                prediction["confidence"] = max(0.10, min(0.92, prediction["confidence"]))

                if abs(report.confidence_adjustment) > 0.01 or report.trust_boost > 0:
                    LOGGER.info(f"🧠 [HUB] {symbol}: Confidence {old_conf:.2f} → {prediction['confidence']:.2f} "
                                f"(adj={report.confidence_adjustment:+.2f}, trust={report.trust_boost:+.2f}, "
                                f"systems={report.active_systems}/{report.total_systems})")

                # Apply dynamic exit levels (override basic SL/TP)
                if report.exit_levels:
                    if "target_price" in report.exit_levels:
                        prediction["target_price"] = report.exit_levels["target_price"]
                    if "stop_loss" in report.exit_levels:
                        prediction["stop_loss"] = report.exit_levels["stop_loss"]
                    if "trailing_stop_pct" in report.exit_levels:
                        prediction["trailing_stop_pct"] = report.exit_levels["trailing_stop_pct"]

                # Store regime info
                if report.regime_info:
                    prediction["market_regime"] = report.regime_info.get("regime", "UNKNOWN")

                # Log signal summary
                active_signals = [s for s in report.signals if s.active]
                if active_signals:
                    signal_summary = ", ".join(
                        f"{s.source}={s.direction}@{s.confidence:.0%}"
                        for s in active_signals[:6]
                    )
                    LOGGER.info(f"🧠 [HUB] {symbol}: Signals: {signal_summary}")

            except Exception as e:
                LOGGER.warning(f"🧠 [HUB] Intelligence hub error for {symbol}: {e}")
                prediction["intel_active_systems"] = 0
                prediction["intel_total_systems"] = 0

            # Calculate smart hold period based on volatility and momentum
            prediction["hold_hours"] = self._calculate_hold_period(symbol, asset_type, prediction)
            prediction["hold_reason"] = self._get_hold_reason(prediction["hold_hours"])

            return prediction
        except Exception as e:
            LOGGER.error(f"🔍 [SCOUT] Prediction error for {symbol}: {e}")
            return None
    
    def _calculate_hold_period(self, symbol: str, asset_type: str, prediction: Dict) -> int:
        """
        Calculate optimal hold period based on asset characteristics.
        
        ENHANCED: Returns 1-7 days based on:
        - Asset volatility (crypto = shorter, stocks = longer)
        - News catalyst (hot news = 1-2 days max)
        - Confidence level (high confidence = can wait longer)
        - RSI momentum exhaustion (overbought/oversold = shorter)
        - Trend strength from prediction data
        
        Returns hours to hold (24-168h = 1-7 days).
        """
        # Base hold in DAYS (1-7 scale)
        base_days = 3  # Default 3-day swing
        
        # Asset type adjustment
        if asset_type == "crypto":
            base_days = 2  # Crypto: faster moves, 1-3 day range
        else:
            base_days = 4  # Stocks: slower, 2-5 day range typical
        
        # News catalyst - shorter hold (ride the wave, don't overstay)
        if prediction.get("news_influenced"):
            base_days = min(base_days, 2)  # News plays = 1-2 days max
        
        # Confidence adjustment
        conf = prediction.get("confidence", 0.5)
        if conf >= 0.85:
            base_days += 1  # Very strong signal = can hold longer for bigger target
        elif conf >= 0.7:
            pass  # Normal confidence = keep base
        elif conf >= 0.5:
            base_days = max(1, base_days - 1)  # Weak signal = shorter hold
        else:
            base_days = 1  # Very weak = quick scalp only
        
        # RSI-based momentum exhaustion (from prediction if available)
        rsi = prediction.get("rsi", 50)
        # FIX: Scout uses BUY/SELL vocabulary, not UP/DOWN (Brain format)
        direction = prediction.get("direction", "BUY")
        
        if direction == "BUY" and rsi > 75:
            # Overbought on bullish = momentum exhausting, take profits soon
            base_days = max(1, base_days - 1)
        elif direction == "SELL" and rsi < 25:
            # Oversold on bearish = bounce coming, don't overstay short
            base_days = max(1, base_days - 1)
        
        # Volatility adjustment (if price change % is extreme)
        price_change = abs(prediction.get("price_change_pct", 0))
        if price_change > 10:
            # Big move already happened - shorter hold to lock in gains
            base_days = max(1, base_days - 1)
        elif price_change < 2:
            # Slow mover - needs more time to develop
            base_days = min(7, base_days + 1)
        
        # Clamp to 1-7 days
        final_days = max(1, min(7, base_days))
        
        # Store the day estimate for downstream use
        prediction["hold_days"] = final_days
        prediction["hold_estimate"] = self._format_hold_estimate(final_days)
        
        # Return hours for backward compatibility (existing code expects hours)
        return final_days * 24
    
    def _format_hold_estimate(self, days: int) -> str:
        """Format hold estimate like '2-3 days' or '5-7 days'"""
        if days == 1:
            return "1-2 days"
        elif days == 2:
            return "2-3 days"
        elif days == 3:
            return "3-4 days"
        elif days == 4:
            return "4-5 days"
        elif days == 5:
            return "5-6 days"
        elif days == 6:
            return "5-7 days"
        else:  # 7
            return "6-7 days"
    
    def _get_hold_reason(self, hours: int) -> str:
        """Get human-readable hold reason based on days"""
        days = hours // 24
        if days <= 1:
            return "day_trade"  # 1 day
        elif days <= 2:
            return "momentum_trade"  # 1-2 days
        elif days <= 3:
            return "swing_trade"  # 2-3 days
        elif days <= 5:
            return "position_trade"  # 4-5 days
        else:
            return "trend_trade"  # 6-7 days
    
    def _get_current_price(self, symbol: str, asset_type: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            if asset_type == "crypto":
                return self._get_crypto_price(symbol)
            else:
                return self._get_stock_price(symbol)
        except Exception as e:
            LOGGER.debug(f"Price fetch failed for {symbol}: {e}")
            return None
    
    def _get_crypto_price(self, symbol: str) -> Optional[float]:
        """Get crypto price from CoinGecko"""
        
        # Use centralized symbol registry (#84: kill duplicate CoinGecko maps)
        cg_id = get_coingecko_id(symbol)
        if not cg_id:
            # Fallback for unknown symbols
            cg_id = symbol.lower()
        
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
        
        # Retry once on 429 rate-limit (CoinGecko free tier)
        for attempt in range(2):
            try:
                resp = self._session.get(url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if cg_id in data:
                        return data[cg_id]["usd"]
                elif resp.status_code == 429 and attempt == 0:
                    import time
                    LOGGER.debug(f"[SCOUT] CoinGecko 429 for {symbol}, retrying in 2s...")
                    time.sleep(2)
                    continue
                # Non-retryable error
                break
            except Exception:
                break
        
        return None
    
    def _get_stock_price(self, symbol: str) -> Optional[float]:
        """Get stock price from Polygon API (Yahoo blocks server requests)"""
        polygon_key = os.getenv("POLYGON_API_KEY")
        if not polygon_key:
            LOGGER.warning(f"No POLYGON_API_KEY - cannot fetch {symbol}")
            return None
        
        try:
            # Polygon prev close endpoint - most reliable
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={polygon_key}"
            resp = self._session.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    price = results[0].get("c")  # close price
                    if price:
                        return float(price)
        except Exception as e:
            LOGGER.debug(f"Polygon price fetch failed for {symbol}: {e}")
        
        return None
    
    def _get_prediction_from_engine(self, symbol: str, asset_type: str, current_price: float) -> Dict:
        """
        Get a real prediction from the Ghost engine.
        
        This integrates with the actual prediction system.
        Returns: direction, target_price, confidence
        """
        # Try to use the actual prediction engine
        try:
            if asset_type == "crypto":
                from core.multi_crypto_predictor import MultiCryptoPredictor
                predictor = MultiCryptoPredictor()
                result = predictor.predict_symbol(symbol)
                
                if result:
                    direction = "BUY" if result.get("confidence", 0) > 0 else "SELL"
                    confidence = abs(result.get("confidence", 0.5))
                    
                    # Calculate target based on confidence
                    if direction == "BUY":
                        target = current_price * (1 + (confidence * 0.1))  # Up to 10% gain
                    else:
                        target = current_price * (1 - (confidence * 0.1))  # Up to 10% drop
                    
                    return {
                        "direction": direction,
                        "entry_price": current_price,
                        "target_price": target,
                        "confidence": confidence
                    }
        except ImportError:
            pass
        except Exception as e:
            LOGGER.debug(f"Engine error for {symbol}: {e}")
        
        # Fallback: Technical analysis based prediction
        return self._technical_prediction(symbol, asset_type, current_price)
    
    def _compute_rsi(self, closes: list, period: int = 14) -> float:
        """Compute RSI (Relative Strength Index). Returns 0-100."""
        if len(closes) < period + 1:
            return 50.0  # neutral
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = deltas[-(period):]
        gains = [d for d in recent if d > 0]
        losses_vals = [-d for d in recent if d < 0]
        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses_vals) / period if losses_vals else 0.0
        if avg_gain == 0 and avg_loss == 0:
            return 50.0  # No movement = neutral
        if avg_loss == 0:
            return 100.0  # All gains, no losses
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _technical_prediction(self, symbol: str, asset_type: str, current_price: float) -> Dict:
        """
        Make a technical analysis based prediction.

        Uses multi-signal approach:
          1. 5/10-day momentum (short-term trend)       — weight 0.35
          2. 10/20-day momentum (medium-term trend)      — weight 0.25
          3. RSI (overbought/oversold)                   — weight 0.25
          4. Price position vs 20-day range (mean-rev)   — weight 0.15
          5. Dynamic mover detection (intraday momentum)

        Direction is signal-driven, NOT hardcoded.
        Ambiguous signals → HOLD (blocked by paper_tracker).
        Confidence reflects signal agreement, capped at 0.92.
        Brain v3 further adjusts after this via analyze_batch().
        """
        # CRITICAL: Dynamic movers (up 5%+ today) — proven momentum, ride the wave
        bullish_movers = getattr(self, '_bullish_movers', set())
        if symbol in bullish_movers:
            LOGGER.info(f"🚀 [SCOUT] {symbol} is a dynamic mover — forcing BUY direction!")
            target = current_price * 1.05
            return {
                "direction": "BUY",
                "entry_price": current_price,
                "target_price": target,
                "confidence": 0.72,
            }

        # Collect price history for multi-signal analysis
        closes = self._fetch_price_history(symbol, asset_type)

        if len(closes) >= 20:
            # ── Signal 1: Short-term momentum (5-day vs 10-day SMA) ──
            sma5 = sum(closes[-5:]) / 5
            sma10 = sum(closes[-10:]) / 10
            short_momentum = (sma5 - sma10) / sma10

            # ── Signal 2: Medium-term momentum (10-day vs 20-day SMA) ──
            sma20 = sum(closes[-20:]) / 20
            med_momentum = (sma10 - sma20) / sma20

            # ── Signal 3: RSI — overbought/oversold indicator ──
            rsi = self._compute_rsi(closes)
            # RSI > 70 = overbought (bearish), RSI < 30 = oversold (bullish)
            # Normalize to -1..+1 range: 50 = neutral, 30 = +0.5, 70 = -0.5
            rsi_signal = (50.0 - rsi) / 40.0  # Positive = bullish, negative = bearish
            rsi_signal = max(-1.0, min(1.0, rsi_signal))

            # ── Signal 4: Mean reversion (current price vs 20-day range) ──
            high_20 = max(closes[-20:])
            low_20 = min(closes[-20:])
            range_20 = high_20 - low_20 if high_20 != low_20 else 1.0
            range_position = (current_price - low_20) / range_20  # 0=bottom, 1=top

            # ── Signal agreement check ──
            # Count how many signals agree on direction
            signals = []
            if short_momentum > 0.005:
                signals.append(1)   # bullish
            elif short_momentum < -0.005:
                signals.append(-1)  # bearish
            else:
                signals.append(0)   # neutral

            if med_momentum > 0.005:
                signals.append(1)
            elif med_momentum < -0.005:
                signals.append(-1)
            else:
                signals.append(0)

            if rsi < 40:
                signals.append(1)   # oversold → bullish
            elif rsi > 60:
                signals.append(-1)  # overbought → bearish
            else:
                signals.append(0)

            signal_sum = sum(signals)
            agreeing = sum(1 for s in signals if s != 0 and s == (1 if signal_sum > 0 else -1))

            # ── Combine signals into direction + confidence ──
            score = 0.0
            score += short_momentum * 10.0 * 0.35   # weight 0.35
            score += med_momentum * 8.0 * 0.25      # weight 0.25
            score += rsi_signal * 0.25               # weight 0.25
            score += (0.5 - range_position) * 0.15   # weight 0.15 (reduced)

            # Direction: require conviction, else HOLD
            if score > 0.02:
                direction = "BUY"
            elif score < -0.02:
                direction = "SELL"
            else:
                # Ambiguous signal — HOLD (paper_tracker will skip this)
                direction = "HOLD"

            # Confidence: base + signal strength + agreement bonus
            raw_conf = 0.50 + abs(score)
            # Bonus for signal agreement (up to +0.10)
            agreement_bonus = agreeing * 0.033
            raw_conf += agreement_bonus
            confidence = max(0.40, min(0.92, raw_conf))

        elif len(closes) >= 10:
            # Fallback: basic 5/10 momentum + RSI (less data available)
            sma5 = sum(closes[-5:]) / 5
            sma10 = sum(closes[-10:]) / 10
            momentum = (sma5 - sma10) / sma10
            rsi = self._compute_rsi(closes)

            threshold = 0.02 if asset_type == "stock" else 0.03
            if momentum > threshold and rsi < 65:
                direction = "BUY"
                confidence = min(0.80, 0.55 + abs(momentum) + (0.05 if rsi < 40 else 0))
            elif momentum < -threshold and rsi > 35:
                direction = "SELL"
                confidence = min(0.80, 0.55 + abs(momentum) + (0.05 if rsi > 60 else 0))
            else:
                direction = "HOLD"  # Conflicting or weak signals → skip
                confidence = 0.40

        else:
            # No meaningful price data — HOLD, don't guess
            direction = "HOLD"
            confidence = 0.35

        # Calculate target price
        if direction == "BUY":
            target = current_price * (1 + (confidence * 0.08))
        else:
            target = current_price * (1 - (confidence * 0.08))

        return {
            "direction": direction,
            "entry_price": current_price,
            "target_price": target,
            "confidence": confidence,
        }

    def _fetch_price_history(self, symbol: str, asset_type: str) -> list:
        """
        Fetch 20-35 days of close prices for technical analysis.

        Returns: list of float close prices (oldest first), empty on failure.
        """
        try:
            if asset_type == "stock":
                polygon_key = os.getenv("POLYGON_API_KEY")
                if not polygon_key:
                    return []
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=35)).strftime("%Y-%m-%d")
                url = (
                    f"https://api.polygon.io/v2/aggs/ticker/{symbol}"
                    f"/range/1/day/{start}/{end}?apiKey={polygon_key}"
                )
                resp = self._session.get(url, timeout=10)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    return [r["c"] for r in results if r.get("c")]
            else:
                cg_id = get_coingecko_id(symbol) or symbol.lower()
                # FIX: days=30 without interval returns HOURLY data (~720 points)
                # Adding interval=daily returns actual daily closes (~30 points)
                # Previous bug: "5-day SMA" was actually 5-HOUR SMA
                url = (
                    f"https://api.coingecko.com/api/v3/coins/{cg_id}"
                    f"/market_chart?vs_currency=usd&days=30&interval=daily"
                )
                resp = self._session.get(url, timeout=5)
                if resp.status_code == 200:
                    return [p[1] for p in resp.json().get("prices", [])]
        except Exception as e:
            LOGGER.debug(f"Price history fetch for {symbol}: {e}")
        return []


class GameResolver:
    """
    🏆 Resolves trades and counts the MONEY
    
    After 24-48 hours, we check:
    - What was the prediction?
    - What actually happened?
    - Did they MAKE MONEY or LOSE MONEY?
    
    Winners rise. Losers fall.
    That's the game.
    """
    
    def __init__(self):
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        LOGGER.info("🏆 [RESOLVER] Ready to count the money!")
    
    def _get_connection(self):
        from core.db_pool import get_sync_connection
        return get_sync_connection()
    
    def resolve_pending_trades(self, hours_old: int = 24) -> Dict:
        """
        Resolve all trades older than X hours.
        
        This is where we find out WHO MADE MONEY!
        """
        from core.money_game_engine import get_money_game
        
        if not self.DATABASE_URL:
            return {"error": "No database"}
        
        game = get_money_game()
        results = {
            "resolved": 0,
            "winners": [],
            "losers": [],
            "total_profit": 0.0,
            "total_loss": 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            with self._get_connection() as conn:
                cur = conn.cursor()
                
                cutoff = datetime.utcnow() - timedelta(hours=hours_old)
                
                # Get unresolved trades
                cur.execute("""
                    SELECT id, symbol, asset_type, direction, entry_price, target_price
                    FROM money_game_trades
                    WHERE resolved_at IS NULL AND created_at < %s
                    ORDER BY created_at
                    LIMIT 100
                """, (cutoff,))
                
                trades = cur.fetchall()
        except Exception as e:
            LOGGER.error(f"🏆 [RESOLVER] DB error fetching trades: {e}")
            return {"error": str(e)}
        
        LOGGER.info(f"🏆 [RESOLVER] Found {len(trades)} trades to resolve...")
        
        # Reuse ONE scout for all price lookups (avoid 100x Yahoo API calls)
        scout = GhostScout(include_dynamic_movers=False)
        
        for trade in trades:
            trade_id, symbol, asset_type, direction, entry_price, target_price = trade
            
            # Get current price
            current_price = scout._get_current_price(symbol, asset_type)
            
            if current_price is None:
                LOGGER.warning(f"🏆 [RESOLVER] Could not get price for {symbol}")
                continue
            
            # Resolve the trade!
            result = game.resolve_trade(trade_id, current_price)
            
            if "error" not in result:
                results["resolved"] += 1
                profit = result.get("profit_pct", 0)
                
                if profit > 0:
                    results["winners"].append({
                        "symbol": symbol,
                        "profit": f"+{profit:.1f}%"
                    })
                    results["total_profit"] += profit
                else:
                    results["losers"].append({
                        "symbol": symbol,
                        "loss": f"{profit:.1f}%"
                    })
                    results["total_loss"] += abs(profit)
        
        # After resolving, update rankings!
        if results["resolved"] > 0:
            LOGGER.info("🏆 [RESOLVER] Updating rankings after resolution...")
            game.update_rankings()
        
        LOGGER.info(f"🏆 [RESOLVER] Resolved {results['resolved']} trades")
        LOGGER.info(f"   💰 Winners: {len(results['winners'])}, Total Profit: +{results['total_profit']:.1f}%")
        LOGGER.info(f"   💸 Losers: {len(results['losers'])}, Total Loss: -{results['total_loss']:.1f}%")
        
        return results


# Convenience functions
def run_scouting_cycle(fast: bool = True) -> Dict:
    """Run a full scouting cycle (fast=True uses parallel ThreadPoolExecutor)"""
    scout = GhostScout()
    if fast:
        return scout.scout_all_fast()
    return scout.scout_all()


def resolve_trades(hours: int = 24) -> Dict:
    """Resolve pending trades"""
    resolver = GameResolver()
    return resolver.resolve_pending_trades(hours)


def get_game_status() -> Dict:
    """Get current game status"""
    from core.money_game_engine import get_money_game
    return get_money_game().get_game_status()
