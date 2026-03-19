"""Routes: money_game — extracted from wolf_app.py (Step 12)"""

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
import httpx
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

try:
    from wolf_helpers import *
except ImportError:
    pass

router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- Routes: money_game (2 endpoints) ---

try:
    @router.get("/api/v3/competition/status")
    async def v3_competition_status():
        """
        🏆 V3: Get competition system status.
        
        Shows TOP 10 stocks and crypto, leaderboards, and pending contenders.
        NO BLACKLIST - everyone competes fairly!
        """
        try:
            from core.v3_competition import get_competition_system
            
            competition = get_competition_system()
            status = competition.get_competition_status()
            
            return {
                "ok": True,
                "philosophy": "No blacklist - everyone competes fairly. Only the best make TOP 10.",
                **status
            }
        
        except Exception as e:
            LOGGER.error(f"[V3-API] Competition status error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/v3/competition/leaderboard/{asset_type}")
    async def v3_competition_leaderboard(asset_type: str, limit: int = 20):
        """
        🏆 V3: Get leaderboard for stocks or crypto.
        
        Args:
            asset_type: "stock" or "crypto"
            limit: Max entries to return (default 20)
        """
        try:
            from core.v3_competition import get_competition_system
            
            if asset_type not in ["stock", "crypto"]:
                return {"ok": False, "error": "asset_type must be 'stock' or 'crypto'"}
            
            competition = get_competition_system()
            leaderboard = competition.get_leaderboard(asset_type, limit)
            
            return {
                "ok": True,
                "asset_type": asset_type,
                "leaderboard": leaderboard,
                "total": len(leaderboard)
            }
        
        except Exception as e:
            LOGGER.error(f"[V3-API] Leaderboard error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/v3/competition/contenders/{asset_type}")
    async def v3_competition_contenders(asset_type: str, limit: int = 10):
        """
        🏆 V3: Get pending assets closest to breaking into TOP 10.
        
        These are the assets "fighting" to get promoted!
        """
        try:
            from core.v3_competition import get_competition_system
            
            if asset_type not in ["stock", "crypto"]:
                return {"ok": False, "error": "asset_type must be 'stock' or 'crypto'"}
            
            competition = get_competition_system()
            contenders = competition.get_pending_contenders(asset_type, limit)
            
            return {
                "ok": True,
                "asset_type": asset_type,
                "message": "These assets are fighting to get into TOP 10!",
                "contenders": contenders,
                "gap_explanation": "gap_to_top_10 = how many ranks away from TOP 10"
            }
        
        except Exception as e:
            LOGGER.error(f"[V3-API] Contenders error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/v3/competition/update-rankings")
    async def v3_competition_update_rankings():
        """
        🏆 V3: Recalculate all rankings based on performance.
        
        This is the MAIN competition logic!
        - Promotes pending assets with better win rates
        - Demotes TOP 10 assets with declining performance
        
        Run daily via cron or manually.
        """
        try:
            from core.v3_competition import get_competition_system
            
            competition = get_competition_system()
            changes = competition.update_rankings()
            
            return {
                "ok": True,
                "message": "Rankings updated!",
                "changes": changes
            }
        
        except Exception as e:
            LOGGER.error(f"[V3-API] Update rankings error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/v3/competition/run-shadow-cycle")
    async def v3_competition_run_shadow():
        """
        🔮 V3: Run shadow predictions for ALL assets in the pool.
        
        Shadow predictions build competition data without sending alerts.
        This allows pending assets to "prove themselves".
        """
        try:
            from core.v3_shadow_predictor import run_shadow_predictions
            
            results = await run_shadow_predictions()
            
            return {
                "ok": True,
                "message": "Shadow prediction cycle complete",
                **results
            }
        
        except Exception as e:
            LOGGER.error(f"[V3-API] Shadow cycle error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/v3/competition/resolve-outcomes")
    async def v3_competition_resolve_outcomes():
        """
        🎯 V3: Resolve shadow predictions and update competitor scores.
        
        Checks if 48h window has passed for pending predictions
        and determines WIN/LOSS outcomes.
        """
        try:
            from core.v3_shadow_resolver import resolve_shadow_outcomes
            
            results = await resolve_shadow_outcomes()
            
            return {
                "ok": True,
                "message": "Outcomes resolved",
                **results
            }
        
        except Exception as e:
            LOGGER.error(f"[V3-API] Resolve outcomes error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/v3/competition/seed-pool")
    async def v3_competition_seed_pool():
        """
        🌱 V3: Seed initial competition pool with default assets.
        
        Run once to bootstrap the competition system.
        """
        try:
            from core.v3_competition import get_competition_system
            from core.v3_shadow_predictor import DEFAULT_STOCKS, DEFAULT_CRYPTO
            
            competition = get_competition_system()
            competition.seed_initial_pool(DEFAULT_STOCKS, DEFAULT_CRYPTO)
            
            return {
                "ok": True,
                "message": "Competition pool seeded!",
                "stocks_added": len(DEFAULT_STOCKS),
                "crypto_added": len(DEFAULT_CRYPTO)
            }
        
        except Exception as e:
            LOGGER.error(f"[V3-API] Seed pool error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/v3/competition/resolution-stats")
    async def v3_competition_resolution_stats():
        """
        📊 V3: Get shadow prediction resolution statistics.
        """
        try:
            from core.v3_shadow_resolver import get_shadow_resolver
            
            resolver = get_shadow_resolver()
            stats = resolver.get_resolution_stats()
            
            return {
                "ok": True,
                **stats
            }
        
        except Exception as e:
            LOGGER.error(f"[V3-API] Resolution stats error: {e}")
            return {"ok": False, "error": str(e)}

    LOGGER.info("✅ V3 Competition System endpoints registered (/api/v3/competition/*)")


except Exception as _sec_e:
    LOGGER.warning(f'Route section error: {_sec_e}')

try:
    @router.get("/api/money-game/status")
    async def money_game_status():
        """
        🎮 MONEY GAME: Full game status
        
        Shows the competition to find the best MONEY MAKERS.
        #1 = Most profitable, rankings based on actual PROFIT potential.
        """
        try:
            from core.money_game_engine import get_money_game
            
            game = get_money_game()
            status = game.get_game_status()
            
            return {
                "ok": True,
                "philosophy": "Money is the score! Find the next bullish money maker!",
                **status
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Status error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/money-game/leaderboard/{asset_type}")
    async def money_game_leaderboard(asset_type: str, limit: int = 20):
        """
        🎮 MONEY GAME: Leaderboard ranked by PROFIT potential
        
        #1 = Best money maker (most profit)
        Lower rank = Less profitable
        
        This is like a video game high score - MONEY = POINTS!
        """
        try:
            from core.money_game_engine import get_money_game
            
            if asset_type not in ["stock", "crypto"]:
                return {"ok": False, "error": "asset_type must be 'stock' or 'crypto'"}
            
            game = get_money_game()
            leaderboard = game.get_leaderboard(asset_type, limit)
            
            return {
                "ok": True,
                "asset_type": asset_type,
                "ranking_by": "money_score (profit potential)",
                "leaderboard": leaderboard,
                "total": len(leaderboard)
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Leaderboard error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/money-game/rising-stars/{asset_type}")
    async def money_game_rising_stars(asset_type: str, limit: int = 5):
        """
        🌟 MONEY GAME: Rising stars - The NEXT BIG DEAL!
        
        These are assets fighting to get into TOP 10.
        They're showing profit potential and could be promoted!
        """
        try:
            from core.money_game_engine import get_money_game
            
            if asset_type not in ["stock", "crypto"]:
                return {"ok": False, "error": "asset_type must be 'stock' or 'crypto'"}
            
            game = get_money_game()
            stars = game.get_rising_stars(asset_type, limit)
            
            return {
                "ok": True,
                "asset_type": asset_type,
                "message": "These are the NEXT BIG DEALS - fighting for TOP 10!",
                "rising_stars": stars
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Rising stars error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/money-game/player/{symbol}")
    async def money_game_player_stats(symbol: str):
        """
        🎮 MONEY GAME: Get detailed stats for a specific player (asset)
        
        Shows their full profile: profit history, rank, tier, momentum.
        """
        try:
            from core.money_game_engine import get_money_game
            
            game = get_money_game()
            stats = game.get_player_stats(symbol.upper())
            
            if not stats:
                return {"ok": False, "error": f"Player {symbol} not found"}
            
            return {
                "ok": True,
                **stats
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Player stats error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/money-game/daily-movers")
    async def money_game_daily_movers(min_gain: float = 5.0):
        """
        🚀 MONEY GAME: Today's biggest gainers!
        
        Dynamic mover detection - catches stocks Ghost might miss.
        Example: Nextpower +16%, Seagate +15% etc.
        
        Args:
            min_gain: Minimum % gain to include (default 5%)
        """
        try:
            from core.ghost_scout import fetch_daily_movers
            
            movers = fetch_daily_movers(min_gain_pct=min_gain)
            
            return {
                "ok": True,
                "message": f"Found {len(movers)} stocks up {min_gain}%+ today!",
                "movers": movers,
                "tip": "These dynamic movers are now being added to Ghost's watchlist automatically!"
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Daily movers error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/scout-all")
    async def money_game_scout_all():
        """
        🔍 MONEY GAME: Run the scout - find ALL money makers!
        
        The scout evaluates EVERY asset and records predictions.
        This builds the data to find who actually makes money.
        
        NEW: Now includes dynamic movers (10%+ daily gainers)!
        NEW: News sentiment integration for ✅ indicator!
        
        Run daily to continuously evaluate all assets.
        """
        try:
            from core.ghost_scout import run_scouting_cycle
            
            results = run_scouting_cycle()
            
            return {
                "ok": True,
                "message": "Scout completed! All assets evaluated.",
                **results
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Scout error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/resolve-trades")
    async def money_game_resolve_trades(hours: int = 24):
        """
        🏆 MONEY GAME: Resolve trades and count the MONEY!
        
        After predictions are made, we wait 24-48h then check:
        - Did they MAKE money?
        - Did they LOSE money?
        
        Winners rise in rankings, losers fall.
        
        Args:
            hours: Resolve trades older than X hours (default 24)
        """
        try:
            from core.ghost_scout import resolve_trades
            
            results = resolve_trades(hours)
            
            return {
                "ok": True,
                "message": "Trades resolved! Money counted.",
                **results
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Resolve error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/update-rankings")
    async def money_game_update_rankings():
        """
        🏆 MONEY GAME: Recalculate rankings based on PROFIT
        
        This determines:
        - Who's #1 (best money maker)
        - Who gets promoted to TOP 10
        - Who gets demoted (not making money)
        
        Run after resolving trades.
        """
        try:
            from core.money_game_engine import get_money_game
            
            game = get_money_game()
            changes = game.update_rankings()
            
            return {
                "ok": True,
                "message": "Rankings recalculated by PROFIT!",
                "changes": changes
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Update rankings error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/seed-players")
    async def money_game_seed_players():
        """
        🌱 MONEY GAME: Seed all players into the game
        
        Adds all stocks and crypto as competitors.
        Run once to initialize the game.
        """
        try:
            from core.money_game_engine import get_money_game
            from core.ghost_scout import ALL_STOCKS, ALL_CRYPTO
            
            game = get_money_game()
            
            # Add all stocks
            for symbol in ALL_STOCKS:
                game.add_player(symbol, "stock")
            
            # Add all crypto
            for symbol in ALL_CRYPTO:
                game.add_player(symbol, "crypto")
            
            return {
                "ok": True,
                "message": "All players added to the game!",
                "stocks": len(ALL_STOCKS),
                "crypto": len(ALL_CRYPTO),
                "total_players": len(ALL_STOCKS) + len(ALL_CRYPTO)
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Seed error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/seed-top10")
    async def money_game_seed_top10():
        """
        🎮 SEED TOP 10: Initialize with known performers
        
        Seeds the Money Game with manual TOP 10 so we can:
        1. See immediate leaderboard results
        2. Watch Ghost naturally promote/demote based on real performance
        3. Confirm the system works when Ghost promotes a NEW symbol
        """
        try:
            import os
            from datetime import datetime
            
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                return {"ok": False, "error": "No DATABASE_URL"}
            
            # Manual TOP 10 seeds - Best performers
            STOCK_SEEDS = [
                ("NVDA", 15.0, 5, 0.80),   # AI king, GPU demand
                ("META", 12.0, 5, 0.80),   # AI + ads recovery
                ("PLTR", 10.0, 4, 0.75),   # AI/defense play
                ("COIN", 8.0, 4, 0.75),    # Crypto proxy
                ("GOOGL", 7.0, 4, 0.75),   # AI catch-up + search
                ("AMZN", 6.0, 4, 0.75),    # AWS + retail
                ("TSLA", 5.0, 4, 0.60),    # Volatile but predictable
                ("AMD", 4.0, 4, 0.70),     # NVDA alternative
                ("MSTR", 8.0, 4, 0.70),    # Bitcoin proxy (150k+ BTC)
                ("HOOD", 6.0, 4, 0.65),    # Retail + crypto exposure
            ]
            
            CRYPTO_SEEDS = [
                ("BTC", 12.0, 5, 0.80),    # King, institutional
                ("ETH", 10.0, 5, 0.80),    # Smart contracts
                ("SOL", 15.0, 4, 0.75),    # Fast L1, meme activity
                ("RNDR", 20.0, 5, 0.81),   # 81% win rate! AI/GPU
                ("TURBO", 18.0, 4, 0.79),  # 79% win rate! Meme momentum
                ("XRP", 8.0, 4, 0.75),     # Payments, legal clarity
                ("LINK", 7.0, 4, 0.70),    # Oracle, DeFi essential
                ("AVAX", 8.0, 4, 0.70),    # L1, gaming/DeFi
                ("SUI", 12.0, 4, 0.65),    # New L1, high volatility
                ("INJ", 10.0, 4, 0.70),    # DeFi focused
            ]
            
            from core.db_pool import get_sync_connection
            with get_sync_connection() as conn:
                cur = conn.cursor()
                
                seeded_stocks = []
                seeded_crypto = []
                
                for seeds, asset_type, result_list in [
                    (STOCK_SEEDS, "stock", seeded_stocks),
                    (CRYPTO_SEEDS, "crypto", seeded_crypto)
                ]:
                    for rank, (symbol, profit, trades, win_rate) in enumerate(seeds, 1):
                        wins = int(trades * win_rate)
                        losses = trades - wins
                        avg_profit = profit / trades
                        money_score = profit * (1 + win_rate)
                        
                        cur.execute("""
                            INSERT INTO money_game_players 
                            (symbol, asset_type, tier, total_profit_pct, avg_profit_per_trade,
                             best_trade_pct, worst_trade_pct, total_trades, wins, losses,
                             win_rate, money_score, recent_profit_pct, momentum, rank, rank_change,
                             last_trade, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (symbol) DO UPDATE SET
                                tier = EXCLUDED.tier,
                                total_profit_pct = EXCLUDED.total_profit_pct,
                                avg_profit_per_trade = EXCLUDED.avg_profit_per_trade,
                                total_trades = EXCLUDED.total_trades,
                                wins = EXCLUDED.wins,
                                losses = EXCLUDED.losses,
                                win_rate = EXCLUDED.win_rate,
                                money_score = EXCLUDED.money_score,
                                rank = EXCLUDED.rank,
                                last_updated = NOW()
                        """, (
                            symbol, asset_type, "elite", profit, avg_profit,
                            profit * 0.4, -profit * 0.1, trades, wins, losses,
                            win_rate, money_score, profit * 0.3, "stable", rank, 0,
                            datetime.utcnow(), datetime.utcnow()
                        ))
                        result_list.append({"rank": rank, "symbol": symbol, "profit": f"+{profit:.1f}%"})
            
            # Reload the game to pick up new data
            from core.money_game_engine import get_money_game
            game = get_money_game()
            game._load_players()
            game._rebuild_elite_lists()
            
            return {
                "ok": True,
                "message": "🎮 TOP 10 SEEDED! Watch for Ghost to promote new symbols!",
                "stocks": seeded_stocks,
                "crypto": seeded_crypto,
                "next_steps": [
                    "Ghost scouts new predictions daily",
                    "After 24h, trades resolve and real profit counted",
                    "If a NEW symbol beats the seeds, it gets PROMOTED!",
                    "Watch for: A symbol you didn't seed making TOP 10"
                ]
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Seed TOP 10 error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/money-game/elite")
    async def money_game_get_elite():
        """
        👑 MONEY GAME: Get the ELITE (TOP 10 of each)
        
        These are the proven MONEY MAKERS.
        Ghost sends predictions only for these elite assets.
        """
        try:
            from core.money_game_engine import get_money_game
            
            game = get_money_game()
            
            return {
                "ok": True,
                "message": "These are the PROVEN money makers!",
                "elite_stocks": game.get_elite_stocks(),
                "elite_crypto": game.get_elite_crypto()
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Elite error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/smart-scout")
    async def money_game_smart_scout():
        """
        🔍 MONEY GAME: Smart scout with rate limiting
        
        Uses batch price fetching and respects API rate limits.
        Better for scouting all 211 assets reliably.
        """
        try:
            from core.smart_scout import smart_scout_all
            
            results = smart_scout_all()
            
            return {
                "ok": True,
                "message": "Smart scout complete!",
                **results
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Smart scout error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/daily-cycle")
    async def money_game_daily_cycle():
        """
        ⏰ MONEY GAME: Run full daily cycle
        
        This runs:
        1. Scout all assets
        2. Resolve 24h old trades
        3. Update rankings
        4. Return elite for alerts
        
        Perfect for daily cron job.
        """
        try:
            from core.smart_scout import run_daily_cycle
            
            results = run_daily_cycle()
            
            return {
                "ok": True,
                "message": "Daily cycle complete!",
                **results
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Daily cycle error: {e}")
            return {"ok": False, "error": str(e)}

    @router.get("/api/money-game/elite-predictions")
    async def money_game_elite_predictions():
        """
        🎯 MONEY GAME: Get elite predictions for Telegram
        
        Returns the TOP 10 stocks and crypto with full details
        for the 8 AM Telegram alert.
        """
        try:
            from core.smart_scout import get_elite_predictions
            
            results = get_elite_predictions()
            
            return {
                "ok": True,
                **results
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Elite predictions error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/telegram-alert")
    async def money_game_telegram_alert(request: Request):
        """
        📱 MONEY GAME: Send TOP 10 money makers to Telegram
        
        This sends the PROVEN money makers (from Money Game rankings)
        instead of just the old whitelist system.
        
        Requires X-Cron-Secret header for security.
        """
        # Check cron secret
        cron_secret = os.getenv("CRON_SECRET", "ghost-cron-2024")
        provided_secret = request.headers.get("X-Cron-Secret", "")
        
        if not cron_secret or provided_secret != cron_secret:
            return {"ok": False, "error": "Unauthorized - invalid X-Cron-Secret"}
        
        try:
            from core.money_game_engine import get_money_game
            from core.smart_scout import SmartScout
            
            game = get_money_game()
            scout = SmartScout()
            
            # Get elite (TOP 10 money makers)
            elite_stocks = game.get_elite_stocks()[:10]
            elite_crypto = game.get_elite_crypto()[:10]
            
            # If no elite yet (still building data), use backup
            if not elite_stocks and not elite_crypto:
                LOGGER.warning("[MONEY-GAME] No elite yet - using default alert system")
                return {"ok": False, "error": "No elite established yet - Money Game still building data"}
            
            # Get prices for elite
            stock_prices = scout.get_stock_prices_batch(elite_stocks) if elite_stocks else {}
            crypto_prices = scout.get_crypto_prices_batch(elite_crypto) if elite_crypto else {}
            
            # Build predictions list
            stock_picks = []
            for symbol in elite_stocks:
                stats = game.get_player_stats(symbol)
                price = stock_prices.get(symbol, 0)
                if stats and price > 0:
                    stock_picks.append({
                        "symbol": symbol,
                        "current": price,
                        "prediction_48h": price * 1.03,  # 3% target
                        "buy_in": price * 0.99,
                        "sell": price * 1.02,
                        "confidence": min(0.85, 0.70 + (stats.get("money_score", 0) / 100)),
                        "direction": "UP",
                        "asset_type": "stock",
                        "money_score": stats.get("money_score", 0),
                        "rank": stats.get("rank", 999)
                    })
            
            crypto_picks = []
            for symbol in elite_crypto:
                stats = game.get_player_stats(symbol)
                price = crypto_prices.get(symbol, 0)
                if stats and price > 0:
                    crypto_picks.append({
                        "symbol": symbol,
                        "current": price,
                        "prediction_48h": price * 1.05,  # 5% target
                        "buy_in": price * 0.99,
                        "sell": price * 1.02,
                        "confidence": min(0.85, 0.70 + (stats.get("money_score", 0) / 100)),
                        "direction": "UP",
                        "asset_type": "crypto",
                        "money_score": stats.get("money_score", 0),
                        "rank": stats.get("rank", 999)
                    })
            
            # Sort by money_score
            stock_picks.sort(key=lambda x: x.get("money_score", 0), reverse=True)
            crypto_picks.sort(key=lambda x: x.get("money_score", 0), reverse=True)
            
            # Build message
            msg_lines = [
                "🎮 *GHOST MONEY GAME - TOP 10*",
                "_Proven money makers ranked by profit_",
                "",
                "📈 *STOCKS (Elite Money Makers)*"
            ]
            
            for i, p in enumerate(stock_picks[:10], 1):
                msg_lines.append(f"  #{i} {p['symbol']}: ${p['current']:.2f} (Score: {p['money_score']:.1f})")
            
            if not stock_picks:
                msg_lines.append("  _Building rankings..._")
            
            msg_lines.append("")
            msg_lines.append("🪙 *CRYPTO (Elite Money Makers)*")
            
            for i, p in enumerate(crypto_picks[:10], 1):
                price_str = f"${p['current']:.4f}" if p['current'] < 1 else f"${p['current']:.2f}"
                msg_lines.append(f"  #{i} {p['symbol']}: {price_str} (Score: {p['money_score']:.1f})")
            
            if not crypto_picks:
                msg_lines.append("  _Building rankings..._")
            
            msg_lines.append("")
            msg_lines.append("💡 _Rankings based on actual profit history_")
            msg_lines.append("🎯 _Higher score = better money maker_")
            
            message = "\n".join(msg_lines)
            
            # Send to Telegram
            success = _tg_send_chat_message(TELEGRAM_CHAT_ID, message)
            
            return {
                "ok": success,
                "message": "Money Game TOP 10 sent!" if success else "Failed to send",
                "stocks_sent": len(stock_picks),
                "crypto_sent": len(crypto_picks)
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Telegram alert error: {e}")
            return {"ok": False, "error": str(e)}

    @router.post("/api/money-game/trigger-now")
    async def money_game_trigger_now():
        """
        🚀 INSTANT TRIGGER: Run scout + send alert NOW
        
        For testing - bypasses schedule and runs immediately:
        1. Run full scout
        2. Send Telegram alert with results
        
        No auth required (public endpoint for quick testing).
        """
        try:
            from core.smart_scout import SmartScout, get_elite_predictions
            
            results = {"steps": []}
            
            # Step 1: Run scout
            LOGGER.info("🚀 [TRIGGER-NOW] Running instant scout...")
            scout = SmartScout()
            scout_result = scout.full_scout()
            
            # Extract counts from nested structure
            stocks_scouted = scout_result.get("stocks", {}).get("scouted", 0) or scout_result.get("total_scouted", 0) // 2
            crypto_scouted = scout_result.get("crypto", {}).get("scouted", 0) or scout_result.get("total_scouted", 0) // 2
            total_scouted = scout_result.get("total_scouted", stocks_scouted + crypto_scouted)
            
            results["scout"] = {
                "stocks_scouted": stocks_scouted,
                "crypto_scouted": crypto_scouted,
                "total_scouted": total_scouted,
                "elapsed_seconds": scout_result.get("elapsed_seconds", 0)
            }
            results["steps"].append(f"Scout complete: {total_scouted} assets")
            
            # Step 2: Get elite
            elite = get_elite_predictions()
            stocks = elite.get("elite_stocks", [])[:5]
            crypto = elite.get("elite_crypto", [])[:5]
            results["elite"] = {"stocks": stocks, "crypto": crypto}
            results["steps"].append("Elite fetched")
            
            # Step 3: Send Telegram (HTML format)
            if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                msg = "🚀 <b>GHOST INSTANT SCAN</b>\n\n"
                msg += f"📊 Scanned: {stocks_scouted} stocks, {crypto_scouted} crypto\n\n"
                
                if stocks:
                    msg += "📈 <b>Top Stocks:</b>\n"
                    for i, s in enumerate(stocks, 1):
                        msg += f"  {i}. {s}\n"
                
                if crypto:
                    msg += "\n🪙 <b>Top Crypto:</b>\n"
                    for i, c in enumerate(crypto, 1):
                        msg += f"  {i}. {c}\n"
                
                if not stocks and not crypto:
                    msg += "<i>Building rankings... (run again after more data)</i>\n"
                
                msg += "\n<i>Instant scan triggered</i>"
                
                success = _tg_send_chat_message(TELEGRAM_CHAT_ID, msg)
                results["telegram_sent"] = success
                results["steps"].append(f"Telegram: {'sent' if success else 'failed'}")
            else:
                results["telegram_sent"] = False
                results["steps"].append("Telegram not configured")
            
            return {
                "ok": True,
                "message": "Instant trigger complete!",
                **results
            }
        
        except Exception as e:
            LOGGER.error(f"[MONEY-GAME] Instant trigger error: {e}")
            return {"ok": False, "error": str(e)}

    LOGGER.info("✅ 🎮 Money Game Engine endpoints registered (/api/money-game/*)")
except Exception as _route_e:
    LOGGER.warning(f'Route section load error: {_route_e}')
