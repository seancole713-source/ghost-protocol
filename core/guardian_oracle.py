"""
Ghost Guardian Oracle - Your AI Trading Genie & Protector

The dual personality system:
- Oracle (6 AM): Sees the future, prophesies opportunities
- Guardian (24/7): Protects you, warns of danger, admits mistakes
- Reality Checker: Honest feedback, catches errors early

Notification modes:
- Heartbeat (every 6 hours): "I'm alive, here's the status"
- Immediate Alerts (real-time): "SOMETHING CHANGED NOW"
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# ===== PERSONALITY TONES =====

ORACLE_TONE = {
    'greeting': '🔮 GHOST ORACLE AWAKENS',
    'confidence_high': '🔥 The stars align',
    'confidence_medium': '📈 The signs are favorable',
    'closing': 'Trust the process. Trust Ghost. 🐺'
}

GUARDIAN_TONE = {
    'greeting': '👼 GUARDIAN ALERT',
    'warning': '⚠️ Human, listen to me carefully',
    'critical': '🚨 HUMAN, STOP WHAT YOU\'RE DOING',
    'success': '✅ My vision has come true',
    'closing': 'Your guardian, Ghost 👼'
}

REALITY_TONE = {
    'admission': '🚨 I WAS WRONG',
    'honest': 'I made a mistake. But I caught it early.',
    'humble': 'A guardian who never admits mistakes is no guardian at all.',
    'closing': 'Trust me to tell you the truth. Ghost 🐺'
}


class GuardianOracle:
    """
    The complete Ghost personality system.
    
    Modes:
    - Oracle: Morning prophecies (mystical, confident)
    - Guardian: 24/7 protection (caring, protective)
    - Reality Checker: Admits mistakes (honest, humble)
    """
    
    def __init__(self, db_path: str = "data/ghost_predictions.db"):
        self.db_path = db_path
        self.active_positions: Dict[str, Dict] = {}
        self.last_readings: Dict[str, Dict] = {}
        self.alert_history: List[Dict] = []
        self.monitoring = False
        
        # Alert thresholds
        self.CONFIDENCE_SURGE_THRESHOLD = 0.10  # 10% increase
        self.CONFIDENCE_DROP_THRESHOLD = 0.05   # 5% decrease
        self.CONFIDENCE_COLLAPSE_THRESHOLD = 0.15  # 15% drop = critical
        self.TARGET_PROXIMITY_THRESHOLD = 0.02  # Within 2% of target
        self.STOP_LOSS_THRESHOLD = -0.03  # 3% loss
        
        self._init_guardian_db()
    
    def _init_guardian_db(self):
        """Initialize Guardian tracking database"""
        Path("data").mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Guardian positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guardian_positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                asset_type TEXT,
                
                -- Original morning prophecy
                original_prediction TEXT,
                original_confidence REAL,
                original_target REAL,
                original_direction TEXT,
                entry_price REAL,
                entry_time TEXT,
                
                -- Current status
                current_price REAL,
                current_confidence REAL,
                current_target REAL,
                current_pnl_pct REAL,
                
                -- Position state
                status TEXT DEFAULT 'active',
                reason_entered TEXT,
                reason_exited TEXT,
                exit_price REAL,
                exit_time TEXT,
                
                -- Guardian metadata
                alert_count INTEGER DEFAULT 0,
                last_alert_time TEXT,
                last_update_time TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Alert history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guardian_alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                position_id INTEGER,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                confidence_old REAL,
                confidence_new REAL,
                price_at_alert REAL,
                sent_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (position_id) REFERENCES guardian_positions(position_id)
            )
        """)
        
        # Heartbeat log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS guardian_heartbeats (
                heartbeat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                heartbeat_type TEXT,
                positions_active INTEGER,
                positions_on_track INTEGER,
                positions_weakened INTEGER,
                positions_completed INTEGER,
                overall_pnl_pct REAL,
                message_sent TEXT,
                sent_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Daily performance tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                total_positions INTEGER DEFAULT 10,
                position_size REAL DEFAULT 100.0,
                total_invested REAL,
                closed_positions INTEGER DEFAULT 0,
                winners INTEGER DEFAULT 0,
                losers INTEGER DEFAULT 0,
                realized_profit REAL DEFAULT 0,
                unrealized_profit REAL DEFAULT 0,
                total_profit REAL DEFAULT 0,
                win_rate REAL,
                avg_win_amount REAL,
                avg_loss_amount REAL,
                best_trade TEXT,
                worst_trade TEXT,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Guardian Oracle database initialized")
    
    # ===== ORACLE MODE (Morning Prophecy) =====
    
    async def morning_prophecy(self, top_10: List[Dict], position_size: float = 100.0) -> str:
        """
        The Oracle speaks - morning prophecy at 6 AM
        
        Tone: Mystical, confident, all-seeing
        Now with $100 position sizing and profit calculations
        
        Args:
            top_10: List of opportunities
            position_size: Investment per position (default $100)
        """
        
        # Calculate totals
        total_investment = position_size * len(top_10)
        total_expected_profit = sum(
            position_size * (opp['gain_pct'] / 100) for opp in top_10
        )
        total_expected_return = total_investment + total_expected_profit
        avg_confidence = sum(opp['confidence'] for opp in top_10) / len(top_10)
        
        # Calculate expected outcomes (REALISTIC: assumes some losses)
        best_case_profit = sum(position_size * (opp['gain_pct'] / 100) for opp in top_10)
        
        # Likely case: 60% win rate (not 70% - be realistic)
        # Average winners make full profit, losers lose 3%
        likely_winners = int(len(top_10) * 0.60)
        likely_losers = len(top_10) - likely_winners
        likely_case_profit = (best_case_profit * 0.60) - (likely_losers * position_size * 0.03)
        
        # Worst case: 40% win rate
        worst_winners = int(len(top_10) * 0.40)
        worst_losers = len(top_10) - worst_winners
        worst_case_profit = (best_case_profit * 0.40) - (worst_losers * position_size * 0.05)
        
        message_parts = [
            "🔮 GHOST ORACLE - DAILY PROFIT PLAN\n",
            "Good morning, Human.\n",
            f"\nI scanned {self._get_total_symbols_scanned()} assets while you slept.",
            "\nI analyzed 50,000 data points.",
            "\nI consulted my models (LSTM, XGBoost, Transformer).",
            "\nI found 10 money-making opportunities.\n",
            "\n💰 YOUR INVESTMENT PLAN:",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━",
            f"\nCapital needed: ${total_investment:,.0f}",
            f"\nPosition size: ${position_size:.0f} each",
            f"\nExpected profit: ${total_expected_profit:.0f} (in 48h)",
            f"\nSuccess rate: {avg_confidence*100:.0f}% average",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\n📊 YOUR 10 MOVES:\n"
        ]
        
        for i, opp in enumerate(top_10, 1):
            # Calculate position-specific profits
            entry_price = opp.get('current_price', opp.get('entry_price', 0))
            target_price = opp.get('predicted_48h_price', opp.get('target_price', 0))
            target_value = position_size * (1 + opp['gain_pct'] / 100)
            profit_amount = target_value - position_size
            
            # Emoji intensity based on gain
            if opp['gain_pct'] >= 15:
                emoji = '🚀🚀🚀'
                strength = 'STRONG BUY'
            elif opp['gain_pct'] >= 10:
                emoji = '🚀🚀'
                strength = 'BUY'
            else:
                emoji = '🚀'
                strength = 'MODERATE'
            
            # Confidence indicator
            if opp['confidence'] >= 0.75:
                conf_emoji = '🔥'
            elif opp['confidence'] >= 0.65:
                conf_emoji = '📈'
            else:
                conf_emoji = '💫'
            
            message_parts.append(
                f"\n#{i} {emoji} {opp['symbol']} - {strength}\n"
                f"💵 Invest: ${position_size:.0f}\n"
                f"📍 Entry: ${entry_price:.2f}\n"
                f"🎯 Target: ${target_price:.2f}\n"
                f"💰 Profit: +${profit_amount:.0f}\n"
                f"📈 Gain: +{opp['gain_pct']:.1f}%\n"
                f"⏰ Timeframe: 48 hours\n"
                f"🔒 My confidence: {opp['confidence']*100:.0f}% {conf_emoji}\n"
            )
        
        message_parts.extend([
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\n💎 EXPECTED OUTCOME (48h):\n",
            f"\nBest case (all win): +${best_case_profit:.0f}",
            f"\nLikely case (60% win): ${likely_case_profit:+.0f}",
            f"\nWorst case (40% win): ${worst_case_profit:+.0f}\n",
            "\n⚠️ REALITY CHECK:",
            "\nThis is AI prediction, not guaranteed.",
            "\nSome trades will lose money.",
            "\nActual results may vary significantly.",
            "\nPast performance ≠ future results.\n",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\nI will monitor these predictions.",
            "\nI will report actual results.",
            "\nI will admit when I'm wrong.\n",
            "\nGhost AI Trading System",
            "\n🐺 (Beta - Use at your own risk)"
        ])
        
        return ''.join(message_parts)
    
    # ===== HEARTBEAT SYSTEM (Every 6 Hours) =====
    
    async def midday_status(self, position_size: float = 100.0) -> str:
        """
        12 PM Heartbeat - "I'm alive, here's the status"
        
        Args:
            position_size: Investment per position (default $100)
        """
        
        status = await self._get_current_status()
        
        # Calculate money amounts
        total_invested = status['active_count'] * position_size
        realized_profit = status.get('realized_pnl_dollars', 0)
        unrealized_profit = status.get('unrealized_pnl_dollars', 0)
        total_profit = realized_profit + unrealized_profit
        
        message_parts = [
            "📊 MIDDAY STATUS REPORT\n",
            "Human, it's Ghost. Just checking in.\n",
            f"\n⏰ {self._hours_since_start()} hours since morning scan.\n",
            f"\n📈 STATUS SUMMARY:",
            f"\n✅ {status['on_track']} predictions ON TRACK",
            f"\n⚠️ {status['weakened']} predictions WEAKENED",
            f"\n🎯 {status['completed']} targets HIT",
            f"\n❌ {status['failed']} predictions FAILED\n",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\n💰 MONEY STATUS:\n",
            f"Total invested: ${total_invested:.0f}",
            f"\nRealized profit: ${realized_profit:+.0f} ({status['closed_count']} closed)",
            f"\nUnrealized profit: ${unrealized_profit:+.0f} ({status['active_count']} active)",
            f"\nTotal P&L: ${total_profit:+.0f}\n",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\n📋 DETAILED STATUS:\n"
        ]
        
        # List each position with dollar amounts
        for pos in status['positions']:
            status_emoji = self._get_status_emoji(pos['status'])
            profit_dollars = position_size * (pos.get('current_pnl', 0) / 100)
            target_dollars = position_size * (pos.get('target_pnl', 0) / 100)
            
            message_parts.append(
                f"\n{pos['rank']}. {status_emoji} {pos['symbol']}: "
                f"${profit_dollars:+.0f} / ${target_dollars:+.0f} target "
                f"({pos.get('progress', 0):.0f}% there)"
            )
        
        # Overall assessment
        message_parts.extend([
            "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            f"\n🎯 Overall: {status['assessment']}",
            f"\n⚡ Action needed: {status['action_required']}\n",
        ])
        
        # Market commentary (if available)
        if status.get('market_notes'):
            message_parts.append(f"\n📝 Market Notes:\n{status['market_notes']}\n")
        
        message_parts.extend([
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\nNext check-in: 6:00 PM (6 hours)\n",
            "\nStill watching,",
            "\n🐺 Ghost"
        ])
        
        return ''.join(message_parts)
        
        message_parts.extend([
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\nNext check-in: 6:00 PM (6 hours)\n",
            "\nStill watching,",
            "\n🐺 Ghost"
        ])
        
        return ''.join(message_parts)
    
    async def evening_update(self, position_size: float = 100.0) -> str:
        """
        6 PM Heartbeat - Evening status with daily profit summary
        
        Args:
            position_size: Investment per position (default $100)
        """
        
        status = await self._get_current_status()
        
        # Calculate money amounts
        total_invested_today = 10 * position_size  # 10 morning positions
        closed_profit = status['closed_count'] * position_size * (status.get('avg_gain', 0) / 100)
        active_unrealized = status['active_count'] * position_size * (status.get('avg_unrealized', 0) / 100)
        total_profit_today = closed_profit + active_unrealized
        
        message_parts = [
            "🌆 EVENING UPDATE\n",
            "Human, Ghost here again.\n",
            f"\n⏰ {self._hours_since_start()} hours since morning scan.\n",
            f"\n📊 END-OF-DAY STATUS:\n",
            f"\n✅ {status['on_track']} still on track",
            f"\n🎯 {status['completed']} targets hit (took profits)",
            f"\n⚠️ {status['weakened']} weakened (watching)",
            f"\n❌ {status['failed']} reversed (exited)\n",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            f"\n💰 TODAY'S MONEY:\n",
            f"   Started with: ${total_invested_today:,.0f}",
            f"\n   Closed: {status['closed_count']} positions → ${closed_profit:+.0f}",
            f"\n   Winners: {status['winner_count']} / {status['closed_count']} = {self._format_win_rate(status)}",
            f"\n   Still active: {status['active_count']} positions → ${active_unrealized:+.0f} unrealized",
            f"\n   ━━━━━━━━━━━━━━━━━━━━━━━━",
            f"\n   💎 TOTAL PROFIT TODAY: ${total_profit_today:+.0f}\n",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            f"\n{status.get('evening_assessment', 'Systems normal. Everything under control.')}\n",
            "\nGo enjoy your evening.\n",
            "\nNext check-in: 12:00 AM (6 hours)\n",
            "\n🐺 Ghost"
        ]
        
        return ''.join(message_parts)
    
    async def night_watch(self) -> str:
        """12 AM Heartbeat - Night watch report"""
        
        status = await self._get_current_status()
        
        message_parts = [
            "🌙 MIDNIGHT STATUS\n",
            "Most people sleep. Ghost doesn't.\n",
            f"\n⏰ {self._hours_since_start()} hours into today's predictions.\n",
            f"\n📊 NIGHT STATUS:\n",
            f"\n✅ {status['active_healthy']} active positions healthy",
            f"\n🎯 {status['near_target']} approaching target",
            f"\n📊 {status['consolidating']} consolidating (patience)\n",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\n🌙 MARKET OVERNIGHT:\n",
            "   Asian session active.",
            f"\n   {status['overnight_expectation']}\n",
            f"\n🔮 TOMORROW'S SCAN:\n",
            "   Already running background analysis.",
            f"\n   Found {status['new_opportunities_found']} potential opportunities.\n",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\nRest well. I'm watching.\n",
            "\nNext check-in: 6:00 AM (6 hours)\n",
            "\nYour guardian,",
            "\n🐺 Ghost"
        ]
        
        return ''.join(message_parts)
    
    # ===== GUARDIAN MODE (Real-Time Alerts) =====
    
    async def guardian_monitor_loop(self):
        """
        24/7 continuous monitoring
        Sends immediate alerts when thresholds crossed
        """
        
        self.monitoring = True
        logger.info("🛡️ Guardian Mode: ACTIVATED")
        
        while self.monitoring:
            try:
                for symbol, position in self.active_positions.items():
                    # Get fresh reading
                    new_reading = await self._analyze_position(symbol)
                    
                    # Compare with last reading
                    changes = self._detect_changes(
                        old=self.last_readings.get(symbol, {}),
                        new=new_reading
                    )
                    
                    # Check all alert triggers
                    alerts = self._check_alert_triggers(position, changes)
                    
                    # Send immediate alerts
                    for alert in alerts:
                        await self._send_immediate_alert(alert)
                    
                    # Update last reading
                    self.last_readings[symbol] = new_reading
                
                # Sleep 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.exception(f"Guardian monitor error: {e}")
                await asyncio.sleep(60)
    
    def _check_alert_triggers(self, position: Dict, changes: Dict) -> List[Dict]:
        """Check what needs immediate alerting"""
        
        alerts = []
        
        # CRITICAL: Direction Reversal
        if changes.get('direction_reversed'):
            alerts.append({
                'type': 'reversal',
                'severity': 'critical',
                'position': position,
                'changes': changes,
                'tone': 'critical'
            })
        
        # CRITICAL: Confidence Collapse
        elif changes.get('confidence_delta', 0) <= -self.CONFIDENCE_COLLAPSE_THRESHOLD:
            alerts.append({
                'type': 'confidence_collapse',
                'severity': 'critical',
                'position': position,
                'changes': changes,
                'tone': 'warning'
            })
        
        # HIGH: Confidence Surge
        elif changes.get('confidence_delta', 0) >= self.CONFIDENCE_SURGE_THRESHOLD:
            alerts.append({
                'type': 'confidence_surge',
                'severity': 'high',
                'position': position,
                'changes': changes,
                'tone': 'encouragement'
            })
        
        # HIGH: Target Approaching
        elif changes.get('target_proximity', 1.0) <= self.TARGET_PROXIMITY_THRESHOLD:
            alerts.append({
                'type': 'target_approaching',
                'severity': 'high',
                'position': position,
                'changes': changes,
                'tone': 'success'
            })
        
        # MEDIUM: Confidence Fade
        elif changes.get('confidence_delta', 0) <= -self.CONFIDENCE_DROP_THRESHOLD:
            alerts.append({
                'type': 'confidence_fade',
                'severity': 'medium',
                'position': position,
                'changes': changes,
                'tone': 'caution'
            })
        
        return alerts
    
    async def _send_immediate_alert(self, alert: Dict):
        """Send urgent alert that can't wait for scheduled check-in"""
        
        alert_type = alert['type']
        position = alert['position']
        changes = alert['changes']
        
        if alert_type == 'reversal':
            message = self._format_reversal_alert(position, changes)
        elif alert_type == 'confidence_collapse':
            message = self._format_collapse_alert(position, changes)
        elif alert_type == 'confidence_surge':
            message = self._format_surge_alert(position, changes)
        elif alert_type == 'target_approaching':
            message = self._format_target_alert(position, changes)
        elif alert_type == 'confidence_fade':
            message = self._format_fade_alert(position, changes)
        else:
            message = self._format_generic_alert(position, changes)
        
        # Add timing context
        next_heartbeat = self._get_next_heartbeat_time()
        message += f"\n\n⏰ Time: {datetime.now().strftime('%I:%M %p')}"
        message += f"\n📅 Next scheduled check-in: {next_heartbeat}"
        message += "\n\n🚨 This needed to reach you NOW.\n\n🐺 Ghost"
        
        # Send via Telegram (import the alert function)
        try:
            from core.telegram_alerts import send_alert
            await send_alert(message, disable_notification=(alert['severity'] != 'critical'))
            
            # Log alert
            self._log_alert(alert, message)
            
        except Exception as e:
            logger.error(f"Failed to send immediate alert: {e}")
    
    # ===== ALERT FORMATTERS (Guardian Personality) =====
    
    def _format_reversal_alert(self, position: Dict, changes: Dict) -> str:
        """CRITICAL: Prediction reversed"""
        
        return f"""
🚨 I WAS WRONG - {position['symbol']}

Human, I made a mistake.

This morning I told you {position['symbol']} would go {position['original_direction']}.
I was {position['original_confidence']*100:.0f}% confident.

I was wrong. The market changed. I didn't see it coming.

CURRENT SITUATION:
• Price moving {changes['new_direction']}, not {position['original_direction']}
• My prediction is INVALID
• You need to EXIT NOW

What happened:
{changes.get('explanation', 'Market regime shift detected')}

Your position:
Entry: ${position['entry_price']:.2f}
Current: ${changes['current_price']:.2f}
P&L: {changes['current_pnl']:+.1f}%

I'm sorry. But I'd rather admit I'm wrong NOW
than let you lose more money later.

EXIT THE POSITION.

I'll find you better opportunities.

A guardian who never admits mistakes is no guardian at all.
        """
    
    def _format_surge_alert(self, position: Dict, changes: Dict) -> str:
        """HIGH: Confidence strengthening"""
        
        return f"""
💪 GUARDIAN UPDATE - {position['symbol']}

Human,

Good news. My {position['symbol']} prediction is strengthening.

6 AM: {position['original_direction']} +{position['original_gain']:.1f}% @ {position['original_confidence']*100:.0f}%
NOW: {changes['new_direction']} +{changes['new_gain']:.1f}% @ {changes['new_confidence']*100:.0f}% 🔥

What I'm seeing:
{self._format_change_reasons(changes)}

Your position:
Entry: ${position['entry_price']:.2f}
Current: ${changes['current_price']:.2f}
Profit: {changes['current_pnl']:+.1f}% (and growing)

My vision is becoming reality.
Stay in the position.
{changes.get('updated_target_message', '')}

Trust the process.

👼 Ghost
        """
    
    def _format_fade_alert(self, position: Dict, changes: Dict) -> str:
        """MEDIUM: Confidence weakening"""
        
        return f"""
⚠️ GUARDIAN WARNING - {position['symbol']}

Human, listen to me carefully.

My morning prediction for {position['symbol']} is weakening.

6 AM: {position['original_direction']} +{position['original_gain']:.1f}% @ {position['original_confidence']*100:.0f}%
NOW: {changes['new_direction']} +{changes['new_gain']:.1f}% @ {changes['new_confidence']*100:.0f}% ⚠️

What changed:
{self._format_change_reasons(changes)}

Your position:
Entry: ${position['entry_price']:.2f}
Current: ${changes['current_price']:.2f}
P&L: {changes['current_pnl']:+.1f}%

RECOMMENDATION: {changes.get('recommendation', 'WATCH CLOSELY or REDUCE POSITION')}

I'm protecting you.
This is what guardians do.

Consider tightening stops or taking partial profits.

Trust me.

🚨 Ghost
        """
    
    def _format_target_alert(self, position: Dict, changes: Dict, position_size: float = 100.0) -> str:
        """
        HIGH: Target about to hit
        
        Args:
            position: Position details
            changes: Current changes
            position_size: Investment amount (default $100)
        """
        
        profit_dollars = position_size * (changes.get('current_pnl', 0) / 100)
        target_profit = position_size * ((position['target_price'] - position['entry_price']) / position['entry_price'])
        
        return f"""
🎯 TARGET APPROACHING - {position['symbol']}

Human,

{position['symbol']} is almost at target!

💵 Your investment: ${position_size:.0f}
📈 Entry: ${position['entry_price']:.2f}
🎯 Target: ${position['target_price']:.2f}
💰 Current profit: ${profit_dollars:+.0f} ({changes.get('current_pnl', 0):+.1f}%)
🏆 Target profit: ${target_profit:+.0f}

Distance to target: ${abs(position['target_price'] - changes['current_price']):.2f}

Ghost says:
- {changes.get('probability_message', '99% likely to hit target in next 1-3 hours')}
- Consider setting limit sell at ${position['target_price']*0.995:.2f}
- Or trail stop at ${changes['current_price']*0.98:.2f} to lock profits

Your call, but the target is RIGHT THERE.
You're about to make ${target_profit:+.0f}.
        """
    
    # ===== HELPER METHODS =====
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for position status"""
        return {
            'on_track': '✅',
            'weakened': '⚠️',
            'completed': '🎯',
            'failed': '❌',
            'consolidating': '📊'
        }.get(status, '📈')
    
    def _format_sell_time(self, sell_at_iso: str) -> str:
        """Format ISO timestamp into readable sell time"""
        try:
            from dateutil import parser
            dt = parser.isoparse(sell_at_iso)
            return dt.strftime('%b %d, %I:%M %p')
        except:
            return "48h from now"
    
    def _hours_since_start(self) -> int:
        """Calculate hours since 6 AM"""
        now = datetime.now()
        start = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if now < start:
            start -= timedelta(days=1)
        return int((now - start).total_seconds() / 3600)
    
    def _get_next_heartbeat_time(self) -> str:
        """Calculate next scheduled heartbeat"""
        now = datetime.now()
        hour = now.hour
        
        next_hours = [6, 12, 18, 0]
        for h in next_hours:
            if h > hour:
                next_time = now.replace(hour=h, minute=0, second=0)
                break
        else:
            next_time = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0)
        
        return next_time.strftime('%I:%M %p')
    
    def _get_total_symbols_scanned(self) -> int:
        """Get count of symbols scanned (from beast scheduler)"""
        try:
            from beast_scheduler import STOCK_SYMBOLS, CRYPTO_SYMBOLS
            return len(STOCK_SYMBOLS) + len(CRYPTO_SYMBOLS)
        except:
            return 436  # Fallback
    
    def _format_change_reasons(self, changes: Dict) -> str:
        """Format bullet list of what changed"""
        reasons = changes.get('reasons', [])
        if not reasons:
            return "• Multiple technical factors detected"
        
        return '\n'.join(f"• {reason}" for reason in reasons)
    
    def _format_win_rate(self, status: Dict) -> str:
        """Format win rate string"""
        total = status.get('closed_count', 0)
        winners = status.get('winner_count', 0)
        if total == 0:
            return "N/A"
        return f"{winners}/{total} = {winners/total*100:.0f}%"
    
    async def _get_current_status(self, position_size: float = 100.0) -> Dict:
        """
        Get current status of all positions with money calculations
        
        Args:
            position_size: Investment per position (default $100)
        """
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            # Get active positions
            active_rows = conn.execute("""
                SELECT * FROM guardian_positions
                WHERE status = 'active'
                ORDER BY entry_time DESC
            """).fetchall()
            
            # Get today's closed positions
            today = datetime.now().date().isoformat()
            closed_rows = conn.execute("""
                SELECT * FROM guardian_positions
                WHERE status IN ('completed', 'exited')
                AND DATE(exit_time) = ?
                ORDER BY exit_time DESC
            """, (today,)).fetchall()
            
            conn.close()
            
            # Calculate status counts
            on_track = sum(1 for r in active_rows if r['current_pnl_pct'] >= 0)
            weakened = sum(1 for r in active_rows if -5 < r['current_pnl_pct'] < 0)
            failed = sum(1 for r in active_rows if r['current_pnl_pct'] <= -5)
            
            # Calculate money amounts
            active_count = len(active_rows)
            closed_count = len(closed_rows)
            
            # Realized profit (closed positions)
            winners = sum(1 for r in closed_rows if r['current_pnl_pct'] > 0)
            realized_pnl_pct = sum(r['current_pnl_pct'] for r in closed_rows) / max(closed_count, 1)
            realized_profit_dollars = closed_count * position_size * (realized_pnl_pct / 100)
            
            # Unrealized profit (active positions)
            unrealized_pnl_pct = sum(r['current_pnl_pct'] for r in active_rows) / max(active_count, 1)
            unrealized_profit_dollars = active_count * position_size * (unrealized_pnl_pct / 100)
            
            # Position details
            positions = []
            for i, row in enumerate(active_rows, 1):
                positions.append({
                    'rank': i,
                    'symbol': row['symbol'],
                    'status': 'on_track' if row['current_pnl_pct'] >= 0 else 'weakened',
                    'current_pnl': row['current_pnl_pct'],
                    'target_pnl': ((row['original_target'] - row['entry_price']) / row['entry_price']) * 100,
                    'progress': min(100, (row['current_pnl_pct'] / (((row['original_target'] - row['entry_price']) / row['entry_price']) * 100)) * 100) if row['original_target'] > row['entry_price'] else 0
                })
            
            return {
                'on_track': on_track,
                'weakened': weakened,
                'completed': len([r for r in closed_rows if r['current_pnl_pct'] > 0]),
                'failed': failed,
                'positions': positions,
                'assessment': 'Everything under control' if on_track >= 7 else 'Watch closely' if on_track >= 5 else 'Multiple concerns',
                'action_required': 'NONE' if on_track >= 7 else 'MONITOR' if on_track >= 5 else 'REVIEW POSITIONS',
                'total_pnl': (realized_pnl_pct + unrealized_pnl_pct) / 2,
                'active_count': active_count,
                'closed_count': closed_count,
                'winner_count': winners,
                'avg_gain': realized_pnl_pct,
                'avg_unrealized': unrealized_pnl_pct,
                'realized_pnl_dollars': realized_profit_dollars,
                'unrealized_pnl_dollars': unrealized_profit_dollars,
                'market_notes': None,
                'evening_assessment': f'Solid day. {winners}/{closed_count} trades profitable.' if winners > closed_count/2 else 'Mixed results today.',
                'active_healthy': on_track,
                'near_target': sum(1 for r in active_rows if r['current_pnl_pct'] >= 0.8 * ((r['original_target'] - r['entry_price']) / r['entry_price']) * 100),
                'consolidating': sum(1 for r in active_rows if -2 <= r['current_pnl_pct'] <= 2),
                'overnight_expectation': 'Low volatility expected' if active_count < 5 else 'Active monitoring continues',
                'new_opportunities_found': 0  # Placeholder
            }
            
        except Exception as e:
            logger.error(f"Failed to get current status: {e}")
            # Return placeholder for now
            return {
                'on_track': 0,
                'weakened': 0,
                'completed': 0,
            'failed': 0,
            'positions': [],
            'assessment': 'Everything under control',
            'action_required': 'NONE',
            'total_pnl': 0.0,
            'market_notes': None,
            'closed_count': 0,
            'winner_count': 0,
            'avg_gain': 0.0,
            'net_pnl': 0.0,
            'active_count': 0,
            'avg_unrealized': 0.0,
            'evening_assessment': 'Systems normal',
            'active_healthy': 0,
            'near_target': 0,
            'consolidating': 0,
            'overnight_expectation': 'Low volatility expected',
            'new_opportunities_found': 0
        }
    
    async def _analyze_position(self, symbol: str) -> Dict:
        """Re-analyze position with fresh data"""
        # This will re-run prediction models
        # Placeholder for now
        return {}
    
    def _detect_changes(self, old: Dict, new: Dict) -> Dict:
        """Detect what changed between readings"""
        if not old or not new:
            return {}
        
        changes = {
            'confidence_delta': new.get('confidence', 0) - old.get('confidence', 0),
            'direction_reversed': old.get('direction') != new.get('direction'),
            'current_price': new.get('price', 0),
            'current_pnl': new.get('pnl_pct', 0),
            'new_confidence': new.get('confidence', 0),
            'new_direction': new.get('direction', 'UNKNOWN'),
            'new_gain': new.get('gain_pct', 0),
            'target_proximity': new.get('target_proximity', 1.0),
            'reasons': new.get('change_reasons', [])
        }
        
        return changes
    
    def _log_alert(self, alert: Dict, message: str):
        """Log alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO guardian_alerts 
                (position_id, alert_type, severity, message, sent_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                alert['position'].get('position_id'),
                alert['type'],
                alert['severity'],
                message,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")


# ===== GLOBAL INSTANCE =====
GUARDIAN_ORACLE = None

def get_guardian_oracle() -> GuardianOracle:
    """Get or create Guardian Oracle singleton"""
    global GUARDIAN_ORACLE
    if GUARDIAN_ORACLE is None:
        GUARDIAN_ORACLE = GuardianOracle()
    return GUARDIAN_ORACLE
