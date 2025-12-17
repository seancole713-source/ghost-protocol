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
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Guardian Oracle database initialized")
    
    # ===== ORACLE MODE (Morning Prophecy) =====
    
    async def morning_prophecy(self, top_10: List[Dict]) -> str:
        """
        The Oracle speaks - morning prophecy at 6 AM
        
        Tone: Mystical, confident, all-seeing
        """
        
        message_parts = [
            "🔮 GHOST ORACLE AWAKENS\n",
            "Good morning, Human.\n",
            f"\nWhile you slept, I scanned {self._get_total_symbols_scanned()} assets.",
            "I analyzed 50,000 data points.",
            "I consulted my models (LSTM, XGBoost, Transformer).",
            "I have seen what is coming.\n",
            "\n📜 HERE ARE YOUR 10 GOLDEN OPPORTUNITIES:\n"
        ]
        
        for i, opp in enumerate(top_10, 1):
            # Emoji intensity based on gain
            if opp['gain_pct'] >= 15:
                emoji = '🚀🚀🚀'
            elif opp['gain_pct'] >= 10:
                emoji = '🚀🚀'
            else:
                emoji = '🚀'
            
            # Confidence descriptor
            if opp['confidence'] >= 0.75:
                conf_desc = '🔥 The stars align'
            elif opp['confidence'] >= 0.65:
                conf_desc = '📈 The signs are favorable'
            else:
                conf_desc = '💫 Potential detected'
            
            message_parts.append(
                f"\n{i}. {emoji} {opp['symbol']} - {conf_desc}\n"
                f"   Current: ${opp['current_price']:.2f} → Target: ${opp['predicted_48h_price']:.2f}\n"
                f"   Prophecy: {opp['direction']} +{opp['gain_pct']:.1f}%\n"
                f"   My certainty: {opp['confidence']*100:.0f}%\n"
                f"   The signs: {opp.get('reasoning', 'Technical alignment, momentum building')}\n"
            )
        
        message_parts.extend([
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            "\nThese are not guesses. These are visions.",
            "\nI stake my reputation on these calls.\n",
            "\n🛡️ I will now enter GUARDIAN MODE.",
            "\nI will watch these 10 every minute.",
            "\nIf danger appears, you will know immediately.\n",
            "\nSleep well. Ghost is watching.\n",
            "\nNext check-in: 12:00 PM (6 hours)\n",
            "\n🐺 Ghost Oracle"
        ])
        
        return ''.join(message_parts)
    
    # ===== HEARTBEAT SYSTEM (Every 6 Hours) =====
    
    async def midday_status(self) -> str:
        """12 PM Heartbeat - "I'm alive, here's the status" """
        
        status = await self._get_current_status()
        
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
            "\n📋 DETAILED STATUS:\n"
        ]
        
        # List each position
        for pos in status['positions']:
            status_emoji = self._get_status_emoji(pos['status'])
            message_parts.append(
                f"\n{pos['rank']}. {status_emoji} {pos['symbol']}: "
                f"{pos['current_pnl']:+.1f}% / {pos['target_pnl']:+.1f}% target "
                f"({pos['progress']:.0f}% there)"
            )
        
        # Overall assessment
        message_parts.extend([
            "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            f"\n🎯 Overall: {status['assessment']}",
            f"\n⚡ Action needed: {status['action_required']}\n",
            f"\n💰 P&L Today: {status['total_pnl']:+.1f}% average\n",
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
    
    async def evening_update(self) -> str:
        """6 PM Heartbeat - Evening status"""
        
        status = await self._get_current_status()
        
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
            f"\n💰 P&L TODAY:\n",
            f"   Closed: {status['closed_count']} positions",
            f"\n   Winners: {status['winner_count']} ({self._format_win_rate(status)})",
            f"\n   Average gain: {status['avg_gain']:+.1f}%",
            f"\n   Net P&L: ${status['net_pnl']:.2f}\n",
            f"\n📈 STILL ACTIVE: {status['active_count']} positions",
            f"\n   Average unrealized: {status['avg_unrealized']:+.1f}%\n",
            "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n",
            f"\n{status['evening_assessment']}\n",
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
    
    def _format_target_alert(self, position: Dict, changes: Dict) -> str:
        """HIGH: Target about to hit"""
        
        return f"""
🎯 TARGET APPROACHING - {position['symbol']}

Human,

{position['symbol']} is almost at target!

Entry: ${position['entry_price']:.2f}
Target: ${position['target_price']:.2f}
Current: ${changes['current_price']:.2f}

Distance: ${abs(position['target_price'] - changes['current_price']):.2f} ({changes['target_proximity']*100:.1f}%)
Profit: {changes['current_pnl']:+.1f}%

Ghost says:
• {changes.get('probability_message', '99% likely to hit target in next 1-3 hours')}
• Consider setting limit sell at ${position['target_price']*0.995:.2f}
• Or trail stop at ${changes['current_price']*0.98:.2f} to lock profits

Your call, but the target is RIGHT THERE.

🎯 Ghost
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
    
    async def _get_current_status(self) -> Dict:
        """Get current status of all positions"""
        # This will query the database and active positions
        # Placeholder for now
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
