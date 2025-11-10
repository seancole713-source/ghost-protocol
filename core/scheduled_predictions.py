"""
Ghost Scheduled Market Predictions
Sends automated predictions at key market times:
- 8:00 AM ET: Pre-market prediction
- 9:35 AM ET: 5 minutes after market open (compare prediction vs actual)
"""

import threading
from datetime import datetime

import pytz

# Will be set by wolf_app.py
TELEGRAM_SEND_FUNC = None
TELEGRAM_CHAT_ID = None
GET_WOLF_PRICE_FUNC = None
EVALUATE_SIGNAL_FUNC = None
LOGGER = None

# Tracking state
_PREDICTION_WORKER: threading.Thread | None = None
_PREDICTION_STOP = threading.Event()
_LAST_PREMARKET_DATE: str | None = None
_LAST_OPEN_CHECK_DATE: str | None = None
_PREMARKET_PREDICTION: dict | None = None  # Store 8am prediction for comparison


def _get_ny_time():
    """Get current time in New York timezone"""
    return datetime.now(pytz.timezone("America/New_York"))


def _is_market_day(dt):
    """Check if it's a weekday (Mon-Fri)"""
    return dt.weekday() <= 4


def start_prediction_scheduler():
    """Start the scheduled prediction worker"""
    global _PREDICTION_WORKER
    if _PREDICTION_WORKER is None or not _PREDICTION_WORKER.is_alive():
        _PREDICTION_STOP.clear()
        _PREDICTION_WORKER = threading.Thread(
            target=_prediction_loop, name="prediction-scheduler", daemon=True
        )
        _PREDICTION_WORKER.start()
        if LOGGER:
            LOGGER.info("📅 Prediction scheduler started (8:00 AM & 9:35 AM ET)")
        print("[PREDICTION SCHEDULER] Started - will send at 8:00 AM & 9:35 AM ET")


def stop_prediction_scheduler():
    """Stop the prediction scheduler"""
    try:
        _PREDICTION_STOP.set()
        if _PREDICTION_WORKER and _PREDICTION_WORKER.is_alive():
            _PREDICTION_WORKER.join(timeout=2.0)
    except Exception:
        pass


def _send_telegram_message(text: str):
    """Send message via Telegram"""
    if TELEGRAM_SEND_FUNC and TELEGRAM_CHAT_ID:
        try:
            TELEGRAM_SEND_FUNC(TELEGRAM_CHAT_ID, text)
            return True
        except Exception as e:
            if LOGGER:
                LOGGER.error(f"Failed to send Telegram prediction: {e}")
            return False
    return False


def _send_premarket_prediction():
    """Send 8:00 AM pre-market prediction"""
    global _PREMARKET_PREDICTION

    try:
        # Get current price and prediction
        if not GET_WOLF_PRICE_FUNC or not EVALUATE_SIGNAL_FUNC:
            return

        price, prev_close, provider = GET_WOLF_PRICE_FUNC()
        signal = EVALUATE_SIGNAL_FUNC()

        # Calculate changes
        if price and prev_close:
            change = price - prev_close
            change_pct = (change / prev_close) * 100
        else:
            change = 0
            change_pct = 0

        # Store prediction for later comparison
        _PREMARKET_PREDICTION = {
            "timestamp": _get_ny_time().isoformat(),
            "price": price,
            "prev_close": prev_close,
            "signal": signal.get("action", "HOLD"),
            "confidence": signal.get("confidence", 0),
            "predicted_direction": signal.get("action", "HOLD"),
            "factors": signal.get("factors", []),
        }

        # Build message
        now_str = _get_ny_time().strftime("%I:%M %p %Z")

        message = f"""🌅 <b>PRE-MARKET PREDICTION</b>
⏰ Time: {now_str}

📊 <b>CURRENT STATUS</b>
Symbol: WOLF
Current: ${price:.2f}
Prev Close: ${prev_close:.2f}
Change: ${change:+.2f} ({change_pct:+.2f}%)
Provider: {provider or "N/A"}

🎯 <b>GHOST PREDICTION</b>
Action: <b>{signal.get("action", "HOLD")}</b>
Confidence: {signal.get("confidence", 0):.0f}%
Direction: {signal.get("action", "HOLD")}

📈 <b>KEY FACTORS:</b>
"""

        # Add factors
        factors = signal.get("factors", [])
        if factors:
            for factor in factors[:5]:  # Top 5 factors
                message += f"• {factor}\n"
        else:
            message += "• No factors available\n"

        message += """
💡 <b>STRATEGY:</b>
"""

        if signal.get("action") == "BUY":
            message += "📈 Ghost predicts UPWARD movement today\n"
            message += "Consider buying if you're comfortable with the confidence level"
        elif signal.get("action") == "SELL":
            message += "📉 Ghost predicts DOWNWARD movement today\n"
            message += "Consider reducing position or waiting"
        else:
            message += "⏸️ Ghost predicts SIDEWAYS/HOLD today\n"
            message += "Market may be uncertain - wait for clearer signals"

        message += "\n⏰ <i>Will check again at 9:35 AM (5 min after market open)</i>"

        # Send message
        success = _send_telegram_message(message)

        if success:
            print(f"[PREDICTION] ✅ Sent pre-market prediction at {now_str}")
            if LOGGER:
                LOGGER.info(f"Sent pre-market prediction: {signal.get('action')} @ ${price:.2f}")

    except Exception as e:
        print(f"[PREDICTION] ❌ Error sending pre-market prediction: {e}")
        if LOGGER:
            LOGGER.error(f"Pre-market prediction error: {e}")


def _send_market_open_comparison():
    """Send 9:35 AM comparison (prediction vs actual)"""
    global _PREMARKET_PREDICTION

    try:
        if not GET_WOLF_PRICE_FUNC or not _PREMARKET_PREDICTION:
            return

        # Get current price
        current_price, _, provider = GET_WOLF_PRICE_FUNC()

        # Get prediction data
        predicted_price = _PREMARKET_PREDICTION.get("price", 0)
        predicted_action = _PREMARKET_PREDICTION.get("predicted_direction", "HOLD")
        predicted_confidence = _PREMARKET_PREDICTION.get("confidence", 0)

        # Calculate actual movement
        if current_price and predicted_price:
            price_change = current_price - predicted_price
            price_change_pct = (price_change / predicted_price) * 100

            # Determine if prediction was correct
            actual_direction = (
                "UP" if price_change > 0 else ("DOWN" if price_change < 0 else "FLAT")
            )

            # Check accuracy
            if predicted_action == "BUY" and price_change > 0:
                accuracy = "✅ CORRECT"
                emoji = "🎯"
            elif predicted_action == "SELL" and price_change < 0:
                accuracy = "✅ CORRECT"
                emoji = "🎯"
            elif predicted_action == "HOLD" and abs(price_change_pct) < 1:
                accuracy = "✅ CORRECT"
                emoji = "🎯"
            else:
                accuracy = "❌ INCORRECT"
                emoji = "⚠️"
        else:
            price_change = 0
            price_change_pct = 0
            actual_direction = "UNKNOWN"
            accuracy = "⚠️ NO DATA"
            emoji = "❓"

        # Build comparison message
        now_str = _get_ny_time().strftime("%I:%M %p %Z")

        message = f"""{emoji} <b>MARKET OPEN CHECK</b>
⏰ Time: {now_str} (5 min after open)

📊 <b>PREDICTION vs REALITY</b>

<b>8:00 AM PREDICTION:</b>
• Price: ${predicted_price:.2f}
• Action: <b>{predicted_action}</b>
• Confidence: {predicted_confidence:.0f}%

<b>9:35 AM ACTUAL:</b>
• Current: ${current_price:.2f}
• Change: ${price_change:+.2f} ({price_change_pct:+.2f}%)
• Direction: <b>{actual_direction}</b>

<b>RESULT:</b> {accuracy}

"""

        if accuracy == "✅ CORRECT":
            message += "🎉 <b>Ghost prediction was CORRECT!</b>\n"
            message += f"Predicted {predicted_action}, market moved {actual_direction}"
        else:
            message += "📝 <b>Ghost prediction needs adjustment</b>\n"
            message += f"Predicted {predicted_action}, but market moved {actual_direction}"

        message += "\n\n💡 Continue monitoring throughout the day..."

        # Send message
        success = _send_telegram_message(message)

        if success:
            print(f"[PREDICTION] ✅ Sent market open comparison at {now_str}")
            if LOGGER:
                LOGGER.info(
                    f"Market open check: {accuracy} - {predicted_action} vs {actual_direction}"
                )

    except Exception as e:
        print(f"[PREDICTION] ❌ Error sending market open comparison: {e}")
        if LOGGER:
            LOGGER.error(f"Market open comparison error: {e}")


def _prediction_loop():
    """Main loop checking for scheduled prediction times"""
    global _LAST_PREMARKET_DATE, _LAST_OPEN_CHECK_DATE

    print("[PREDICTION SCHEDULER] Loop started, checking every 30 seconds...")

    while not _PREDICTION_STOP.is_set():
        try:
            now = _get_ny_time()

            # Only run on market days (Mon-Fri)
            if not _is_market_day(now):
                _PREDICTION_STOP.wait(60.0)  # Check every minute on weekends
                continue

            current_date = now.strftime("%Y-%m-%d")
            current_time = now.time()

            # Check for 8:00 AM pre-market prediction (within 5-minute window)
            premarket_time = datetime.strptime("08:00", "%H:%M").time()
            time_diff_premarket = abs(
                (
                    datetime.combine(now.date(), current_time)
                    - datetime.combine(now.date(), premarket_time)
                ).total_seconds()
            )

            if (
                time_diff_premarket <= 150 and _LAST_PREMARKET_DATE != current_date
            ):  # 2.5 min window
                print(
                    f"[PREDICTION] 🌅 Triggering pre-market prediction at {now.strftime('%H:%M')}"
                )
                _send_premarket_prediction()
                _LAST_PREMARKET_DATE = current_date

            # Check for 9:35 AM market open comparison (within 5-minute window)
            open_check_time = datetime.strptime("09:35", "%H:%M").time()
            time_diff_open = abs(
                (
                    datetime.combine(now.date(), current_time)
                    - datetime.combine(now.date(), open_check_time)
                ).total_seconds()
            )

            if time_diff_open <= 150 and _LAST_OPEN_CHECK_DATE != current_date:  # 2.5 min window
                print(f"[PREDICTION] 📊 Triggering market open check at {now.strftime('%H:%M')}")
                _send_market_open_comparison()
                _LAST_OPEN_CHECK_DATE = current_date

        except Exception as e:
            print(f"[PREDICTION] ❌ Loop error: {e}")
            if LOGGER:
                LOGGER.error(f"Prediction loop error: {e}")

        finally:
            # Check every 30 seconds
            _PREDICTION_STOP.wait(30.0)


def force_premarket_prediction():
    """Manually trigger pre-market prediction (for testing)"""
    print("[PREDICTION] 🧪 Forcing pre-market prediction...")
    _send_premarket_prediction()


def force_market_open_check():
    """Manually trigger market open check (for testing)"""
    print("[PREDICTION] 🧪 Forcing market open check...")
    _send_market_open_comparison()
