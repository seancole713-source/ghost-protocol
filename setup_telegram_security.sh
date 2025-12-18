#!/bin/bash
# Quick Setup - Telegram Bot Security Tools
# Run this to test the tools locally

echo "============================================================"
echo "🔧 TELEGRAM BOT SECURITY - QUICK SETUP"
echo "============================================================"
echo ""

# Check if token is set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "⚠️  TELEGRAM_BOT_TOKEN not set in environment"
    echo ""
    echo "Get your token from Railway:"
    echo "  1. Go to Railway dashboard"
    echo "  2. Click on ghost-protocol project"
    echo "  3. Go to Variables tab"
    echo "  4. Copy TELEGRAM_BOT_TOKEN value"
    echo ""
    echo "Then run:"
    echo "  export TELEGRAM_BOT_TOKEN='paste_token_here'"
    echo "  ./setup_telegram_security.sh"
    echo ""
    exit 1
fi

echo "✅ Token found: ${TELEGRAM_BOT_TOKEN:0:10}...${TELEGRAM_BOT_TOKEN: -5}"
echo ""

# Check if CHAT_ID is set
if [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "⚠️  TELEGRAM_CHAT_ID not set (optional, but recommended for alerts)"
    echo ""
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
python3 -c "import requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing requests..."
    pip install requests -q
fi
echo "✅ Dependencies OK"
echo ""

# Run bot name check
echo "============================================================"
echo "1️⃣  CHECKING CURRENT BOT NAME"
echo "============================================================"
echo ""
python monitor_telegram_bot.py
CHECK_RESULT=$?
echo ""

# If name is wrong, offer to fix
if [ $CHECK_RESULT -ne 0 ]; then
    echo "⚠️  Bot name needs fixing!"
    echo ""
    read -p "Reset bot name to 'Ghost Protocol Bot'? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "============================================================"
        echo "2️⃣  RESETTING BOT NAME"
        echo "============================================================"
        echo ""
        python reset_telegram_bot_name.py
        echo ""
    else
        echo "Skipped reset. Run manually: python reset_telegram_bot_name.py"
        echo ""
    fi
else
    echo "✅ Bot name is correct - no action needed"
    echo ""
fi

# Offer to start monitoring
echo "============================================================"
echo "3️⃣  MONITORING OPTIONS"
echo "============================================================"
echo ""
echo "Choose monitoring mode:"
echo "  1) Single check (completed above)"
echo "  2) Start continuous monitoring (every 5 min)"
echo "  3) Skip monitoring"
echo ""
read -p "Enter choice (1-3): " -n 1 -r
echo ""

case $REPLY in
    2)
        echo ""
        echo "Starting continuous monitoring..."
        echo "Press Ctrl+C to stop"
        echo ""
        python monitor_telegram_bot.py --continuous
        ;;
    3)
        echo "Skipped monitoring"
        ;;
    *)
        echo "Single check complete"
        ;;
esac

echo ""
echo "============================================================"
echo "✅ SETUP COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  • Check Telegram to verify bot name"
echo "  • Test messaging: python test_telegram_send.py"
echo "  • View logs: cat logs/telegram_bot_monitor.log"
echo "  • Read docs: cat TELEGRAM_BOT_SECURITY.md"
echo ""
