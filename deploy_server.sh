#!/bin/bash
# Ghost 24/7 Deployment Script - AWS EC2 / VPS
# Usage: Run this on a fresh Ubuntu server

set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║     Ghost Trading System - Server Deployment          ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

# Update system
echo "📦 Updating system packages..."
sudo apt update -qq
sudo apt upgrade -y -qq

# Install dependencies
echo "📦 Installing Python and Git..."
sudo apt install -y python3 python3-pip python3-venv git curl

# Clone repository
if [ ! -d "$HOME/GHOST" ]; then
    echo "📥 Cloning Ghost repository..."
    cd ~
    git clone https://github.com/seancole713-source/GHOST.git
else
    echo "📂 Ghost directory already exists, pulling latest..."
    cd ~/GHOST
    git pull
fi

cd ~/GHOST

# Setup Python environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Create secrets file
echo "🔐 Setting up secrets..."
read -p "Enter GHOST_API_TOKEN (or press Enter for default): " TOKEN
TOKEN=${TOKEN:-supersecret123jamaica713}

read -p "Enter POLYGON_API_KEY: " POLYGON_KEY
read -p "Enter ALPHAVANTAGE_API_KEY: " ALPHA_KEY

cat > secrets.env << EOF
GHOST_API_TOKEN=$TOKEN
POLYGON_API_KEY=$POLYGON_KEY
ALPHAVANTAGE_API_KEY=$ALPHA_KEY
GHOST_FOCUS_TICKER=WOLF
WOLF_SQLITE_PATH=/home/$USER/GHOST/data/wolf.db
PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
EOF

echo "✅ Secrets configured"

# Create systemd service
echo "⚙️  Creating systemd service..."
sudo tee /etc/systemd/system/ghost.service > /dev/null << SVCEOF
[Unit]
Description=Ghost Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/GHOST
Environment="PATH=/home/$USER/GHOST/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PYTHONUNBUFFERED=1"
EnvironmentFile=/home/$USER/GHOST/secrets.env
ExecStartPre=/bin/mkdir -p /tmp/ghost_prom
ExecStartPre=/bin/mkdir -p /home/$USER/GHOST/data
ExecStart=/home/$USER/GHOST/.venv/bin/uvicorn wolf_app:app --host 0.0.0.0 --port 5000
Restart=always
RestartSec=10
StandardOutput=append:/var/log/ghost.log
StandardError=append:/var/log/ghost.log

[Install]
WantedBy=multi-user.target
SVCEOF

# Create log file with proper permissions
sudo touch /var/log/ghost.log
sudo chown $USER:$USER /var/log/ghost.log

# Create daily backup cron
echo "💾 Setting up daily backups..."
(crontab -l 2>/dev/null || true; echo "0 2 * * * cd $HOME/GHOST/data && tar -czf ~/ghost-backup-\$(date +\\%Y\\%m\\%d).tar.gz *.db && find ~ -name 'ghost-backup-*.tar.gz' -mtime +7 -delete") | crontab -

# Enable and start service
echo "🚀 Starting Ghost service..."
sudo systemctl daemon-reload
sudo systemctl enable ghost
sudo systemctl start ghost

# Wait for startup
sleep 3

# Check status
echo
echo "📊 Service Status:"
sudo systemctl status ghost --no-pager || true

echo
echo "🧪 Testing health endpoint..."
sleep 2
PUBLIC_IP=$(curl -s ifconfig.me)
curl -sS http://localhost:5000/health | jq . || echo "Health check pending..."

echo
echo "╔════════════════════════════════════════════════════════╗"
echo "║           Ghost is now running 24/7! 🎉               ║"
echo "╚════════════════════════════════════════════════════════╝"
echo
echo "📍 Public IP: $PUBLIC_IP"
echo "🌐 Access Ghost at: http://$PUBLIC_IP:5000"
echo
echo "Useful commands:"
echo "  sudo systemctl status ghost   # Check status"
echo "  sudo systemctl restart ghost  # Restart"
echo "  sudo journalctl -u ghost -f   # View logs"
echo "  tail -f /var/log/ghost.log    # View app logs"
echo
echo "🔐 Don't forget to configure firewall:"
echo "  sudo ufw allow 5000/tcp"
echo "  sudo ufw enable"
echo
echo "🧪 Test health:"
echo "  curl http://localhost:5000/health/detailed | jq ."
