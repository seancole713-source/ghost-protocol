#!/bin/bash
set -e

echo "╔════════════════════════════════════════════════════════╗"
echo "║      Ghost Trading - VPS Docker Deployment        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root or with sudo"
    exit 1
fi

# Get VPS info
read -p "Enter your VPS IP address: " VPS_IP
read -p "Enter SSH username [root]: " SSH_USER
SSH_USER=${SSH_USER:-root}

echo
echo "🔐 This script will:"
echo "  1. Setup Docker on your VPS"
echo "  2. Clone Ghost repository"
echo "  3. Configure environment"
echo "  4. Deploy Ghost with Docker Compose"
echo
read -p "Continue? (y/n): " CONTINUE

if [ "$CONTINUE" != "y" ]; then
    echo "Deployment cancelled."
    exit 0
fi

# Connect to VPS and setup
echo "📡 Connecting to VPS..."

ssh $SSH_USER@$VPS_IP << 'ENDSSH'
set -e

echo "📦 Installing dependencies..."
apt update
apt install -y git curl

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    apt install -y docker-compose
fi

# Clone repository
echo "📥 Cloning Ghost repository..."
cd /opt
if [ -d "GHOST" ]; then
    echo "Repository already exists, pulling latest..."
    cd GHOST
    git pull origin main
else
    git clone https://github.com/seancole713-source/GHOST.git
    cd GHOST
fi

# Create production environment file
echo "⚙️  Creating production environment file..."
read -r -p "Enter GHOST_API_TOKEN (required): " GHOST_API_TOKEN
read -r -p "Enter POLYGON_API_KEY (press Enter to skip): " POLYGON_API_KEY
read -r -p "Enter ALPHAVANTAGE_API_KEY (press Enter to skip): " ALPHAVANTAGE_API_KEY
read -r -p "Enter TELEGRAM_BOT_TOKEN (optional): " TELEGRAM_BOT_TOKEN
read -r -p "Enter TELEGRAM_CHAT_ID (optional): " TELEGRAM_CHAT_ID

cat > .env.production <<EOF
GHOST_API_TOKEN=$GHOST_API_TOKEN
POLYGON_API_KEY=$POLYGON_API_KEY
ALPHAVANTAGE_API_KEY=$ALPHAVANTAGE_API_KEY
GHOST_FOCUS_TICKER=WOLF
WOLF_PERSIST_MODE=sqlite
WOLF_SQLITE_PATH=/app/data/wolf.db
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID
EOF

echo
echo "✅ Secrets written to /opt/GHOST/.env.production"

# Create docker-compose.yml if not exists
if [ ! -f "docker-compose.yml" ]; then
    echo "📝 Creating docker-compose.yml..."
    cat > docker-compose.yml << 'EOFCOMPOSE'
version: '3.8'

services:
  ghost:
    build: .
    container_name: ghost-trading
    restart: unless-stopped
    ports:
      - "80:5000"
    env_file:
      - .env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
EOFCOMPOSE
fi

# Create Dockerfile if not exists
if [ ! -f "Dockerfile" ]; then
    echo "📝 Creating Dockerfile..."
    cat > Dockerfile << 'EOFDOCKER'
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/logs

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["uvicorn", "wolf_app:app", "--host", "0.0.0.0", "--port", "5000"]
EOFDOCKER
fi

# Create directories
mkdir -p data logs

# Build and start
echo "🚀 Building and starting Ghost..."
docker-compose down 2>/dev/null || true
docker-compose up -d --build

# Wait for startup
echo "⏳ Waiting for Ghost to start..."
sleep 10

# Check health
echo "🏥 Checking health..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ Ghost is healthy!"
else
    echo "⚠️  Health check failed. Checking logs..."
    docker-compose logs --tail=50
fi

echo
echo "╔════════════════════════════════════════════════════════╗"
echo "║              Deployment Complete!                  ║"
echo "╚════════════════════════════════════════════════════════╝"
echo
echo "🎉 Ghost is now running at: http://$HOSTNAME"
echo
echo "📊 Useful commands:"
echo "  View logs:    docker-compose logs -f"
echo "  Restart:      docker-compose restart"
echo "  Stop:         docker-compose down"
echo "  Update:       git pull && docker-compose up -d --build"
echo
echo "🔒 Next steps:"
echo "  1. Setup firewall (ufw allow 80/tcp)"
echo "  2. Configure domain name"
echo "  3. Setup SSL with certbot"
echo "  4. Configure backups"
echo

ENDSSH

echo
echo "✅ VPS deployment complete!"
echo "🔗 Access Ghost at: http://$VPS_IP"
echo
echo "Test: curl http://$VPS_IP/health"
