# Running Ghost 24/7 - Deployment Guide

## Overview

Ghost currently runs in a Codespace which stops when inactive. This guide provides multiple options for running Ghost
24/7 in production.

---

## 🎯 Quick Comparison

| Option | Cost | Setup Time | Best For |
|--------|------|------------|----------|
| **Railway**| ~$5-20/mo | 15 min | Fastest deployment, automatic SSL |
|**Render**| Free tier available | 20 min | Cost-conscious, simple setup |
|**DigitalOcean**| $6-12/mo | 30 min | Full control, scalability |
|**AWS EC2**| $5-15/mo | 45 min | Enterprise features, AWS ecosystem |
|**Docker + VPS**| $5-10/mo | 60 min | Maximum flexibility |

---

## 🚀 Option 1: Railway (Recommended - Easiest)**Pros**: One-click deploy, automatic SSL, zero-downtime deploys

**Cons**: ~$5-10/month cost
**Setup Time**: 15 minutes

### Step 1: Prepare Your Repository

```bash

# Ensure all changes are committed

git add -A
git commit -m "Prepare for Railway deployment"
git push origin main

```text

### Step 2: Create Railway Configuration

Railway will automatically detect your Python app. Create these files:

**railway.toml**(already provided in repo):

```toml

[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn wolf_app:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

```text

### Step 3: Deploy to Railway

1. Go to [railway.app](<<<<<https://railway.ap>>>>>p)
2. Click "Start a New Project"
3. Choose "Deploy from GitHub repo"
4. Select your GHOST repository
5. Railway will auto-detect Python and deploy


### Step 4: Set Environment Variables

In Railway dashboard, add these variables:

```bash

# Required

GHOST_API_TOKEN=<Railway:GHOST_API_TOKEN>
POLYGON_API_KEY=<Railway:POLYGON_API_KEY>
ALPHAVANTAGE_API_KEY=<Railway:ALPHAVANTAGE_API_KEY>

# Optional

GHOST_FOCUS_TICKER=WOLF
TELEGRAM_BOT_TOKEN=<Railway:TELEGRAM_BOT_TOKEN>
TELEGRAM_CHAT_ID=<Railway:TELEGRAM_CHAT_ID>
WOLF_PERSIST_MODE=sqlite

```text

### Step 5: Add PostgreSQL (Optional but Recommended)

1. In Railway project, click "New"
2. Select "Database" → "PostgreSQL"
3. Railway will auto-inject `DATABASE_URL` into your app


Then update Ghost to use PostgreSQL instead of SQLite for production.

### Step 6: Verify Deployment

```bash

# Your app will be available at

# <<<<<https://your-app-name.up.railway.app>>>>>

# Test health

curl <<<<<https://your-app-name.up.railway.app/health>>>>>

# Test cockpit

curl <<<<<https://your-app-name.up.railway.app/api/cockpit>>>>>

```text**📚 Railway Deployment Script**: See `scripts/deploy_railway.sh`

---

## 🆓 Option 2: Render (Free Tier Available)

**Pros**: Free tier, automatic deploys
**Cons**: Spins down after 15min inactivity (free tier), slower cold starts
**Setup Time**: 20 minutes

### Step 1: Create render.yaml

```yaml

services:

  - type: web


    name: ghost-trading
    env: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn wolf_app:app --host 0.0.0.0 --port $PORT"
    healthCheckPath: /health
    envVars:

      - key: GHOST_API_TOKEN


        sync: false

      - key: POLYGON_API_KEY


        sync: false

      - key: ALPHAVANTAGE_API_KEY


        sync: false

      - key: GHOST_FOCUS_TICKER


        value: WOLF

      - key: PYTHON_VERSION


        value: "3.12"

```text

### Step 2: Deploy to Render

1. Go to [render.com](<<<<<https://render.co>>>>>m)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render auto-detects Python
5. Set environment variables in dashboard


### Free Tier Limitations

- Spins down after 15 minutes of inactivity
- 750 hours/month free (enough for 24/7 on one service)
- Use paid tier ($7/mo) for always-on


---

## 💧 Option 3: DigitalOcean App Platform

**Pros**: Simple, scalable, $6/month starter plan
**Cons**: Requires credit card
**Setup Time**: 30 minutes

### Step 1: Create App Spec

```yaml

name: ghost-trading
services:

- name: web


  github:
    repo: your-username/GHOST
    branch: main
  build_command: pip install -r requirements.txt
  run_command: uvicorn wolf_app:app --host 0.0.0.0 --port 8080
  http_port: 8080
  health_check:
    http_path: /health
  instance_count: 1
  instance_size_slug: basic-xxs
  envs:

  - key: GHOST_API_TOKEN


    scope: RUN_TIME
    type: SECRET

  - key: POLYGON_API_KEY


    scope: RUN_TIME
    type: SECRET

```text

### Step 2: Deploy

1. Go to [cloud.digitalocean.com/apps](<<<<<https://cloud.digitalocean.com/app>>>>>s)
2. Click "Create App"
3. Choose GitHub source
4. Select GHOST repository
5. Configure resources and environment variables
6. Deploy


**Cost**: $6/month (Basic plan with 512MB RAM)

---

## 🖥️ Option 4: VPS with Docker (Most Flexible)

**Pros**: Full control, cheaper long-term, any provider
**Cons**: More setup, you manage infrastructure
**Setup Time**: 60 minutes

### Recommended VPS Providers

- **DigitalOcean**: $6/month droplet
- **Linode**: $5/month nanode
- **Vultr**: $6/month instance
- **Hetzner**: €4.51/month CX11


### Step 1: Setup VPS

```bash

# SSH into your VPS

ssh root@your-vps-ip

# Update system

apt update && apt upgrade -y

# Install Docker

curl -fsSL <<<<<https://get.docker.com>>>>> -o get-docker.sh
sh get-docker.sh

# Install Docker Compose

apt install docker-compose -y

```text

### Step 2: Clone Repository

```bash

cd /opt
git clone <<<<<https://github.com/your-username/GHOST.git>>>>>
cd GHOST

```text

### Step 3: Create Production Environment File

```bash

cat > .env.production << 'EOF'
GHOST_API_TOKEN=<Railway:GHOST_API_TOKEN>
POLYGON_API_KEY=<Railway:POLYGON_API_KEY>
ALPHAVANTAGE_API_KEY=<Railway:ALPHAVANTAGE_API_KEY>
GHOST_FOCUS_TICKER=WOLF
TELEGRAM_BOT_TOKEN=<Railway:TELEGRAM_BOT_TOKEN>
TELEGRAM_CHAT_ID=<Railway:TELEGRAM_CHAT_ID>
WOLF_PERSIST_MODE=sqlite
WOLF_SQLITE_PATH=/app/data/wolf.db
EOF

```text

### Step 4: Create docker-compose.yml

```yaml

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
      test: ["CMD", "curl", "-f", "<<<<<http://localhost:5000/health"]>>>>>
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

```text

### Step 5: Deploy with Docker

```bash

# Build and start

docker-compose up -d

# View logs

docker-compose logs -f

# Check status

docker-compose ps

# Verify health

curl <<<<<http://localhost/health>>>>>

```text

### Step 6: Setup Automatic Updates

```bash

# Create update script

cat > /opt/GHOST/update.sh << 'EOF'

#!/bin/bash

cd /opt/GHOST
git pull origin main
docker-compose down
docker-compose up -d --build
EOF

chmod +x /opt/GHOST/update.sh

# Add to cron for daily updates (optional)

echo "0 4 ***/opt/GHOST/update.sh >> /var/log/ghost-update.log 2>&1" | crontab -

```text

### Step 7: Setup Nginx Reverse Proxy (Optional)

```bash

# Install Nginx

apt install nginx -y

# Create Nginx config

cat > /etc/nginx/sites-available/ghost << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass <<<<<http://localhost:5000;>>>>>
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable site

ln -s /etc/nginx/sites-available/ghost /etc/nginx/sites-enabled/
nginx -t
systemctl reload nginx

```text

### Step 8: Setup SSL with Let's Encrypt (Optional)

```bash

apt install certbot python3-certbot-nginx -y
certbot --nginx -d your-domain.com

```text

---

## ☁️ Option 5: AWS EC2 (Enterprise)**Pros**: Scalable, AWS ecosystem, enterprise features

**Cons**: More complex, slightly higher cost
**Setup Time**: 45 minutes

### Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2
2. Click "Launch Instance"
3. Choose Ubuntu 22.04 LTS
4. Select t3.micro ($10/month) or t3.small ($20/month)
5. Configure security group:
   - Allow HTTP (80)
   - Allow HTTPS (443)
   - Allow SSH (22) from your IP


### Step 2: Connect and Setup

```bash

# SSH into instance

ssh -i your-key.pem ubuntu@your-ec2-ip

# Follow VPS setup steps above (Docker method)

```text

### Step 3: Use RDS for Database (Optional)

Instead of SQLite, use AWS RDS PostgreSQL:

1. Create RDS PostgreSQL instance (db.t3.micro)
2. Update environment variables:


   ```bash

   DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/ghost

   ```text

### Step 4: Setup Auto-Scaling (Optional)

Use AWS ECS/Fargate for automatic scaling based on traffic.

---

## 🔒 Security Best Practices

### 1. Environment Variables

**Never commit secrets to git**. Use environment variables or secrets management:

```bash

# .env.production (add to .gitignore)

GHOST_API_TOKEN=$(openssl rand -hex 32)
POLYGON_API_KEY=<Railway:POLYGON_API_KEY>
ALPHAVANTAGE_API_KEY=<Railway:ALPHAVANTAGE_API_KEY>

```text

### 2. Firewall Configuration

```bash

# UFW (Ubuntu)

ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw enable

```text

### 3. Regular Updates

```bash

# System updates

apt update && apt upgrade -y

# Application updates

cd /opt/GHOST
git pull origin main
docker-compose up -d --build

```text

### 4. Backups

```bash

# Backup script

cat > /opt/backup-ghost.sh << 'EOF'

#!/bin/bash

BACKUP_DIR=/opt/backups
DATE=$(date +%Y%m%d_%H%M%S)

# Backup databases

cp /opt/GHOST/data/wolf.db $BACKUP_DIR/wolf_$DATE.db
cp /opt/GHOST/data/ai_memory.db $BACKUP_DIR/ai_memory_$DATE.db

# Keep only last 7 days

find $BACKUP_DIR -name "*.db" -mtime +7 -delete
EOF

chmod +x /opt/backup-ghost.sh

# Run daily at 3 AM

echo "0 3 ***/opt/backup-ghost.sh" | crontab -

```text

---

## 📊 Monitoring & Logging

### 1. Health Check Endpoint

Ghost provides `/health` and `/health/detailed` endpoints:

```bash

# Simple check

curl <<<<<https://your-domain.com/health>>>>>

# Detailed diagnostics

curl <<<<<https://your-domain.com/health/detailed>>>>> | jq '.'

```text

### 2. Uptime Monitoring

Use external services:

-**UptimeRobot**(free): Check every 5 minutes
-**Pingdom**: Advanced monitoring

- **Better Uptime**: Modern alternative


### 3. Log Management

```bash

# View Docker logs

docker-compose logs -f --tail=100

# View specific service

docker-compose logs -f ghost

# Search logs

docker-compose logs ghost | grep ERROR

```text

### 4. Prometheus Metrics (Optional)

Ghost includes Prometheus metrics at `/metrics`:

```bash

# Check metrics

curl <<<<<https://your-domain.com/metrics>>>>>

```text

Set up Grafana for visualization.

---

## 🔄 Deployment Workflow

### Recommended CI/CD with GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml

name: Deploy Ghost

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:

      - uses: actions/checkout@v3

      - name: Deploy to Railway


        run: |
          curl -X POST ${{ secrets.RAILWAY_WEBHOOK_URL }}

```text

---

## 💰 Cost Comparison

| Option | Monthly Cost | Setup | Maintenance |
|--------|-------------|-------|-------------|
| Railway | $5-20 | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐⭐ Automatic |
| Render Free | $0 | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Automatic |
| Render Paid | $7 | ⭐⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐⭐ Automatic |
| DigitalOcean | $6 | ⭐⭐⭐⭐ Moderate | ⭐⭐⭐⭐ Low |
| VPS + Docker | $5-10 | ⭐⭐⭐ Moderate | ⭐⭐⭐ Manual |
| AWS EC2 | $10-20 | ⭐⭐ Complex | ⭐⭐ Manual |

---

## 🎯 Recommendation by Use Case

### Personal/Testing

→ **Render Free Tier**or**Railway Hobby**($5)

### Small Production

→**Railway Pro**($10) or**DigitalOcean App**($6)

### High Availability

→**DigitalOcean Droplet**($12) + Load Balancer

### Enterprise

→**AWS ECS/Fargate**with RDS and CloudWatch

---

## 🆘 Troubleshooting

### Issue: Container Keeps Restarting

```bash

# Check logs

docker-compose logs ghost

# Common causes

# 1. Missing environment variables

# 2. Port conflicts

# 3. Database connection issues

```text

### Issue: Out of Memory

```bash

# Increase container memory limit

# In docker-compose.yml

services:
  ghost:
    mem_limit: 1g
    mem_reservation: 512m

```text

### Issue: Slow Performance

```bash

# Check resource usage

docker stats

# Consider upgrading instance size

```text

---

## 📞 Support

For deployment issues:

1. Check `/health/detailed` endpoint
2. Review logs: `docker-compose logs -f`
3. Verify environment variables
4. Test database connections


---

## 🚀 Quick Start Commands

```bash

# Railway (fastest)

railway login
railway init
railway up

# Docker on VPS

git clone <<<<<https://github.com/your-username/GHOST.git>>>>>
cd GHOST
cp .env.example .env.production

# Edit .env.production with your secrets

docker-compose up -d

# Verify

curl <<<<<http://localhost/health>>>>>

```text

---**Next Steps**:

1. Choose your deployment option
2. Follow the setup guide
3. Configure monitoring
4. Set up backups
5. Test thoroughly


**Recommended**: Start with Railway for fastest deployment, then migrate to VPS if you need more control.
