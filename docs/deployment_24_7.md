# Ghost 24/7 Deployment Guide

**Keep Ghost Running When Codespace Closes**---

## Problem

GitHub Codespaces stop all processes when you close the browser or the codespace shuts down. Ghost needs to run
continuously for live trading/monitoring.

---

## Solution Options

### ⭐ Option 1: Deploy to Cloud VM (Recommended)**Best for**: Production use, reliability, cost-effective

#### A. AWS EC2 (Free Tier Available)

```bash

# 1. Launch EC2 instance (t2.micro free tier)

# 2. Connect via SSH

ssh -i your-key.pem ubuntu@your-ec2-ip

# 3. Install dependencies

sudo apt update
sudo apt install -y python3 python3-pip git

# 4. Clone Ghost

git clone <<<<<https://github.com/seancole713-source/GHOST.git>>>>>
cd GHOST

# 5. Set up environment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 6. Configure secrets

cat > secrets.env << 'EOF'

# Copy the production values from Railway → Variables

GHOST_API_TOKEN=<Railway:GHOST_API_TOKEN>
POLYGON_API_KEY=<Railway:POLYGON_API_KEY>
ALPHAVANTAGE_API_KEY=<Railway:ALPHAVANTAGE_API_KEY>
TELEGRAM_BOT_TOKEN=<Railway:TELEGRAM_BOT_TOKEN>
TELEGRAM_CHAT_ID=<Railway:TELEGRAM_CHAT_ID>
EOF

# 7. Run with systemd (auto-restart on crash)

sudo tee /etc/systemd/system/ghost.service > /dev/null << 'EOF'
[Unit]
Description=Ghost Trading System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/GHOST
Environment="PATH=/home/ubuntu/GHOST/.venv/bin"
Environment="PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom"
ExecStartPre=/bin/mkdir -p /tmp/ghost_prom
ExecStart=/home/ubuntu/GHOST/.venv/bin/uvicorn wolf_app:app --host 0.0.0.0 --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 8. Enable and start

sudo systemctl daemon-reload
sudo systemctl enable ghost
sudo systemctl start ghost

# 9. Check status

sudo systemctl status ghost

```text

**Cost**: ~$3-10/month (t2.micro free for 12 months)

---

#### B. DigitalOcean Droplet

```bash

# Similar to AWS but simpler interface

# 1. Create $6/month droplet (Ubuntu 22.04)

# 2. Follow same steps as AWS

# 3. DigitalOcean has better docs for beginners

```text

**Cost**: $6/month basic droplet

---

#### C. Google Cloud (GCP)

```bash

# 1. Create e2-micro instance (always free tier)

# 2. Follow AWS steps above

# 3. Use GCP's firewall rules for port 5000

```text

**Cost**: Free tier available (e2-micro)

---

### Option 2: Docker + Cloud Container Service

**Best for**: Scalability, easy deployment

#### A. Create Dockerfile (already exists, enhance it)

```dockerfile

# Dockerfile

FROM python:3.12-slim

WORKDIR /app

# Install dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application

COPY . .

# Create prometheus directory

RUN mkdir -p /tmp/ghost_prom

# Expose port

EXPOSE 5000

# Environment variables

ENV PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
ENV PYTHONUNBUFFERED=1

# Health check

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('<<<<<http://localhost:5000/health>>>>>').raise_for_status()"

# Run

CMD ["uvicorn", "wolf_app:app", "--host", "0.0.0.0", "--port", "5000"]

```text

#### B. Deploy to Cloud Run (Google)

```bash

# 1. Build and push to Google Container Registry

GCP_PROJECT_ID="$(gcloud config get-value project)"
gcloud builds submit --tag "gcr.io/${GCP_PROJECT_ID}/ghost"

# 2. Deploy to Cloud Run

export GHOST_API_TOKEN=$(openssl rand -hex 32)
export POLYGON_API_KEY=$(railway variables get POLYGON_API_KEY)
export ALPHAVANTAGE_API_KEY=$(railway variables get ALPHAVANTAGE_API_KEY)

gcloud run deploy ghost \
  --image "gcr.io/${GCP_PROJECT_ID}/ghost" \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GHOST_API_TOKEN=$GHOST_API_TOKEN,POLYGON_API_KEY=$POLYGON_API_KEY,ALPHAVANTAGE_API_KEY=$ALPHAVANTAGE_API_KEY \
  --min-instances 1

```text

**Cost**: ~$5-15/month (with min-instances=1 for 24/7)

---

#### C. Deploy to AWS ECS/Fargate

```bash

# 1. Push to ECR

aws ecr create-repository --repository-name ghost
docker build -t ghost .
AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
AWS_REGION="us-east-1"
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/ghost"
docker tag ghost:latest "$ECR_URL:latest"
docker push "$ECR_URL:latest"

# 2. Create ECS task definition and service

# (Use AWS Console or Terraform)

```text

**Cost**: ~$10-20/month

---

### Option 3: Railway.app (Easiest)

**Best for**: Quick deployment, no DevOps

```bash

# 1. Install Railway CLI

npm i -g @railway/cli

# 2. Login

railway login

# 3. Initialize project

railway init

# 4. Set environment variables

railway variables set GHOST_API_TOKEN "$(openssl rand -hex 32)"
railway variables set POLYGON_API_KEY "$(railway variables get POLYGON_API_KEY --environment production)"
railway variables set ALPHAVANTAGE_API_KEY "$(railway variables get ALPHAVANTAGE_API_KEY --environment production)"

# 5. Deploy

railway up

```text

**Advantages**:

- ✅ Zero config deployment
- ✅ Auto SSL/HTTPS
- ✅ Automatic restarts
- ✅ Built-in logs/metrics
- ✅ Free tier available ($5 credit/month)


**Cost**: $5/month after free tier

**Railway Dashboard**: <<<<<https://railway.app>>>>>

---

### Option 4: Render.com (Similar to Railway)

**Best for**: Simple deployment with PostgreSQL

1. Go to <<<<<https://render.com>>>>>
2. Connect GitHub repo
3. Create "Web Service"
4. Set environment variables in dashboard
5. Click "Deploy"


**Cost**: $7/month for always-on service

---

### Option 5: Fly.io (Global Edge)

**Best for**: Low latency worldwide

```bash

# 1. Install flyctl

curl -L <<<<<https://fly.io/install.sh>>>>> | sh

# 2. Login

flyctl auth login

# 3. Create app

flyctl launch

# 4. Set secrets

flyctl secrets set GHOST_API_TOKEN="$(openssl rand -hex 32)"
flyctl secrets set POLYGON_API_KEY="$(railway variables get POLYGON_API_KEY)"

# 5. Deploy

flyctl deploy

```text

**Cost**: ~$3-10/month

---

### Option 6: VPS (Most Control)

**Best for**: Full control, custom setup

Popular providers:

- **Linode**: $5/month
- **Vultr**: $6/month
- **Hetzner**: €4/month (~$4.50)
- **OVH**: €3.50/month


Setup same as AWS EC2 above.

---

## Comparison Table

| Option | Cost/Month | Ease | Reliability | Control | Best For |
|--------|-----------|------|-------------|---------|----------|
| **Railway**| $5-10 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Quick start |
|**Render**| $7 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Simple deployment |
|**Fly.io**| $3-10 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Global edge |
|**AWS EC2**| $3-10 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Production |
|**GCP Cloud Run**| $5-15 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Auto-scale |
|**Linode/Vultr VPS**| $5-6 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Full control |

---

## My Recommendation for You

### 🏆**Start with Railway.app**(Fastest Setup)**Why**

1. ✅ Deploy in 5 minutes
2. ✅ No server management
3. ✅ Automatic SSL
4. ✅ Free $5/month credit
5. ✅ Easy rollbacks
6. ✅ Built-in monitoring


**Setup Steps**:

```bash

# 1. Install Railway CLI

npm i -g @railway/cli

# 2. In your GHOST directory

railway login
railway init

# 3. Set environment variables

railway variables set GHOST_API_TOKEN=supersecret123jamaica713
railway variables set POLYGON_API_KEY="$(railway variables get POLYGON_API_KEY)"
railway variables set ALPHAVANTAGE_API_KEY="$(railway variables get ALPHAVANTAGE_API_KEY)"
railway variables set GHOST_FOCUS_TICKER=WOLF
railway variables set WOLF_SQLITE_PATH=/data/wolf.db

# 4. Deploy

railway up

# 5. Get your URL

railway domain

```text

**Your Ghost will be live at**: `https://ghost-production-xxxx.up.railway.app`

---

### 🥈 **Alternative: AWS EC2 Free Tier**(Production-Grade)

If you want more control and don't mind managing a server:

```bash

# Quick setup script for EC2

#!/bin/bash

# Save as deploy.sh and run on fresh EC2 instance

# Update system

sudo apt update && sudo apt upgrade -y

# Install Python

sudo apt install -y python3 python3-pip python3-venv git

# Clone Ghost

cd ~
git clone <<<<<https://github.com/seancole713-source/GHOST.git>>>>>
cd GHOST

# Setup Python environment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create secrets

cat > secrets.env << EOF
GHOST_API_TOKEN=supersecret123jamaica713
POLYGON_API_KEY=$(railway variables get POLYGON_API_KEY)
ALPHAVANTAGE_API_KEY=$(railway variables get ALPHAVANTAGE_API_KEY)
EOF

# Create systemd service

sudo tee /etc/systemd/system/ghost.service > /dev/null << 'SVCEOF'
[Unit]
Description=Ghost Trading System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/GHOST
Environment="PATH=/home/$USER/GHOST/.venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom"
EnvironmentFile=/home/$USER/GHOST/secrets.env
ExecStartPre=/bin/mkdir -p /tmp/ghost_prom
ExecStart=/home/$USER/GHOST/.venv/bin/uvicorn wolf_app:app --host 0.0.0.0 --port 5000
Restart=always
RestartSec=10
StandardOutput=append:/var/log/ghost.log
StandardError=append:/var/log/ghost.log

[Install]
WantedBy=multi-user.target
SVCEOF

# Enable and start

sudo systemctl daemon-reload
sudo systemctl enable ghost
sudo systemctl start ghost

# Check status

sudo systemctl status ghost

echo "Ghost is now running! Access at <<<<<http://$(curl>>>>> -s ifconfig.me):5000"

```text**Free Tier**: 750 hours/month (24/7) for 12 months

---

## Database Persistence

**Important**: When deploying to cloud, ensure databases persist:

### Option A: Use Volume Mounts

```yaml

# docker-compose.yml (for cloud VMs)

version: '3.8'
services:
  ghost:
    build: .
    ports:

      - "5000:5000"


    volumes:

      - ./data:/app/data  # Persist databases


    environment:

      - GHOST_API_TOKEN=${GHOST_API_TOKEN}
      - POLYGON_API_KEY=${POLYGON_API_KEY}


    restart: always

```text

### Option B: Use External Database

```python

# For production, consider PostgreSQL instead of SQLite

# Update wolf_app.py to use PostgreSQL connection

WOLF_SQLITE_PATH = os.getenv("DATABASE_URL", "/data/wolf.db")

# On Railway/Render, add PostgreSQL add-on

# They provide DATABASE_URL automatically

```text

---

## Monitoring & Alerts

### 1. UptimeRobot (Free)

```bash

# Setup monitoring

1. Go to <<<<<https://uptimerobot.com>>>>>
2. Add monitor: <<<<<https://your-ghost-url.com/health>>>>>
3. Get alerts via email/SMS/Telegram if down


```text

### 2. Ghost Built-in Monitoring

```bash

# Already have /health/detailed endpoint

curl <<<<<https://your-ghost-url.com/health/detailed>>>>>

# Set up cron to check health

crontab -e

# Add

*/5 *** * curl -f <<<<<https://your-ghost-url.com/health>>>>> || echo "Ghost down!" | mail -s "Ghost Alert" you@email.com

```text

---

## Security Checklist

When deploying 24/7:

- [ ] Change `GHOST_API_TOKEN` from default
- [ ] Enable HTTPS (Railway/Render do this automatically)
- [ ] Set firewall rules (only allow port 5000)
- [ ] Use environment variables for secrets (never commit)
- [ ] Enable authentication on all write endpoints
- [ ] Set up log rotation
- [ ] Configure backup cron for databases
- [ ] Use strong API keys


---

## Quick Start Command (Railway)

```bash

# One command to deploy Ghost 24/7

npm i -g @railway/cli && \
  railway login && \
  railway init && \
  railway variables set GHOST_API_TOKEN=supersecret123jamaica713 && \
  railway up && \
  railway domain

```text

That's it! Ghost will be running 24/7 at your Railway URL.

---

## Backup Strategy

### Automated Daily Backups

```bash

# On your cloud VM, add to crontab

0 0 *** cd /path/to/GHOST && tar -czf backup-$(date +\%Y\%m\%d).tar.gz data/ && rsync -avz backup-*.tar.gz user@backup-server:/backups/

# Or use cloud storage

0 0 ***cd /path/to/GHOST/data && aws s3 sync . s3://ghost-backups/$(date +\%Y\%m\%d)/

```text

---

## Cost Summary

| Deployment | Monthly Cost | Free Tier |
|------------|-------------|-----------|
| Railway | $5-10 | $5 credit |
| Render | $7 | N/A |
| Fly.io | $3-10 | $5 credit |
| AWS EC2 t2.micro | $0-10 | 12 months |
| GCP e2-micro | $0 | Always free |
| Linode | $5 | $100 credit |
| Vultr | $6 | N/A |
| DigitalOcean | $6 | $200 credit |**My Pick**: Railway ($5/month) for simplicity + AWS EC2 (free) for production

---

## Next Steps

1. **Choose deployment method**(I recommend Railway for quick start)


2.**Deploy Ghost**using commands above
3.**Test health endpoint**: `curl <<<<<https://your-url.com/health/detailed`>>>>>

1. **Set up monitoring**with UptimeRobot


5.**Configure backups**for data/wolf.db and data/ai_memory.db
6.**Migrate to NVDA ticker**for real-time pricing (see GHOST_HEALTH_REPORT.md)


---

## Support

If you need help with deployment:

1. Check logs: `railway logs` or `sudo journalctl -u ghost -f`
2. Test locally first: `uvicorn wolf_app:app --host 0.0.0.0 --port 5000`
3. Verify environment variables: `railway variables` or `systemctl show ghost -p Environment`**Ghost is designed to run 24/7 with automatic restarts and health monitoring!**
