# 🚀 Ghost 24/7 Deployment - Quick Start

## Choose Your Deployment Method

### ⚡ Fastest: Railway (15 minutes)

```bash

# Install Railway CLI

npm install -g @railway/cli

# Run deployment script

./scripts/deploy_railway.sh

```text

**Cost**: ~$5-10/month | **Difficulty**: ⭐ Easy

______________________________________________________________________

### 🆓 Free Option: Render

1. Go to [render.com](<<<<<https://render.co>>>>>m)
2. Connect GitHub repo
3. Use `render.yaml` configuration (already included)
4. Add environment variables in dashboard


**Cost**: Free (with limitations) or $7/month | **Difficulty**: ⭐ Easy

______________________________________________________________________

### 🖥️ Full Control: Docker on VPS

```bash

# On your VPS (Ubuntu/Debian)

curl -fsSL <<<<<https://raw.githubusercontent.com/your-username/GHOST/main/scripts/deploy_docker_vps.sh>>>>> | sudo bash

```text

**Cost**: $5-10/month | **Difficulty**: ⭐⭐ Moderate

**Manual Setup**:

```bash

# 1. Setup VPS

ssh root@your-vps-ip
apt update && apt install -y docker docker-compose git

# 2. Clone repo

cd /opt
git clone <<<<<https://github.com/your-username/GHOST.git>>>>>
cd GHOST

# 3. Configure environment

cp .env.example .env.production
nano .env.production  # Add your secrets

# 4. Deploy

docker-compose up -d

# 5. Verify

curl <<<<<http://localhost/health>>>>>

```text

______________________________________________________________________

## Required Environment Variables

```bash

# Copy these from Railway → tender-benevolence / ghost-protocol / Variables

GHOST_API_TOKEN=<Railway:GHOST_API_TOKEN>
POLYGON_API_KEY=<Railway:POLYGON_API_KEY>
ALPHAVANTAGE_API_KEY=<Railway:ALPHAVANTAGE_API_KEY>
GHOST_FOCUS_TICKER=WOLF  # Optional (default: WOLF)
TELEGRAM_BOT_TOKEN=<Railway:TELEGRAM_BOT_TOKEN>
TELEGRAM_CHAT_ID=<Railway:TELEGRAM_CHAT_ID>

```text

______________________________________________________________________

## After Deployment

### Verify Health

```bash

# Basic health

curl <<<<<https://your-domain.com/health>>>>>

# Detailed diagnostics

curl <<<<<https://your-domain.com/health/detailed>>>>> | jq '.'

# Portfolio status

curl <<<<<https://your-domain.com/api/cockpit>>>>> | jq '{kpis, portfolio}'

```text

### Setup Monitoring

```bash

# Install monitor script (runs in background)

./scripts/monitor_ghost.sh &

# Or add to crontab for automatic monitoring

crontab -e

# Add: */5 *** * /opt/GHOST/scripts/monitor_ghost.sh

```text

### Setup Backups

```bash

# Run backup script

./scripts/backup_ghost.sh

# Schedule daily backups at 3 AM

crontab -e

# Add: 0 3 ***/opt/GHOST/scripts/backup_ghost.sh

```text

### Setup SSL (if using VPS)

```bash

./scripts/setup_ssl.sh

```text

______________________________________________________________________

## Deployment Files

| File | Purpose | |------|---------| | `railway.toml` | Railway configuration | |
`render.yaml` | Render configuration | | `docker-compose.yml` | Docker Compose
configuration | | `scripts/deploy_railway.sh` | Railway deployment automation | |
`scripts/deploy_docker_vps.sh` | VPS Docker deployment | | `scripts/setup_ssl.sh` | SSL
certificate setup | | `scripts/monitor_ghost.sh` | Health monitoring | |
`scripts/backup_ghost.sh` | Database backup | | `scripts/restore_ghost.sh` | Restore
from backup |

______________________________________________________________________

## Cost Comparison

| Platform | Monthly Cost | Setup Time | Always On |
|----------|-------------|------------|-----------| | Railway | $5-10 | 15 min | ✅ Yes |
| Render Free | $0 | 20 min | ⚠️ Spins down | | Render Paid | $7 | 20 min | ✅ Yes | |
DigitalOcean | $6 | 30 min | ✅ Yes | | Linode | $5 | 30 min | ✅ Yes | | AWS EC2 | $10-20
| 45 min | ✅ Yes |

______________________________________________________________________

## Troubleshooting

### Container won't start

```bash

# Check logs

docker-compose logs ghost

# Common fixes

# 1. Verify environment variables

cat .env.production

# 2. Check port conflicts

lsof -i :5000

# 3. Rebuild

docker-compose down
docker-compose up -d --build

```text

### Health check fails

```bash

# Check detailed health

curl <<<<<http://localhost:5000/health/detailed>>>>> | jq '.'

# View recent logs

docker-compose logs --tail=100 ghost

# Restart

docker-compose restart ghost

```text

### Out of memory

```bash

# Check usage

docker stats

# Increase memory limit in docker-compose.yml

services:
  ghost:
    mem_limit: 1g
    mem_reservation: 512m

```text

______________________________________________________________________

## Recommended: Railway (Easiest)**Why Railway?**- ✅ One-click deployment

- ✅ Automatic SSL
- ✅ Zero-downtime deploys
- ✅ Built-in monitoring
- ✅ Easy scaling
- ✅ GitHub integration**Deploy in 3 steps:**```bash


npm install -g @railway/cli
railway login
./scripts/deploy_railway.sh

```text

______________________________________________________________________

## Support

📖**Full Documentation**: `docs/24_7_DEPLOYMENT.md`

🔧 **Scripts Directory**: `scripts/`

❓ **Issues**: Check `/health/detailed` endpoint first

______________________________________________________________________

## Quick Commands

```bash

# Health check

curl <<<<<http://localhost:5000/health>>>>>

# View logs

docker-compose logs -f ghost

# Restart

docker-compose restart ghost

# Update

git pull origin main
docker-compose up -d --build

# Backup

./scripts/backup_ghost.sh

# Restore

./scripts/restore_ghost.sh

```text

______________________________________________________________________

**🎉 That's it! Your Ghost trading system will now run 24/7.**

For detailed instructions, see `docs/24_7_DEPLOYMENT.md`
