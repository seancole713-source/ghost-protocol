╔══════════════════════════════════════════════════════════════╗ ║ GHOST → RAILWAY
DEPLOYMENT - STEP BY STEP ║
╚══════════════════════════════════════════════════════════════╝

📍 YOU ARE HERE: Railway account ready 🎯 GOAL: Deploy Ghost to run 24/7

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: COMMIT YOUR CODE TO GITHUB ✅

First, let's make sure all deployment files are in your repo:

```bash

# Check what files are new/modified

git status

# Add all deployment files

git add railway.toml render.yaml scripts/ docs/24_7_DEPLOYMENT.md DEPLOY_24_7.md

# Commit

git commit -m "Add Railway deployment configuration"

# Push to GitHub

git push origin main

```text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 2: INSTALL RAILWAY CLI

```bash

npm install -g @railway/cli

# Verify installation

railway --version

```text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 3: LOGIN TO RAILWAY

```bash

railway login

```text

This will open your browser to authenticate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 4: CREATE NEW PROJECT

```bash

# Initialize Railway project in your GHOST directory

cd /workspaces/GHOST
railway init

# Railway will ask

#   "Create new project or link existing?"

#   → Choose "Create new project"

#   "Project name?"

#   → Type: ghost-trading

```text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 5: SYNC ENVIRONMENT VARIABLES

All canonical secrets already live inside Railway → **tender-benevolence / ghost-protocol / Variables**.
Do **not** paste placeholder strings into the repo—pull the real values from that screen (or via CLI) and
reapply them to the new service:

```bash

# Export the production env to JSON (safe locally, never commit)

railway variables --service ghost-protocol --environment production --json > prod-env.json

# Reapply only the fields you actually need (example)

railway variables set POLYGON_API_KEY "$(jq -r '.POLYGON_API_KEY' prod-env.json)"
railway variables set ALPHAVANTAGE_API_KEY "$(jq -r '.ALPHAVANTAGE_API_KEY' prod-env.json)"
railway variables set GHOST_API_TOKEN "$(jq -r '.GHOST_API_TOKEN' prod-env.json)"

```text

Use the dashboard if you prefer a GUI: Settings → Variables → “Copy value”. Never invent stand-in strings;
the zero-placeholder gate will block deploys if those appear anywhere in the repo.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 6: DEPLOY TO RAILWAY 🚀

```bash

# Deploy your code

railway up

# Railway will

#   ✅ Detect Python application

#   ✅ Install dependencies from requirements.txt

#   ✅ Use railway.toml configuration

#   ✅ Start uvicorn server

#   ✅ Provide a public URL

```text

This takes 2-5 minutes...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 7: GET YOUR DEPLOYMENT URL

```bash

# Get your live URL

railway domain

# Example output

# ghost-trading-production.up.railway.app

```text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 8: VERIFY DEPLOYMENT ✅

Test your live Ghost instance:

```bash

# Set your Railway URL (from previous step)

GHOST_URL="<<<<<https://ghost-trading-production.up.railway.app">>>>>

# Test basic health

curl $GHOST_URL/health

# Expected: {"ok":true,"ts":...}

# Test detailed health

curl $GHOST_URL/health/detailed | jq '.ok, .issues'

# Test portfolio

curl $GHOST_URL/api/cockpit | jq '.kpis'

```text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 9: RESTORE YOUR POSITION DATA

Your portfolio data is in Codespace. Let's move it to Railway:

```bash

# Get current position data

curl -H "Authorization: Bearer supersecret123jamaica713" \
  $GHOST_URL/api/position \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"qty": 8.41959051, "avg_cost": 359.28}'

# Verify

curl $GHOST_URL/api/cockpit | jq '.portfolio'

```text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 10: SETUP CUSTOM DOMAIN (OPTIONAL)

In Railway Dashboard:

1. Go to your project
2. Click "Settings" → "Domains"
3. Click "Generate Domain" or add custom domain
4. Railway handles SSL automatically ✅


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ YOU'RE DONE! Ghost is now running 24/7

Your Ghost instance: • Runs 24/7 independently • Auto-deploys on git push • Has
automatic SSL/HTTPS • Restarts automatically on failures • Costs ~$5-10/month

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USEFUL RAILWAY COMMANDS

View logs: railway logs

Open dashboard: railway open

Check status: railway status

Redeploy: git push origin main # Auto-deploys!

# OR manually

railway up

Add database (optional): railway add # Choose PostgreSQL

View environment variables: railway variables

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 QUICK START SCRIPT

Or run this all-in-one script:

```bash

./scripts/deploy_railway.sh

```text

This script will:

1. ✅ Check Railway CLI
2. ✅ Login to Railway
3. ✅ Create project
4. ✅ Set environment variables (prompts you)
5. ✅ Deploy Ghost
6. ✅ Show your live URL


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MONITORING YOUR DEPLOYMENT

Railway Dashboard: <<<<<https://railway.app/dashboard>>>>>

View Metrics: • CPU usage • Memory usage • Request count • Response times

Ghost Health Endpoint: <<<<<https://your-app.railway.app/health/detailed>>>>>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 TROUBLESHOOTING

Issue: "Build failed" Fix: Check railway logs railway logs

Issue: "Health check failed" Fix: Verify environment variables are set railway variables

Issue: "502 Bad Gateway" Fix: Wait 30 seconds for startup, then check logs

Issue: "Database not persisting" Fix: Railway uses ephemeral filesystem Consider adding
PostgreSQL: railway add # Choose PostgreSQL Update DATABASE_URL in code

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 COST ESTIMATE

Railway Pricing: • Hobby Plan: $5/month (500 hours) • Pro Plan: $20/month (unlimited) •
Pay only for what you use

Ghost typically uses: • ~0.5 vCPU • ~512 MB RAM • Estimated: $5-10/month

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎉 SUCCESS CHECKLIST

Once deployed, verify:

□ Health endpoint returns OK curl <<<<<https://your-app.railway.app/health>>>>>

□ Portfolio loads correctly curl <<<<<https://your-app.railway.app/api/cockpit>>>>> | jq
'.portfolio'

□ AI Memory accessible curl <<<<<https://your-app.railway.app/ai/memory/stats>>>>>

□ Price fetching works curl <<<<<https://your-app.railway.app/api/price/WOLF>>>>>

□ Ghost accessible from anywhere (not just Codespace!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📞 NEED HELP?

Railway Docs: <<<<<https://docs.railway.app>>>>> Ghost Health:
<<<<<https://your-app.railway.app/health/detailed>>>>> Railway Discord: <<<<<https://discord.gg/railway>>>>>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ready to deploy? Start with Step 1! 🚀
