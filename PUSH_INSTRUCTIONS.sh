#!/bin/bash
# ====================================================
# DEPLOYMENT STATUS - READY TO PUSH
# ====================================================

cat << 'EOF'

✅ REPOSITORY CLEANED AND COMMITTED
====================================================

Latest commits ready to deploy:
  77e7175 (HEAD) - deploy: middleware fix, live routes, scanners; add deploy/verify tools; ignore caches/runtime data
  2d772ed - chore(api): simplify request middleware to always return JSON 500 on error
  95bdee8 - fix(middleware): robust error handling in _log_requests middleware

Changes included:
  ✅ Middleware simplified (70+ lines → 14 lines, eliminates 499s)
  ✅ .gitignore added (__pycache__, *.db, runtime JSON)
  ✅ 88 __pycache__ files removed from tracking
  ✅ 26 data/*.db files removed from tracking
  ✅ Deployment scripts added (smoke tests, diagnostics)
  ✅ Documentation added (verification steps, quick commands)

Working tree status: CLEAN
Branch: master
Remotes: Not configured in dev container

====================================================
MANUAL ACTIONS REQUIRED (FROM HOST MACHINE)
====================================================

Since git remotes aren't configured in this dev container, 
you need to push from your host machine where Railway/GitHub 
remotes are set up.

STEP 1: Push from host machine
----------------------------------------------------
# On your host machine:
cd /path/to/ghost-cockpit

# Pull the commits from container (if container is git-synced)
# OR copy the commits from container to host

git log --oneline -3  # Verify you have commits 77e7175, 2d772ed, 95bdee8

# Push to your default remote (GitHub, GitLab, etc.)
git push origin master

# If Railway has a separate remote:
git push railway master

ALTERNATIVE: If you need to copy commits from container
----------------------------------------------------
# From container, create a patch:
git format-patch -3 HEAD -o /tmp/patches

# Copy patches to host and apply:
git am /tmp/patches/*.patch
git push origin master


STEP 2: Redeploy on Railway
----------------------------------------------------
Option A: Automatic (if Railway watches your repo)
  - Railway will auto-deploy after git push

Option B: Manual redeploy
  1. Open Railway Dashboard
  2. Go to your Ghost Cockpit service
  3. Click "Deployments" tab
  4. Find latest deployment (should show commit 77e7175)
  5. Click "⋯" menu → "Redeploy"
  6. Wait for "Healthcheck succeeded!" (~2-3 minutes)


STEP 3: Post-Deployment Validation
----------------------------------------------------
# Set environment variables
export GHOST_BASE_URL="https://ghost-sniper-bot-seancole713-production.up.railway.app"
export GHOST_API_TOKEN="edaa4eac-6455-4693-a745-142cb6deef03"

# Run automated smoke tests
bash /app/deployment_smoke_test.sh

# OR run manual tests
bash /app/DEPLOYMENT_QUICK_COMMANDS.sh


STEP 4: Verify SSE Stream (Browser)
----------------------------------------------------
Open in browser:
https://ghost-sniper-bot-seancole713-production.up.railway.app/api/cockpit/stream

Expected:
  event: status
  event: ping (every 30s)
  event: snapshot


STEP 5: Monitor Railway Logs (5 minutes)
----------------------------------------------------
Railway Dashboard → Logs → HTTP Logs

Watch for:
  ❌ 0 occurrences of HTTP 499 (Client Closed Request)
  ❌ 0 occurrences of HTTP 502 (Bad Gateway)
  ✅ Any 500s should be JSON: {"error": "internal_error"}
  ✅ All routes return 200 OK


====================================================
EXPECTED TEST RESULTS
====================================================

After deployment, all endpoints should work:

✅ /api/status              → 200 OK, {"active": true}
✅ /api/tick                → Counter increments on successive calls
✅ /api/regime/current      → 200 OK (not 404)
✅ /api/price/diagnostics   → AAPL returns AAPL price (not WOLF)
✅ /api/cache/purge         → {"ok": true, "purged_count": N}
✅ /api/scan/health         → 200 OK, movers scanner health
✅ /api/scan/movers         → 200 OK, active movers data
✅ /api/cockpit/stream      → SSE events flowing continuously
✅ OpenAPI schema           → Contains all new routes

Success criteria:
  • 0 × 499 errors in 5-minute window
  • 0 × 502 errors in 5-minute window
  • Average response time < 3000ms
  • All tested endpoints return 200


====================================================
ROLLBACK PLAN (IF NEEDED)
====================================================

If deployment causes issues:

Railway Dashboard:
  Deployments → Select previous stable → Click "Redeploy"

OR via Railway CLI:
  railway rollback


====================================================
TROUBLESHOOTING
====================================================

If 404 errors persist:
  → Check OpenAPI schema for missing routes
  → Verify middleware isn't blocking route registration
  → Review Railway deployment logs for startup errors

If 499 errors persist:
  → Verify middleware changes deployed (commit 2d772ed)
  → Check for timeout issues in upstream services
  → Review exception handling in background tasks

If timeouts (10s+) persist:
  → Set PRICE_PROVIDER_TIMEOUT_S=1.0 in Railway env vars
  → Check provider health in diagnostics endpoint
  → Review httpx/requests client timeout configuration

If SSE stream fails:
  → Check browser console for connection errors
  → Verify CORS headers in Railway logs
  → Test with curl: curl -N $GHOST_BASE_URL/api/cockpit/stream


====================================================
FILES READY FOR DEPLOYMENT
====================================================

Code Changes:
  • wolf_app.py (middleware simplified)
  • .gitignore (caches and runtime data excluded)

Deployment Tools:
  • deployment_smoke_test.sh (automated validation)
  • backend_diagnostic.sh (comprehensive tests)
  • generate_status_report.py (JSON metrics)
  • quick_validate.sh (fast smoke test)
  • restart_and_validate.sh (full restart sequence)
  • DEPLOYMENT_QUICK_COMMANDS.sh (command reference)

Documentation:
  • DEPLOYMENT_VERIFICATION_STEPS.md (full guide)
  • BACKEND_STABILIZATION_COMPLETE.md (implementation details)
  • BACKEND_STABILIZATION_SUMMARY.json (status overview)
  • API_ROUTING_FIXES_COMPLETE.md (routing fixes)
  • RAILWAY_ENV_OPTIMIZATIONS.txt (env var settings)


====================================================
NEXT ACTION
====================================================

🚀 Push commits 77e7175, 2d772ed, 95bdee8 from host machine
   Then redeploy on Railway and run smoke tests.

EOF
