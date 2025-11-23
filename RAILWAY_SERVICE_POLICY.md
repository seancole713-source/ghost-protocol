# Railway Service Policy

Ghost Protocol runs as a single Railway service (`ghost-protocol`). This document
defines the hard requirements for deployment, observability, and rollback.

## 1. Single UI + health contract
- `/health` must answer within <100 ms from cold start (FastAPI route in `wolf_app.py`).
- `/cockpit` serves **Cockpit V3** only; fallback toggles such as `USE_NEW_COCKPIT`
  are removed. `/api/v3/cockpit/version` returns `{"ui":"cockpit_v3","status":"live"}`.
- Legacy bundles remain accessible via `/ui` for archival viewing but are not part of
  the production health signal. See `docs/UI_BASELINE_CONTRACT.md` for the frozen
  layout contract.

## 2. Verification steps
1. **Before deploy**: run `scripts/check_no_placeholders.sh` locally. This ensures the
   codebase is placeholder-free and the smoke endpoints respond.
2. **After deploy**: run `scripts/check_railway_service.sh https://ghost-protocol-production.up.railway.app`
   (or set `RAILWAY_URL`). The script immediately fails on any non-2xx response and
   enforces the cockpit version JSON.
3. **CI**: `.github/workflows/ghost_smoke.yml` mirrors these steps on every push/PR.

## 3. Environment + secrets
- All secrets live in Railway variables; never commit `.env` files.
- Missing secrets are treated as deployment blockers—the smoke script will surface the
  failing endpoint instead of substituting placeholders.
- Any new secret must be documented here and in `README.md` before use.

## 4. Incident / rollback
- If `scripts/check_railway_service.sh` fails after a deploy, immediately roll back via
  `railway rollback` or redeploy the previous green commit.
- Keep `/tmp/ghost-ci.log` artifacts from CI failures for analysis.
- Document every incident in `DEPLOYMENT_STATUS.md` with the failing endpoint and fix.

## 5. Ownership
- The enforcement scripts + hook definitions live in `scripts/` and `.githooks/`.
- Any change to health, cockpit routing, or smoke coverage requires updating this file
  plus `GHOST_NO_PLACEHOLDER_ENFORCEMENT.md`.
