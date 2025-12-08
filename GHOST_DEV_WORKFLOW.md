# Ghost Development Workflow (Zero-Placeholder Edition)

This workflow keeps Ghost Protocol fast, deterministic, and production-ready. Every
change must follow the steps below—no shortcuts, no placeholders.

## 1. Daily dev loop

- **Start Ghost locally**on port `8080` (`uvicorn wolf_app:APP --host 127.0.0.1 --port 8080`).


-**Export real env vars**(`.env` or Railway secrets). Do not invent temporary keys.
-**Keep Cockpit V3 open**at `http://localhost:8080/cockpit` to verify live data.


## 2. Code + tests

- Tackle work in small branches; avoid long-lived drift.
- Prefer FastAPI unit coverage plus targeted scripts in `scripts/`.
- Never comment out logic “for later”; either ship it or remove it.


## 3. Canonical validation (required)

Run the gate**before every commit/push**:

```text
scripts/check_no_placeholders.sh

```text

This script:

1. Scans repo files for disallowed tokens (placeholder patterns listed in the enforcement charter).
2. Hits `/health`, `/cockpit`, and `/api/v3/cockpit/version` on the active target


   (local by default, `RAILWAY_URL` when set) to ensure Cockpit V3 is the sole UI.

  - Override the local target with `LOCAL_GHOST_URL` when running on a non-8080 port.


The script fails fast with concrete paths + line numbers whenever a violation occurs.

## 4. Git hooks & CI

- Run `scripts/install_hooks.sh` once per clone. It points `core.hooksPath` to


  `.githooks/`, so the **pre-push hook**blocks any branch that fails the gate.

- GitHub Actions (`.github/workflows/ghost_smoke.yml`) launches Ghost on the runner,


  executes the same script, and prints server logs if anything regresses.

## 5. Deployment readiness

- Production smoke = `scripts/check_railway_service.sh <<<<<https://ghost-protocol-production.up.railway.app`>>>>>


  (or export `RAILWAY_URL`). This reuses the same smoke function and exits non-zero on
  the first error.

- Railway deployments only proceed when the smoke script passes**and** Cockpit V3 is


  the single rendered UI (see `docs/UI_BASELINE_CONTRACT.md`).

## 6. Documentation contract

- Update this file and `RAILWAY_SERVICE_POLICY.md` whenever workflow steps change.
- Cross-reference `GHOST_NO_PLACEHOLDER_ENFORCEMENT.md` for the enforcement charter.


Stay disciplined: no fake data, no broken smoke gates, no hidden toggles. Every push
must be ready for Railway in one command.
