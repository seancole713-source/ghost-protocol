# GHOST AUTOMATION SYSTEM

Ghost Protocol now ships with a zero-excuse automation layer that blocks pushes, fails CI, and
guards production against placeholder data. Local hooks, reusable smoke scripts, and GitHub
Actions now validate every change the same way on laptops and in continuous integration.

## Components

### `scripts/ghost_smoke.sh`

- Canonical smoke script with `local`, `railway`, and `full` modes.
- Verifies `/health`, `/cockpit`, and the primary V3 API endpoints.
- Confirms the rendered cockpit references `cockpit_v3.css` and `cockpit_v3.js` while rejecting any


   V1/V2 markers.

- Enforces the "no SIM / no placeholder" rule via a repo-wide grep gate before any HTTP calls run.


### `scripts/check_railway_service.sh`

- Lightweight wrapper that reuses the same defaults as the smoke script.
- Runs the production `/health` and `/cockpit` checks with a clear ✅ / ❌ result for on-call usage.


### `.githooks/pre-push` + `scripts/install_hooks.sh`

- `scripts/install_hooks.sh` points `core.hooksPath` to `.githooks/` and ensures the hook is


   executable.

- The pre-push hook runs `bash scripts/ghost_smoke.sh local` and blocks pushes if any check fails.


### `.github/workflows/ghost_pipeline.yml`

- Runs on every `push` and `pull_request` targeting `main`.
- Steps: checkout → Python 3.11 → install deps → `pytest` (when configured) → `ruff check .` →


   launch uvicorn → `ghost_smoke.sh local` → optional Railway smoke on `main` when secrets exist.

## Required User Actions

1. Install hooks once per clone:


    ```bash
    bash scripts/install_hooks.sh

    ```text

1. Set GitHub repository secrets for Railway verification:

    - `GHOST_RAILWAY_BASE_URL` (defaults to `https://ghost-protocol-production.up.railway.app`).
    - `GHOST_RAILWAY_HEALTH_PATH` (defaults to `/health`).


No other manual steps are needed; every push now triggers the local smoke, and CI mirrors the same
checks automatically.

## Hard Guarantees

- **Pre-push parity:**no commit can be pushed unless `ghost_smoke.sh local` passes, keeping Cockpit


   V3 and the core APIs live.

-**CI enforcement:**GitHub Actions reproduces the same smoke plus unit tests and lint; failures

   block merges by default.

-**No SIM / placeholder regression:**the smoke script fails immediately if any banned tokens

   (`SIM_MODE=1`, `SIMULATION_MODE`, `your_key_here`, `example_api_key`, `PLACEHOLDER`) appear in the
   repo.

-**Single UI contract:**`/cockpit` must serve the V3 assets only; any legacy template markers halt

   the pipeline.

-**Railway verification:** the optional Railway smoke keeps production honest with the same checks

   once secrets are configured.

Further confirmations (date, commit hash, and pass status) will be appended after the first fully
green run across pre-push, CI, and Railway smoke.
