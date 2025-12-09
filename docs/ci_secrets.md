# CI/CD Secrets Guide

This document lists the secrets used across workflows and what they do.

## Verification workflows

- `VERIFY_BASE_URL` (optional): Base URL used by scheduled verify workflow (`verify-live-schedule.yml`). If empty, job is skipped.
- `GHOST_API_TOKEN` (optional): Bearer token for protected endpoints during verification.
- `ALPHAVANTAGE_API_KEY` (optional): Enables price parity checks.
- `POLYGON_API_KEY` (optional): Enables news parity and prev-close checks.

PR pre-merge (`pr-verify.yml`):

- Uses localhost server spun up in CI; works without secrets. If you add `GHOST_API_TOKEN`, provider keys, parity checks are enriched.

## Deploy workflow (`deploy.yml`)

- `VPS_SSH_KEY`: Private key to SSH into the production VPS.
- `VPS_HOST`: SSH host for production.
- `VPS_USER`: SSH user for production.
- `VPS_PATH`: Directory path on VPS where repo is synced.
- `DEPLOY_BASE_URL` (optional): Public base URL of production. If set, the workflow runs live verification after deploy.
- `GHOST_API_TOKEN` (optional): Bearer token used by the post-deploy verifier.
- `ALPHAVANTAGE_API_KEY` / `POLYGON_API_KEY` (optional): Enable external parity checks during post-deploy verification.

## Staging workflow (`staging.yml`)

- `STAGING_SSH_KEY`: Private key to SSH into the staging server.
- `STAGING_HOST`: SSH host for staging.
- `STAGING_USER`: SSH user for staging.
- `STAGING_PATH`: Directory path on staging where repo is synced.
- `STAGING_BASE_URL`: Public base URL of the staging environment. Used by the live verifier.
- `STAGING_GHOST_API_TOKEN` (optional): Bearer token for staging verification.
- `ALPHAVANTAGE_API_KEY` / `POLYGON_API_KEY` (optional): Shared provider keys for parity checks.

Notes:

- Secrets should be stored at the repository level unless you’re using environments; in that case, put them under the environment for better isolation and approvals.
- Provider keys are optional; the verifier will WARN (not fail) on parity checks when keys are missing or rate-limited.
