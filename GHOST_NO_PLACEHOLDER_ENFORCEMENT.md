# Zero-Placeholder Enforcement Charter

Ghost Protocol ships **real data only**. This charter explains the rules, tooling, and
failure modes that keep fake keys, dummy text, and simulated placeholders out of the
codebase and Railway service.

## Rules

1. **Banned tokens**: any string that literally starts with "your" followed by `_`, plus
  `your-key-here`, `dummy-`, `fake-api-key`, `changeme`, `lorem ipsum`, `test-value`,
  and related variants. Extend the list as new anti-patterns appear.
2. **No stubs**: Never check in mock endpoints, fake accuracy data, or “temporary” UI
   values. If the real dependency is unavailable, fix it or block the merge.
3. **Docs vs. runtime**: Documentation may *describe* placeholders only inside fenced
   code blocks. Runtime files (.py, .sh, .env, .toml, .json, etc.) may not contain the
   banned strings at all.

## Tooling

- `scripts/check_no_placeholders.sh`: scans the repo, ignores vendor caches, and
  validates `/health`, `/cockpit`, `/api/v3/cockpit/version` on the current target.
  - Output on failure: `❌ PLACEHOLDERS FOUND – fix these before commit/push` followed
    by `file:line:value` entries.
  - Output on success: `✅ GHOST CHECKS PASSED`.
- `.githooks/pre-push`: runs the same script; pushes fail until all issues are fixed.
- `.github/workflows/ghost_smoke.yml`: CI launches Ghost, waits for `/health`, then runs
  the script in `GHOST_ENV=local` mode.
- `scripts/check_railway_service.sh`: reuse of the smoke routine for the live Railway
  URL (no placeholder scanning, remote-only validation).

## Exit codes

- `0`: clean scan + healthy cockpit responses.
- `1`: placeholder detected **or** any smoke check failed. The script prints context and
  exits immediately.

## Developer workflow

1. Run `scripts/install_hooks.sh` once per clone to enforce hooks.
2. Develop normally, then execute `scripts/check_no_placeholders.sh` before committing.
3. If the script points to a placeholder line, remove it and implement the real value—do
   not replace it with another fake token.
4. For production, run `scripts/check_railway_service.sh <railway-url>` right after
   Railway finishes deploying to confirm the live environment matches Cockpit V3.

By following this charter we guarantee that Ghost Protocol never serves fake data, never
ships dummy UI, and never regresses to multi-UI chaos.
