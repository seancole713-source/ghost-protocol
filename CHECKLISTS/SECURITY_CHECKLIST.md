# GHOST Security Checklist

**Version**: 1.0
**Date**: October 4, 2025
**Owner**: Security Team / DevSecOps
**Frequency**: Pre-release + Quarterly

---

## Authentication & Authorization

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ All write endpoints require bearer token | ⚠️ **PARTIAL**| P0 | 3 debug endpoints lack auth (GH-AUD-002) |
| ✅ `GHOST_API_TOKEN` stored in Railway env (not code) | ✅**DONE**| P0 | Verified via git grep |
| ✅ Token rotation procedure documented | ✅**DONE**| P0 | See `SECURITY_INCIDENT_P0_SECRETS.md` |
| ✅ Rate limiting configured for write endpoints | ⚠️**PARTIAL**| P1 | `RATE_LIMIT_WRITE_RPM=0` (disabled by default)
|
| ✅ Admin IP allowlist configured (if needed) | ⚠️**OPTIONAL**| P2 | `ADMIN_IP_ALLOWLIST` exists but empty |
| ✅ Bearer token exposed in logs | ✅**SAFE**| P0 | Verified: no token logging detected |**Action Items**:

- [ ] Add auth to `/debug/telegram_test` (GH-AUD-002)
- [ ] Add auth to `/debug/prev_close` (GH-AUD-002)
- [ ] Add auth to `/debug/price_diag` (GH-AUD-002)
- [ ] Enable rate limiting in production: `RATE_LIMIT_WRITE_RPM=60`


---

## Secrets Management

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ No secrets in git history | ❌ **FAIL**| P0 |**GH-AUD-001**: `secrets.env` committed Sept 10 |
| ✅ All secrets in Railway environment variables | ⚠️ **PENDING**| P0 | Must rotate 5 keys after history cleanup |
| ✅ `.gitignore` blocks `secrets.env` | ✅**DONE**| P0 | Added in audit remediation |
| ✅ `secrets.env.template` provided for devs | ✅**DONE**| P1 | Created during audit |
| ✅ Pre-commit hooks prevent secret commits | ❌**MISSING**| P1 | Need `detect-secrets` hook |
| ✅ API keys rotated after exposure | ⚠️**PENDING**| P0 | Awaiting rotation |**Action Items**:

- [ ] **IMMEDIATE**: Rotate all 5 API keys (POLYGON, ALPHAVANTAGE, GHOST_API_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
- [ ] Remove `secrets.env` from git history (BFG Repo-Cleaner)
- [ ] Install `detect-secrets` pre-commit hook
- [ ] Add `.secrets.baseline` to repo
- [ ] Document rotation procedure in runbook


---

## Network Security

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ HTTPS enforced for all external APIs | ✅ **DONE** | P0 | HTTPAdapter uses HTTPS |
| ✅ CORS configured (not open `*` in prod) | ⚠️ **PARTIAL** | P1 | Default `ALLOWED_ORIGINS=*` (dev mode) |
| ✅ CSP headers enabled | ✅ **DONE**| P1 | Line 114: CSP middleware |
| ✅ HSTS enabled | ✅**DONE**| P1 | Line 122: `HSTS_ON=1` |
| ✅ Telegram webhook signature validation | ❌**MISSING**| P2 |**GH-AUD-007**: No secret token check |
| ✅ No eval/exec of user input | ✅ **SAFE**| P0 | Grep: zero unsafe code execution |
| ✅ SSRF protection (external fetches limited) | ✅**DONE**| P1 | Whitelist on Reuters feeds |**Action Items**:

- [ ] Set `ALLOWED_ORIGINS` to specific domains in production
- [ ] Generate Telegram webhook secret: `openssl rand -base64 32`
- [ ] Add webhook validation to `/telegram/webhook` (GH-AUD-007)
- [ ] Review CSP policy: ensure `script-src` not `unsafe-inline`


---

## Data Protection

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ PII/credentials never logged | ✅ **SAFE**| P0 | Verified: sanitized logging |
| ✅ SQLite database files in persistent volume | ✅**DONE**| P1 | Railway `/data` mount |
| ✅ Database backups configured | ⚠️**MANUAL**| P2 | `scripts/railway_backup.py` exists |
| ✅ Sensitive fields encrypted at rest | ⚠️**N/A**| P2 | No credit cards/SSNs stored |
| ✅ Position data requires auth to view | ✅**DONE**| P1 | `/api/positions` requires token |**Action Items**:

- [ ] Automate database backups (daily cron via Railway)
- [ ] Test restore procedure from backup
- [ ] Add encrypted field support if storing bank account numbers


---

## Dependency Security

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ `pip-audit` run on requirements.txt | ⚠️ **UNKNOWN**| P1 | Not run during audit |
| ✅ All packages pinned to specific versions | ⚠️**PARTIAL**| P2 | Some use `>=` (e.g. `fastapi>=0.100`) |
| ✅ Dependabot enabled on GitHub | ⚠️**UNKNOWN**| P2 | Check repo settings |
| ✅ Known CVEs addressed | ⚠️**UNKNOWN**| P1 | Run `pip-audit` to verify |**Action Items**:

- [ ] Run `pip-audit` and document results
- [ ] Pin all packages to exact versions (no `>=`)
- [ ] Enable Dependabot security alerts
- [ ] Schedule monthly dependency review


---

## Incident Response

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Incident response plan documented | ⚠️ **PARTIAL**| P1 | `SECURITY_INCIDENT_P0_SECRETS.md` is first |
| ✅ On-call rotation defined | ⚠️**UNKNOWN**| P2 | Check ops runbook |
| ✅ Log aggregation for forensics | ✅**DONE**| P1 | Structured JSON logs to Railway |
| ✅ Breach notification procedure | ⚠️**MISSING**| P2 | Need policy document |
| ✅ Post-mortem template | ⚠️**MISSING**| P2 | Need template |**Action Items**:

- [ ] Create full incident response runbook
- [ ] Define security escalation contacts
- [ ] Document log retention policy (how many days?)
- [ ] Create post-mortem template (5 Whys format)


---

## Testing & Validation

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Security tests in test suite | ⚠️ **PARTIAL**| P1 | `test_debug_auth.py` needed |
| ✅ Penetration test performed | ❌**NEVER**| P2 | Consider for v1.0 launch |
| ✅ OWASP Top 10 checklist reviewed | ⚠️**UNKNOWN**| P1 | Need formal review |
| ✅ Auth bypass attempts tested | ⚠️**PARTIAL**| P1 | Manual curl tests only |**Action Items**:

- [ ] Create `tests/test_security.py` (auth bypass, CORS, CSP)
- [ ] Run OWASP ZAP scan against staging
- [ ] Document pen test findings if performed
- [ ] Add security regression tests to CI/CD


---

## Compliance & Auditing

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Audit trail for admin actions | ⚠️ **PARTIAL**| P2 | Logs exist but no dedicated audit table |
| ✅ User consent for data collection | ⚠️**N/A**| P3 | No user registration yet |
| ✅ Data retention policy | ⚠️**MISSING**| P2 | How long keep AI memory? |
| ✅ Privacy policy published | ⚠️**N/A**| P3 | Single-user system |**Action Items**:

- [ ] Create audit log table (user, action, timestamp, IP)
- [ ] Define AI memory retention (e.g., keep 365 days)
- [ ] Add `/admin/audit` endpoint for admin actions review


---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Security Lead | TBD | YYYY-MM-DD | ____________ |
| DevOps Lead | TBD | YYYY-MM-DD | ____________ |
| Product Owner | TBD | YYYY-MM-DD | ____________ |

**Next Review**: 2026-01-04 (Quarterly)

---

**Checklist Maintained By**: Security Team
**Last Updated**: October 4, 2025
**Version**: 1.0
**Related Documents**: `SECURITY_INCIDENT_P0_SECRETS.md`, `GHOST_DEEP_AUDIT.md`
