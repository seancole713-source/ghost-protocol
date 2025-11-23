# 🚨 P0 SECURITY INCIDENT: secrets.env Exposed in Git History

**Date Discovered**: October 4, 2025\
**Severity**: **CRITICAL (P0)**\
**Status**: ⚠️ **IMMEDIATE ACTION REQUIRED**

______________________________________________________________________

## Issue Summary

The file `secrets.env` containing API keys and tokens was committed to the Git
repository in the initial import (September 10, 2025) and has remained tracked since.
This means **all secrets in this file are potentially compromised** if the repository
has ever been:

- Public on GitHub
- Shared with external collaborators
- Forked by other users
- Cloned to untrusted machines

______________________________________________________________________

## Evidence

```bash
$ git log --all --full-history -- secrets.env | head -20
commit 4bd3bd60c2698b3d6ec5671a20e9efa9a2826416
Author: seancole713-source <seancole713@gmail.com>
Date:   Wed Sep 10 18:18:41 2025 -0500
    Add files via upload
    Initial import of Ghost.
```

**Conclusion**: `secrets.env` was committed in the initial import and remains in git
history.

______________________________________________________________________

## Affected Secrets (Assume All Compromised)

Based on `wolf_app.py` lines 385-395, the following keys may be exposed:

1. **POLYGON_API_KEY** - Market data provider
2. **ALPHAVANTAGE_API_KEY** / **ALPHA_VANTAGE_API_KEY** - Market data provider
3. **GHOST_API_TOKEN** - Ghost API authentication token
4. **TELEGRAM_BOT_TOKEN** - Telegram bot authentication
5. **TELEGRAM_CHAT_ID** - Telegram chat identifier

**Impact**:

- Unauthorized API usage → billing charges, rate limit exhaustion
- Telegram bot hijacking → malicious messages sent as Ghost
- Ghost API abuse → unauthorized trading actions, data exfiltration

______________________________________________________________________

## Immediate Actions Taken

✅ **Step 1**: Added `secrets.env` to `.gitignore` (prevents future commits)

______________________________________________________________________

## Required Remediation Steps

### 🔥 **CRITICAL - Do These NOW**

#### 1. Rotate ALL API Keys (Assume Compromised)

**Polygon**:

- [ ] Log into https://polygon.io/dashboard
- [ ] Revoke existing API key
- [ ] Generate new API key
- [ ] Update Railway environment variable: `POLYGON_API_KEY=<new_key>`

**AlphaVantage**:

- [ ] Log into https://www.alphavantage.co/support/#api-key
- [ ] Request new API key (or disable old one if possible)
- [ ] Update Railway: `ALPHAVANTAGE_API_KEY=<new_key>`

**Telegram Bot**:

- [ ] Open BotFather in Telegram
- [ ] Send `/revoke` command for Ghost bot
- [ ] Generate new token via `/newbot` or `/token`
- [ ] Update Railway: `TELEGRAM_BOT_TOKEN=<new_token>`
- [ ] Re-set webhook:
  `curl -X POST https://api.telegram.org/bot<NEW_TOKEN>/setWebhook -d url=https://your-ghost-url/telegram/webhook`

**Ghost API Token**:

- [ ] Generate new random token: `openssl rand -hex 32`
- [ ] Update Railway: `GHOST_API_TOKEN=<new_token>`
- [ ] Update any external clients/scripts using old token

#### 2. Remove secrets.env from Git History

**Option A: BFG Repo-Cleaner (Recommended)**

```bash
# Install BFG
brew install bfg  # macOS
# or download from https://rtyley.github.io/bfg-repo-cleaner/

# Clone a fresh mirror
git clone --mirror git@github.com:seancole713-source/GHOST.git ghost-cleanup
cd ghost-cleanup

# Remove secrets.env from ALL commits
bfg --delete-files secrets.env

# Force push cleaned history
git reflog expire --expire=now --all
git gc --prune=now --aggressive
git push --force
```

**Option B: git filter-branch (Manual)**

```bash
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch secrets.env" \
  --prune-empty --tag-name-filter cat -- --all

git push --force --all
git push --force --tags
```

**⚠️ WARNING**: Force-pushing will rewrite history. Coordinate with all collaborators to
re-clone after cleanup.

#### 3. Verify Railway Environment Variables

```bash
# List current Railway vars
railway variables

# Ensure these are set with NEW rotated keys:
# - POLYGON_API_KEY
# - ALPHAVANTAGE_API_KEY  
# - GHOST_API_TOKEN
# - TELEGRAM_BOT_TOKEN
# - TELEGRAM_CHAT_ID
```

#### 4. Audit for Secret Usage

```bash
# Check if secrets.env is actually used in production
railway logs | grep -i "secrets.env"

# Verify Railway uses environment variables (not secrets.env)
# Expected: Railway should NOT load secrets.env (file doesn't exist in deployed container)
```

______________________________________________________________________

## Prevention Measures (Future)

### ✅ Implemented

- [x] Added `secrets.env` to `.gitignore`

### 🔄 Recommended

- [ ] **Pre-commit hook**: Install `git-secrets` or `detect-secrets` to scan for API
  keys

  ```bash
  # Install detect-secrets
  pip install detect-secrets

  # Initialize baseline
  detect-secrets scan > .secrets.baseline

  # Add pre-commit hook
  cat > .git/hooks/pre-commit << 'EOF'
  #!/bin/bash
  detect-secrets-hook --baseline .secrets.baseline $(git diff --cached --name-only)
  EOF
  chmod +x .git/hooks/pre-commit
  ```

- [ ] **Environment variable documentation**: Create `ENV_VARS.md` listing all required
  vars

- [ ] **Template file**: Create `secrets.env.template` that lists required keys with empty values plus a comment pointing
  engineers back to Railway → Variables. Never embed sample secrets in version control.

- [ ] **Railway-only secrets**: Document that production should ONLY use Railway
  environment variables, never local files

- [ ] **Secret scanning**: Enable GitHub secret scanning (if repo is on GitHub)

______________________________________________________________________

## Monitoring for Compromise

### Check for Unauthorized Usage

**Polygon API**:

```bash
# Check usage dashboard for unusual spikes
# https://polygon.io/dashboard/usage
```

**AlphaVantage**:

```bash
# Monitor daily API call limit
# https://www.alphavantage.co/support/#api-key
```

**Telegram Bot**:

```bash
# Check bot message history for unauthorized sends
# Use /getUpdates to see recent activity
curl https://api.telegram.org/bot<TOKEN>/getUpdates
```

**Ghost API**:

```bash
# Review Railway logs for suspicious requests
railway logs --filter "401\|403\|anomaly"
```

### Signs of Compromise

- [ ] Unexpected API billing charges
- [ ] Rate limit exhaustion during off-hours
- [ ] Telegram messages you didn't send
- [ ] Unusual Ghost API activity in logs
- [ ] Failed authentication attempts with old tokens

______________________________________________________________________

## Verification Checklist

Once remediation is complete:

- [ ] All 5 API keys rotated
- [ ] Railway environment variables updated with new keys
- [ ] secrets.env removed from git history
- [ ] All collaborators notified to re-clone
- [ ] Pre-commit hooks installed
- [ ] secrets.env.template created
- [ ] No unauthorized API usage detected
- [ ] Ghost bot functioning with new token
- [ ] This incident documented in CHANGELOG.md

______________________________________________________________________

## Lessons Learned

1. **Never commit secrets**: Use environment variables or secret management tools
2. **Review .gitignore early**: Ensure sensitive files are ignored before first commit
3. **Audit initial imports**: Mass file uploads can accidentally include secrets
4. **Use templates**: Provide `.env.template` files instead of actual secrets
5. **Enable scanning**: GitHub secret scanning, pre-commit hooks, detect-secrets

______________________________________________________________________

## References

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- [detect-secrets](https://github.com/Yelp/detect-secrets)
- [git-secrets](https://github.com/awslabs/git-secrets)

______________________________________________________________________

**Next Steps**: Complete the checklist above, then continue with the deep audit to
identify other security gaps.
