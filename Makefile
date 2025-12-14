# Helpful shortcuts
# Use: make help

.PHONY: help verify-live run dev lint lint-full test dev-verify

help:
	@echo "Available targets:"
	@echo "  make run           - Run FastAPI locally on :5000"
	@echo "  make dev           - Install deps and run locally"
	@echo "  make verify-live   - Run utils/verify_live.py against GHOST_URL (default http://127.0.0.1:5000)"
	@echo "  make lint          - Fast lint (syntax/undefined-name)"
	@echo "  make lint-full     - Full repo lint (may be noisy)"

run:
	uvicorn wolf_app:app --host 0.0.0.0 --port 5000

dev:
	pip install -r requirements.txt
	uvicorn wolf_app:app --host 0.0.0.0 --port 5000

verify-live:
	@echo "Running live verifier..."
	@GHOST_URL=$${GHOST_URL:-http://127.0.0.1:5000} \
	GHOST_API_TOKEN=$${GHOST_API_TOKEN:-} \
	ALPHAVANTAGE_API_KEY=$${ALPHAVANTAGE_API_KEY:-} \
	POLYGON_API_KEY=$${POLYGON_API_KEY:-} \
	python utils/verify_live.py

lint:
	# CI-safe lint: catch real breakages without blocking on repo-wide style debt
	# E9: syntax errors, F7/F82: name errors, F63: invalid escape sequences
	ruff check --select E9,F63,F7,F82 core services api utils tests

lint-full:
	ruff check .

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q

dev-verify:
	nohup uvicorn wolf_app:app --host 127.0.0.1 --port 5000 >/dev/null 2>&1 & echo $$! > .ghost_pid
	@sleep 1; echo 'Waiting for /health...'; for i in $$(seq 1 30); do curl -fsS http://127.0.0.1:5000/health >/dev/null && break || sleep 1; done
	GHOST_URL=http://127.0.0.1:5000 python utils/verify_live.py || true
	@kill $$(cat .ghost_pid) 2>/dev/null || true; rm -f .ghost_pid
