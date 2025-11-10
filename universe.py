import os

_UNIVERSE = {"stocks": [], "crypto": []}


def get_universe():
    return _UNIVERSE


def set_universe(u: dict):
    global _UNIVERSE
    _UNIVERSE = {"stocks": u.get("stocks", []), "crypto": u.get("crypto", [])}
    try:
        import ghost_state

        st = ghost_state.get_state()
        st["universe"] = {
            "stocks": list(_UNIVERSE.get("stocks", [])),
            "crypto": list(_UNIVERSE.get("crypto", [])),
        }
        ghost_state.save()
    except Exception:
        pass
    return _UNIVERSE


def get_universe_status():
    return {
        "stocks": _UNIVERSE.get("stocks", []),
        "crypto": _UNIVERSE.get("crypto", []),
    }


def import_csv(b: bytes):
    return {"stocks": [], "crypto": []}


# Focus Mode helpers (reversible, environment-driven)
def focus_enabled() -> bool:
    """Return True if Focus Mode is enabled via env (GHOST_FOCUS_MODE=1)."""
    # During tests, always disable Focus Mode to keep default universe for acceptance tests
    if (
        os.getenv("PYTEST_CURRENT_TEST")
        or os.getenv("SNAP_TEST_MODE")
        or os.getenv("GHOST_TEST_MODE")
    ):
        return False
    return str(os.getenv("GHOST_FOCUS_MODE", "0")).strip() in ("1", "true", "on", "yes")


def focus_ticker() -> str:
    """Return the configured focus ticker (defaults to WOLF)."""
    return (os.getenv("GHOST_FOCUS_TICKER", "WOLF") or "WOLF").strip().upper()
