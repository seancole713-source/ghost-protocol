import sys
import traceback

sys.path.insert(0, ".")
try:
    from wolf_app import APP

    print("OK: imported APP")
    rows = []
    for r in getattr(APP, "routes", []):
        p = getattr(r, "path", None)
        m = getattr(r, "methods", None)
        if p:
            rows.append((p, ",".join(sorted(m)) if m else ""))
    print("TOTAL_ROUTES", len(rows))
    for p, m in sorted(rows):
        if "/news" in p.lower():
            print("NEWS", m, p)
except Exception as e:
    print("IMPORT_ERROR:", e)
    traceback.print_exc()
