# Runtime toggles quick guide

These fields are exposed via GET/POST `/api/runtime/config` and can be adjusted at runtime from the Cockpit Admin panel.

- price_ttl_s (int): Price cache TTL (seconds) when market is closed.
- price_ttl_open_s (int): Price cache TTL (seconds) when market is open.
- news_ttl_s (int): News cache TTL (seconds).
- yahoo_first (int: 0|1): Prefer Yahoo HTTP quote before other providers.
- price_max_deviation_open (float): Max acceptable deviation ratio for live prices during market hours before tripping anomaly guardrails (e.g., 0.8 = 80%).
- reuters_feeds_on (int: 0|1): Enable ingestion of Reuters feeds if configured.
- diag_collapse_dupes (int: 0|1): Collapse duplicate events in the diagnostics ring when they occur within 1s.
- diag_ring_size (int): Size of in-memory diagnostics ring buffer.


Tips

- Changes take effect immediately and are persisted in-process; they are not written to disk by default.
- Increasing `diag_ring_size` will reset the ring; historical events beyond the new size are dropped.
- The Diagnostics panel dedupes on the client as well; server-side dedupe can be tuned with `diag_collapse_dupes`.
- Keep calculations precise internally (6dp) while presenting user-facing percentages at 2dp; rows/KPIs continue to expose 6dp floats via `/api/cockpit`.