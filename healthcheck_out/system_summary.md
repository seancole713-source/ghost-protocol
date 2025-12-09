# Ghost Healthcheck Summary

## Top clues

### route_diff.txt

```text

```text

### data_feeds.txt

```text

/api/price/WOLF 200 133 15 ms
/api/price/SPY 200 63 3 ms
/api/price/BTC-USD 200 67 2 ms
/api/news 200 5238 138 ms
/api/news/recent 200 106 148 ms

```text

### forecasting_status.txt

```text

forecasts_count 0
accuracy_keys ['error', 'count', 'symbol', 'days']

```text

### feature_matrix.md

```text

- news_routes: OK
- agent_decisions_route: OK
- agent_stats_route: OK
- forecast_api: OK
- crypto_supported: MISSING
- ui_dist_present: OK


```text

### exceptions.tsv

```text

./SYSTEM_STATUS_COMPLETE.md 21 - [x] Exception logging implemented
./SYSTEM_STATUS_COMPLETE.md 102 - [x] Exception logging tests
./SYSTEM_STATUS_COMPLETE.md 173 5. **TYPE_ERRORS_FIXED.md**(2,000+ lines)
./SYSTEM_STATUS_COMPLETE.md 179 6.**TYPE_ERRORS_SUMMARY.txt**(Visual summary)
./GHOST_CAPABILITY_AUDIT_REPORT.md 438 except Exception as e:
./GHOST_CAPABILITY_AUDIT_REPORT.md 920**System Architecture Quality**: **9/10** - Exceptionally well-designed, modular,
observable, resilient. The two-line overlay backend is production-grade code. Only gaps are UI binding and data source
./CRYPTO_MODULE_QUICKSTART.md 83 raise HTTPException(404, f"Unable to fetch price for {symbol}")
./CRYPTO_MODULE_QUICKSTART.md 110 except Exception:
./CRYPTO_MODULE_QUICKSTART.md 115 raise HTTPException(400, "symbol required")
./CRYPTO_MODULE_QUICKSTART.md 132 except Exception as e:
./CRYPTO_MODULE_QUICKSTART.md 134 raise HTTPException(500, f"Crypto prediction failed: {str(e)[:200]}")
./CRYPTO_MODULE_QUICKSTART.md 157 except Exception as e:
./test_apex_integration.py 238 except Exception as e:
./test_general_queries.py 26 "status": "ERROR",
./test_general_queries.py 47 except Exception as e:
./test_general_queries.py 50 "status": "ERROR",
./test_general_queries.py 90 print(f"🚨 ERROR: {r.get('error', 'Unknown')}")
./PROMETHEUS_METRICS_DEBUG.md 47 except Exception as e:
./PROMETHEUS_METRICS_DEBUG.md 143 except Exception as e:
./PROMETHEUS_METRICS_DEBUG.md 151 except Exception:
./signals.py 11 except Exception:  # pragma: no cover - optional
./signals.py 36 except Exception:
./signals.py 49 except Exception:
./signals.py 58 except Exception:
./signals.py 61 except Exception:  # pragma: no cover
./signals.py 81 except Exception:
./signals.py 86 except Exception:
./signals.py 157 except Exception:
./signals.py 243 except Exception:
./signals.py 254 except Exception:
./signals.py 265 if isinstance(r, BaseException) or r is None:
./signals.py 268 except Exception:
./signals.py 279 except Exception:

```text

### services_status.txt

```text

-rw-r--r-- 1 studio713 staff 44K Oct 14 00:01 watchlist.db
-rw-r--r--@ 1 studio713 staff 0B Oct 14 08:56 wolf.db

routes/:
total 40
-rw-r--r--@ 1 studio713 staff 77B Oct 14 12:24 __init__.py
drwxr-xr-x@ 5 studio713 staff 160B Oct 14 22:48 __pycache__
-rw-r--r--@ 1 studio713 staff 3.6K Oct 14 22:18 crypto_ohlcv_routes.py
-rw-r--r--@ 1 studio713 staff 11K Oct 14 12:53 news_routes.py

```text
