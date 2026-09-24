# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

Pre-implementation. The only source of truth is `plan/implementation-plan.md` (in Russian): read it before any non-trivial change. No build/lint/test commands exist yet. Once the `uv` project and `Makefile` are created, add the real commands here.

## Project

Moscow Transport hackathon, track 2 (https://mt-hackathon.ru/): forecast tram ridership at three horizons (1 day, 1 month, 1 year). Aggregate forecasts by route, stop and time interval. Serve them in a web map UI and include a backtest quality report against naive baselines. All in Python.

Organizer data arrives 2026-09-25 after 10:00. Until then, everything runs on synthetic data (`scripts/make_synthetic.py`) generated in the canonical schema.

## Intended architecture

```
raw organizer files
  → adapters/            # ONLY place aware of organizer format → canonical parquet
  → aggregate.py         # canonical → stop_hour.parquet (route × stop × hour)
  → features.py → train.py (LightGBM + rolling-origin backtest, CLI)
  → forecast.py          # batch forecasts → forecasts.parquet
  → api.py               # FastAPI; serves /api/* and static/ (MapLibre GL + ECharts, single page)
```

Key design constraints:
- **Adapter isolation:** everything downstream of `adapters/` depends only on the canonical schema (`validations`, `telematics`, `stops`, `route_stops`; see plan §3). Real-data integration should require only a new adapter.
- Storage is parquet + DuckDB. Processing uses polars/duckdb, with DuckDB out-of-core if data exceeds memory.
- **Forecasts are precomputed.** The API reads parquet and never loads models in the request path.
- **Models:** one global LightGBM per horizon (direct strategy). Each horizon uses only lags that are available at forecast time: day ≥1d, month ≥35d, year ≥364d. Each is compared to its horizon-specific naive baseline (plan §6).
- **Validation:** time-ordered rolling-origin backtest with ≥3 windows and no shuffling. The primary metric is WAPE; MAE and peak-hour metrics are also reported.
- **Target:** the MVP target is `boardings` per (route, stop, hour). Segment load (boardings − alightings) is a stretch goal. It needs trip chaining if `card_id` exists, or a gravity-style approximation if not.
- If validations lack `stop_id`, recover it via `vehicle_id + ts` → nearest telematics point (`merge_asof`) → nearest route stop (KD-tree).

Planned layout: `src/tram_forecast/`, `scripts/`, `tests/`. Toolchain: `uv`. The Makefile must follow the user's Makefile standard.
