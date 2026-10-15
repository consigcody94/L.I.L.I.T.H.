# Delivery plan and acceptance criteria

All packages contribute to worldwide scope; the sequence is an engineering dependency order, not a decision to restrict the product permanently to one country. No hosting decision is needed to complete the contracts, research, data adapters and local integration.

| Package | Deliverable | Acceptance evidence | Current state |
|---|---|---|---|
| A: audited baseline | Repository audit, explicit capabilities and measured training baselines | Reproducible manifests/results; no invented model skill | Implemented research foundation |
| B: trustworthy records | Immutable daily/hourly archive, source lineage, restart-safe reports | Restart/retry/concurrency/replay tests; storage-failure behavior | Single-host pilot implemented; global object archive pending |
| C: global catalog | Government/open-data inventory, global station/grid IDs and coverage | Per-source licensing/access review, stable IDs, live probes, missingness map | Registry and NOAA history decoder implemented; fleet pending |
| D: independent observations | GHCN daily/hourly and near-real-time regional station intake; gauge/radar/satellite adapters | Interval/unit/QC matching tests, lag watermarks, corrections, fair queue retries | Historical retrieval exists; operational independent matching pending |
| E: fixed global evaluation panel | Scheduled station/grid issue cycles independent of visits | Equal-support baseline scores, location/season/horizon coverage and sufficient counts | Training cohort exists; operational panel pending |
| F: calibrated weather models | Point residual/ensemble models, global NWP bridge, subseasonal products | Frozen temporal/spatial backtests and prospective shadow reports beat agreed baselines | Several real trained candidates; no model promoted |
| G: official alert pipeline | WMO/national source registry, intake, state graph, spatial index and audit | Duplicate/update/cancel/out-of-order/expired/test/multilingual/antimeridian replay | CAP parser foundation implemented; lifecycle/intake pending |
| H: event modeling | Hazard-specific labels, models and calibrated risk products | Recall/false-alarm/lead-time/object metrics on held-out events and controls | Planned; no severe-event model trained |
| I: report library | Immutable daily/weekly/monthly/event/training reports | Rebuild exact report from manifests; corrected versions linked | Training reports retained; scheduled operational library pending |
| J: full site | Global forecasts/alerts/history/accuracy/coverage/archive with mobile/accessibility | Browser workflow tests including unavailable/stale/sparse data | Existing forecast site improved; full UX pending |
| K: alert subscriptions | Opt-in routing, outbox, delivery, receipts and cancellation follow-through | Simulation, idempotent retries, user-controlled opt-out and channel tests | Planned; no external messages sent |
| L: production deployment | User-chosen hosting, SQL/object storage, workers, monitoring and backups | Load/failover/restore drills, actual capacity/cost and shadow operation | Deferred until user supplies destination |

## Required test matrix

- Data: GHCN sentinel/QC/trace/scaling, real leap days, unit conversions, out-of-order updates, station relocation, historical period coverage and unavailable providers.
- Training: no cross-boundary target/input windows, no test-derived normalization/normals, no held-out stations in training, missing-label masks, checkpoint reproduction and separate variable metrics.
- Archive: full 90 leads, every returned hourly field, exact issue replay, idempotent retry, concurrent writes, restart, failed transaction and stale/corrected comparison versions.
- Verification: forecast versus observation interval, local day/DST, source availability, partial data, independent truth versus reanalysis, same-case baseline support, period/region/model filters and correction replay.
- Alerts: authoritative identity, geometry conventions, update/cancel graph, duplicates, multilingual info, Test/Exercise filtering, delayed parents, expiry, feed outage and outbox retry/receipt.
- Site: location change and units, mobile map fallback, keyboard navigation, no mock historical data, explicit reference/model labels, unavailable values, coverage-aware alert emptiness and run/report deep links.

## Priorities for Astra refinement

1. Inspect the actual experiments and exclusion/coverage patterns before changing model size. Aggregate improvement is insufficient; diagnose geographic and precipitation failures.
2. Reconcile open PR #2. Verify its chronological split still purges overlapping labels and its scalers/climate indices are fit only on permissible history. Do not treat its prose as proof.
3. Specify exact issue-time/daily-interval contracts and the independent-station matcher. The pilot explicitly refuses to register a station comparison without that contract.
4. Replace heuristic serving climatology/synthetic history and unused API options. Preserve archived responses when changing inference.
5. Turn the source registry into tested connectors, add WIS2/national authorities and global coverage reporting, and implement durable fair work queues.
6. Complete alert lifecycle/spatial tests and offline delivery simulation before connecting real notification channels.
7. Select production schema/object chunking after measuring the numerical storage budget and query workload. Avoid a gigantic SQL row-per-value design.

## Completion standard

The requested **global production system is complete only when** all forecast products have explicit supported coverage; independent and grid-reference verification run continuously; retained reports are reproducible; alert lifecycle/delivery meets measured objectives; trained models pass the agreed promotion gates; the full site passes realistic workflows; and the user-selected deployment is operational.

This branch is concrete heavy groundwork and measured research. It must not be represented as completion of those remaining systems.
