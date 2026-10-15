# Lilith global weather system

Planning and research baseline, 2026-10-14. Requested scope: worldwide weather, professional alerts, permanent forecast records, comparison with observed outcomes, retained accuracy reports, model training before hosting is selected, and an Astra refinement handoff.

This directory is the implementation plan. [ASTRA_HANDOFF.md](ASTRA_HANDOFF.md) is the continuation entry point. A global plan is not a claim that the current model or alert ingestion already covers every location.

| Document | Purpose |
|---|---|
| [AUDIT.md](AUDIT.md) | What the existing repository does, evidence, and unresolved problems |
| [PRODUCT_AND_ARCHITECTURE.md](PRODUCT_AND_ARCHITECTURE.md) | Site behavior, global coverage, alert workflows, API and worker boundaries |
| [DATA_AND_VERIFICATION.md](DATA_AND_VERIFICATION.md) | Truth types, history, source contracts, matching, accuracy and report retention |
| [STORAGE.md](STORAGE.md) | Compact storage design, capacity math, database entities and lifecycle |
| [TRAINING.md](TRAINING.md) | Model families, temporal and spatial evaluation, training commands and promotion |
| [DELIVERY_PLAN.md](DELIVERY_PLAN.md) | Work packages, dependencies and acceptance criteria |
| [sources.json](sources.json) | Machine-readable government/open-data connector backlog with explicit status |
| [training_runs.json](training_runs.json) | Measured experiment results and artifact hashes |

## What is implemented in this change

- A transactionally persisted single-host SQLite archive for **all returned daily leads**, plus exact hourly-reference payloads; immutable run identifiers and replay endpoints.
- Versioned modeled-reference comparisons, bounded background work, real report-window filtering and exact forecast-point filtering. Daily reports explicitly identify modeled estimates and do not claim station verification.
- Real GHCN station history retrieval with source hashes, quality filtering and nullable missing values. The historical page now requests actual station records instead of displaying invented charts and records.
- An actual hourly NWP-reference path replacing random hourly forecasts and fixed humidity/wind values. Provider outages return unavailable.
- A CAP 1.2 parsing building block retaining official identity, references and multilingual information. This is **not** a complete alert state machine, global ingestion service, or notification service.
- A portable calendar-safe, train/validation/test-separated station experiment pipeline; actual 14-day and 90-day Transformer training on Modal; trained empirical-climatology anomaly-regression baselines; retained prediction arrays and reports.

The production object-store archive, independent station matching service, global government connector fleet, scheduled global verification, calibrated probabilities, severe-event models, alert lifecycle/delivery and operational global site remain delivery work. No hosting service or recurring cloud job is deployed by this change.
