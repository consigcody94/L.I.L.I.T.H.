# Astra handoff — Lilith global weather research and platform foundation

## User intent and constraints

Build Lilith as a **global** weather system, not a local-only product: forecast temperature and other weather, identify potential major events, integrate free/open government sources, provide professional authoritative alerts, retain original predictions and observations, compare against actual outcomes, and keep detailed accuracy reports. The user wants model training before deployment. Latest steering specifies the existing NAS for storage now and future hosting on Proxmox with local NAS access. Historical records from 1980 and earlier are valuable; retain full source history where quality and availability permit.

The user asked this agent to do the heavy groundwork and keep a handoff for Astra to refine it. This file is the handoff; no separate Astra chat was created or messaged.

## Repository and branch

- Repository: https://github.com/consigcody94/L.I.L.I.T.H.
- Starting revision: `10a3a00` on master.
- Working branch: `codex/global-weather-foundation`.
- Existing related work: [PR #2](https://github.com/consigcody94/L.I.L.I.T.H./pull/2), branch `improvements/2026-data-and-perf`; inspected but not merged. It contains overlapping decoder/embedding fixes and separate data/full-trainer/ensemble work. Resolve overlaps deliberately.
- Planning index: [README.md](README.md). Read [AUDIT.md](AUDIT.md) before trusting legacy README claims.

## Implemented groundwork

1. **Durable forecast evidence:** `web/api/storage/ledger.py` writes complete daily runs, all returned daily leads and complete hourly reference payloads transactionally to SQLite. Content-derived IDs, replay paths, idempotent duplicate writes and separate versioned reference comparisons replace the old daily in-memory/seven-lead store. A failed archive write returns unavailable. No database is silently claimed to support global scale.
2. **Honest reference comparison:** reports apply real issue-time and exact-point filters. GET report reads do not start provider downloads. Public manual verification mutation returns 405; bounded worker/CLI is provided. Open-Meteo old forecasts/analysis are labeled as modeled references, with zero independent station verification. Partial truth stays null and does not become a zero-error sample.
3. **Real historical records:** `data/providers/ghcn_records.py` decodes missing/QC/measurement/source flags, correct units and actual calendars. `/v1/historical` fetches any valid cataloged global GHCN daily station ID with source hash. The historical page displays actual observations/completeness instead of invented charts and record events. Geographic station discovery remains unfinished.
4. **Real hourly reference:** API now fetches actual hourly NWP temperature, humidity, rain, wind, cloud/pressure values with UTC timestamps and m/s units; random hourly fallback and constant unmodeled fields were removed. It is explicitly a reference product, not a Lilith hourly model. Hourly scoring is not implemented.
5. **Alert intake building block:** `data/providers/cap_alerts.py` preserves CAP identity/status/references/languages/areas and rejects dangerous XML declarations and missing timezone information. It is not a complete CAP validator, authority authenticator, lifecycle engine, spatial matcher or notification service.
6. **Research training:** calendar-aligned real NOAA station data, whole-window time splits, station holdout, train-only normalization, same-case empirical climatology/persistence baselines and physical-unit metrics. Modal GPU and Kaggle reproduction ran; reports, checkpoints and per-case arrays are retained.
7. **Model fixes required by validation:** full LILITH station embedding receives the documented 4D input; decoder preserves station/time axes and uses normalization to bound rollout; a shared graph is offset across batch elements. Regression tests cover 14/90-step output. These changes do not turn the full architecture into a trained global model. New LayerNorm parameters require migration/retraining for old full-model weights.
8. **Other concrete fixes:** legacy GHCN SNOW/SNWD scaling corrected; training package imports are lazy; missing frontend `cn` utility added so production builds succeed; uncertainty and accuracy UI labels no longer claim calibrated 95% intervals or independent station checks.

## Actual completed training

See [training_runs.json](training_runs.json), the full retained reports and `TRAINING_RESULTS.md` in the local outputs. All model status is `research_not_promoted`.

| Candidate/cohort | Test daily high / low RMSE °C | Same-case climatology °C |
|---|---|---|
| Transformer 14 days; 72 selected; 1991-based | 4.578 / 4.168 | 4.134 / 3.865 |
| Transformer 90 days; 72 selected; 1991-based | 5.241 / 4.663 | 4.256 / 3.986 |
| Transformer 14 days; 144 selected; 1980-based | 4.400 / 4.534 | 4.447 / 4.563 |
| Anomaly regression 14 days; 72 selected | 3.989 / 3.705 | 4.134 / 3.865 |
| Anomaly regression 90 days; 72 selected | 4.229 / 3.954 | 4.256 / 3.986 |
| Anomaly regression 14 days; 144 selected | 4.235 / 4.375 | 4.447 / 4.563 |
| Kaggle T4 independent 14-day Transformer fit | 4.489 / 4.133 | 4.134 / 3.865 |

Do not directly compare absolute errors between different cohorts. The original 72 selected stations produce only **17** scored test stations at 14 days and **14** at 90 days; the 144-station cohort produces **42**. Complete-window filtering creates strong coverage bias. Held-out station counts are 4, 4 and 8, respectively. Their separate scores are in the reports. No outcome supports a “verified everywhere” claim.

Training/validation/test are 1991–2016 (or 1980–2016), 2017–2020, 2021–2025. Station holdouts are excluded from fitting. Test windows are nonoverlapping by target in the 14-day case; 90-day targets overlap because stride is 14, so uncertainty must use time/station blocks. Normalization is training-only. Baseline station normals use training years; held-out stations use a nearest-training-station normal, with this limitation recorded.

The initial raw Transformers lose to climatology; the larger 1980 model shows small temperature gains; the statistical residual baseline is stronger. Rainfall gains are absent or negligible. None is calibrated for rare events or compared to archived issue-matched NWP yet. The benchmark was reused during refinement after its first results were inspected; require a new future/prospective audit.

### Compute and artifact locations

Modal workspace/profile `consigcody94`; private volume `lilith-verified-research`. Completed applications:

- 72-station h14: https://modal.com/apps/consigcody94/main/ap-ILkDpisijLcRObEyXDxMrr
- 72-station h90: https://modal.com/apps/consigcody94/main/ap-1Nmmlyv7YoQfXLpIzGGci6
- 144-station h14 1980: https://modal.com/apps/consigcody94/main/ap-qQW9qv1Pf2DKq46zP168On

Volume outputs: `runs/global-h14-20261014`, `runs/global-h90-20261014`, `runs/global-h14-1980-20261014`. Dataset folders originally `datasets/h14` and `datasets/h90`; the later 1980 run superseded `datasets/h14`. Original h14 data is preserved locally and in the frozen private Kaggle dataset. The launcher now uses unique dataset paths and refuses completed-name reuse.

Kaggle authenticated successfully using `uv run --with kaggle kaggle auth login`:

- Private benchmark: https://www.kaggle.com/datasets/codychurchwell/lilith-global-station-benchmark-20261014
- Completed private script: https://www.kaggle.com/code/codychurchwell/lilith-global-heldout-reproduction-20261014 (successful version 2).
- `docs/lilith/reproduction` contains launch metadata and code. Original source/data archives are also retained with outputs.
- T4/FP16 versus L4/BF16 training is not numerically identical; frameworks/devices and actual results are recorded. Do not call it bitwise reproduction.

Colab: account was readable and an unrelated session was active. It was not used, altered or terminated. Never stop a session belonging to unrelated work. Kaggle and Modal jobs created for Lilith have bounded execution; no recurring job or public inference endpoint was deployed.

Local research outputs are under this chat’s `outputs/training`; checkpoint filenames are `candidate.pt` or `coefficients.npz`. Full source/report/checkpoint hashes are in the summary. Git contains code, plans and compact results, not large trained weights. The handoff is copied into `outputs/plan` with its companion documents.

## Verification completed

- Python suite: 97 passed, zero failures in the final validation run; includes archive revisions/reversions, partial comparisons, calendar splits, CAP ingestion, shared-graph batching and hourly reference handling.
- Frontend production build: passed, including TypeScript checks and static generation.
- New Python modules: focused Ruff fatal/undefined/unused-import checks passed; full repository style cleanup is not part of this change.
- Live station-history browser workflow: loaded Central Park 1979 and 1980 records, correct 365/366-day counts and NOAA source hash; no sample chart values. Browser’s only observed console error was the existing missing favicon.
- Live reference capture: six cities spanning the Americas, Europe, Africa, Asia and Oceania, 90 returned daily leads each, exact archive replay. These are Open-Meteo + existing heuristic seasonal fallback outputs, not promoted Lilith model output. No future actuals are invented.

Live hourly New York reference data also returned 24 hours and replayed exactly. A Tokyo hourly probe timed out upstream and correctly returned 503. The prospective archive and its capture report are in local `outputs/shadow`. It starts evidence retention; it is not continuous verification or a completed accuracy evaluation.

## Priority continuation

Read [RESEARCH_DECISIONS.md](RESEARCH_DECISIONS.md) and its three detailed companion reviews first. They incorporate NOAA's current GHCNh replacement for ISD, the March 2026 SPC intensity schema, marine/upper-air validation, modern global-model papers, proper hazard verification and the existing NAS/future Proxmox topology. Use the NAS for immutable bulk artifacts and local guest disk for hot SQL/WAL. The research documents are evidence and proposals, not implementation of the remaining connectors or qualification of model skill. Private host/capacity details and the checksum-verified transfer receipt are retained in local outputs rather than public Git.

1. Audit the case-level prediction arrays, physical violations, geography and missingness. Broaden data using causal masks rather than dropping almost every incomplete station window. Preserve older history without pretending historical revisions were available at issue time.
2. Specify the observation-interval and source-availability contracts. Implement independent station matching, including station ID, distance/elevation, local/reporting-day support, source/QC and correction reconciliation. Current pilot deliberately rejects a `station` comparison without this contract.
3. Add original issue-time NWP predictors/baselines, observation release-lag handling, calibrated distributions and hazard-specific targets. Remove or disclose legacy synthetic serving input and heuristic climatology before production. Raw-model and blend skill must be scored separately.
4. Reconcile PR #2 and rerun its tests. Check preprocessing/scaling/climate-index publication leakage and purge boundaries, not just the existence of chronological split code.
5. Implement tested global data and official-authority connectors from `sources.json`; registry entries marked planned are not wired feeds. There is no single universal free government API that completes this work.
6. Build CAP reference graph/state, authoritative feed review, spatial normalization/intersection, outage status and notification outbox/simulation. Do not send live notifications or call generated experimental risk messages official warnings.
7. Build object-array storage, compact SQL metadata and mergeable score partitions. `STORAGE.md` quantifies why dense global hourly forever storage is not small. SQLite is a development pilot with memory-materialized reports, not the production architecture.
8. Complete the site views in `PRODUCT_AND_ARCHITECTURE.md`, including coverage-aware accuracy maps, report library and archive UI. The delivered historical page is functional groundwork, not the finished global product.
9. Use the existing NAS archive and future Proxmox hosting plan. Resolve guest/service details only when needed for deployment; do not re-ask which bulk storage to use. Continue reversible local/research work without repeated permission requests. External real notifications require explicit channel/recipient authorization.

## Important remaining limitations

- Operational report aggregates currently include complete modeled three-variable comparisons only; independent station scores, partial-variable metrics and model-separated operational aggregations remain work.
- Background processing is bounded but not a durable fair scheduler; outages/missing data can starve a pending batch. Add retries/watermarks/work claims and SQL/columnar aggregation before global scale.
- Automatic reconciliation does not yet revisit completed modeled comparisons after source corrections; append-only revision support exists but a correction job is still needed.
- Timezones and daily aggregation windows need resolution. Naive legacy times are treated as UTC; run serving with `TZ=UTC` during the pilot.
- Actual source input manifests, raw preblend output and blend weights are not fully archived by the serving path yet. Full returned payloads and weights hash are archived.
- API options such as arbitrary `start_date`, `variables` and ensemble counts remain inconsistently supported. Reject unsupported requests before production.
- Global severe-event modeling, calibrated alert probabilities, complete authoritative alert coverage, subscription delivery and production deployment are **not complete**.
