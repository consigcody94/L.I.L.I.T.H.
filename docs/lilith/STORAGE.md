# Storage and size budget

## Recommended shape

Keep relational metadata compact and large numeric arrays compressed outside the transactional database:

- **PostgreSQL + PostGIS:** sources, station metadata/version history, grids, model/run manifests, object references, ingest jobs/watermarks, subscriptions, official-alert lifecycle, report index, delivery outbox and compact metric aggregates. Timescale is optional after measured workload, not a prerequisite.
- **S3-compatible object storage:** raw source payloads, normalized station Parquet, gridded Zarr/NetCDF/GRIB products, forecast arrays, checkpoints and immutable report artifacts. Use partition pruning and chunk sizes aligned to real reads.
- **Redis:** optional cache and short-lived locks/queue acceleration; it is not the durable forecast record.
- **SQLite pilot:** current single-host archive and local verification development. WAL and transactional writes survive restart on a persistent disk. It is not a shared database for independently mounted Modal/serverless workers.

The hosting provider can be selected later. Avoid tying the record format to Modal volumes, Colab Drive, Kaggle outputs or a specific hosting platform. Those are compute/research artifacts that must export into the durable archive.

## Capacity math

Illustrative uncompressed **float32 value bytes**, excluding metadata, indexes, replicas, probabilities, ensembles and compression. Decimal units are used. These are sizing calculations, not a price quote or a claim of observed compression.

| Product assumption | Values per year | Raw bytes/year |
|---|---:|---:|
| 100,000 points × 90 daily leads × 3 variables × 1 issue/day × 365 | 9,855,000,000 | 39.42 GB |
| Same, 4 issues/day | 39,420,000,000 | 157.68 GB |
| 1° global grid (180×360) × 90 leads × 3 variables × 1 issue/day | 6,386,040,000 | 25.54 GB |
| 0.25° grid (720×1440) × 90 leads × 3 variables × 1 issue/day | 102,176,640,000 | 408.71 GB |
| 0.25° grid × 168 hourly leads × 10 variables × 4 issues/day | 2,543,063,040,000 | 10.17 TB |

A SQL row for every point/lead/variable multiplies this by row/index overhead: even the first row is nearly 10 **billion** variable values/year. Ensembles multiply array volume by member count. Dense global high-frequency forever retention will not be a tiny database.

The compact solution keeps a small SQL catalog and compressed numeric objects; it also chooses which full products to retain long-term. Actual compression and retrieval latency must be measured on real arrays. Do not budget with an assumed 10× ratio.

## Compact product and retention policy proposal

User intent is to keep evidence permanently. Maintain immutable reports and reproducible forecast evidence; never discard evidence silently to meet an arbitrary storage target.

- Archive each scheduled global model run once on its native grid. Many user requests reference that run plus an extraction specification; do not duplicate the full forecast per request. If interpolation or calibration changes the returned values, archive the exact extracted output or the deterministic versioned transformation sufficient to reproduce it, with a stable response digest.
- Store float32 compressed chunks, masks separately, categorical metadata as dictionaries, and shared coordinates only once. Quantization needs an error budget and before/after metric checks before adoption.
- Public products: hourly 0–7 days, daily 1–14, weekly 15–42, monthly/seasonal 43–90. Retain full research daily horizons for experiments with explicit archive budgets.
- Keep full grids hot for an initial proposed 30–90 days, move older objects to economical storage, and retain selected issue cycles/sampled evaluation panels/events longer. Changing retention of full arrays needs a recorded decision; permanent compact metrics cannot alone reconstruct deleted forecasts.
- Retain station observations in annual/monthly Parquet partitions by source/variable/geography with revisions; retain sparse official alert/event payloads and report/manifests long-term.
- Keep full output/truth pairs for fixed evaluation samples and all predefined severe-event windows, including matched non-event controls. Preserve enough raw data and versioned transforms to reproduce every published claim.
- Promote only selected checkpoints; retain failed-run manifests/reports and at least the current reproducibility window for discarded weights. Never keep only successful experiments.

## Entities and keys

| Entity | Key content / invariant |
|---|---|
| source | Authority, endpoint/discovery, license, access policy, variable/time/space coverage, freshness contract |
| station_version | Source station ID, WMO/ICAO crosswalks, coordinates/elevation, instrument and valid interval; changes do not overwrite old station metadata |
| grid | CRS, cell geometry, ordering and area weights, coordinate object hash |
| variable | Standard name, physical units, aggregation and interval semantics |
| ingest_batch | Source/cursor, fetch/available time, raw hashes, parser/QC version, counts and status |
| observation_version | Station/grid, valid start/end, variable, value/flags, source issuance/revision, first available time and raw-object reference |
| model_version | Architecture/config, checkpoint/code/training snapshot hashes, supported products, promotion decision |
| forecast_run | Model, issued time and information cutoff, inputs, calibration/blend versions, immutable output manifest |
| forecast_chunk | Run, product, grid/station partition, variable/lead/member range, compression/schema, content hash and object key |
| verification_pair | Forecast value identity, observation version, matcher/metric version, eligibility and exclusion reasons |
| metric_partition | Model/source, region/station/grid, valid/issue period, lead, variable/event, basis and cohort |
| report_version | Input partition hashes, query/cohort, watermarks, code version, artifact keys and supersession |
| alert_message | Authority/sender/identifier/sent/hash, raw/normalized payload and reference graph |
| notification_outbox | Subscription, alert revision/transition/channel identity, attempts, state and delivery receipt |

Separate issue time, valid interval, actual source availability and ingestion time. UTC timestamps do not replace local aggregation definitions. Keep geospatial identifiers stable across source grid changes.

## Mergeable accuracy statistics

For deterministic errors retain `n`, sum of signed errors, sum of absolute errors and sum of squared errors. MAE and RMSE can be reconstructed for retained partitions without retaining all SQL rows. Use double precision or numerically stable accumulation. Store per-model and **paired-case** baseline sufficient statistics; a skill score requires matching support.

Probability/event partitions retain Brier components, probability-bin counts/outcomes and contingency counts at frozen thresholds. CRPS/quantile scores can retain score sums/counts. Distribution diagnostics and block bootstrap intervals require retained block-level summaries or matched samples; sums alone do not recover arbitrary quantiles, event tracks or case-level revisions.

Observation corrections generate delta/recomputed partitions tied to source versions; retain as-published report manifests. “Permanent reports” and “latest corrected dashboard” are separate materialized views.

## Object naming and operations

Suggested keys: `raw/{source}/{available_date}/{sha256}`, `observations/{source}/{schema}/year=YYYY/region=R/...parquet`, `forecasts/{model}/{issued_cycle}/{product}/{chunk}`, `reports/{report_id}/{version}/...`, `models/{model_id}/{checkpoint_sha256}`. Object keys are append-only; SQL transactions publish complete manifests only after all chunk checksums are present.

Use least-privilege worker roles, private raw/checkpoint buckets where required, validated public report exports, retention/lifecycle monitoring and backup restoration drills. Alert outbox and metadata need point-in-time recovery; object manifests need integrity checks and orphan detection. Monitor SQL size, object bytes, request egress, index/query latency and failed ingestion. Establish budgets after measuring real data rather than equating “free API” with free global storage and compute.
