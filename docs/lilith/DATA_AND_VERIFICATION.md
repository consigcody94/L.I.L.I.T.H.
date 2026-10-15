# Data, historical records and forecast verification

## Truth must have a type

| Record type | Examples | Permitted interpretation |
|---|---|---|
| Independent station observation | QC-approved GHCN temperature, ASOS observation | Station verification when location, variable and observation interval match |
| Gauge/radar/satellite estimate | Rain gauge, MRMS, satellite precipitation | Variable-specific estimate with its own error and coverage; no blanket ground-truth label |
| Gridded analysis/reanalysis | ERA5, ERA5-Land, analysis fields | Spatially complete modeled estimate; separately reported grid-reference agreement |
| Historical NWP forecast | Forecast issued before its valid time | A competing forecast/baseline, never an observation |
| Preliminary event report | Local storm report | Provisional positive evidence; lack of a report does not prove an event did not occur |
| Curated event record | Finalized storm/cyclone database | Event evaluation with version, delay, reporting/selection bias and spatial matching |
| Official warning | CAP warning message | Authority-issued risk communication; not proof an event occurred |

[Open-Meteo’s historical API](https://open-meteo.com/en/docs/historical-weather-api) is based on modeled reanalysis/analysis. Its recent `past_days` values are not independent station truth. The current pilot makes this distinction explicit. “Free” also needs an access/usage contract: the [Open-Meteo terms](https://open-meteo.com/en/terms) distinguish allowed usage, limits and commercial service.

## Historical data policy

Retain the **full available, quality-controlled source history** per station/variable. A 1980 cutoff is not imposed on storage. Some stations have much older records, but periods and completeness differ. Use records before 1980 for station normals, rare-event climatology, long-period backtests and regime studies where defensible.

Do not flatten all eras into an undifferentiated training set. Record relocations, instrumentation, elevation, observation-time changes and missingness. Compare recent-period and full-history training, climate-drift weighting and stable-station cohorts. Keep a consistent gridded reanalysis product across a hindcast; do not splice changing operational model estimates into “homogeneous observations.”

The initial 1991-based pilots preserve their frozen manifests. The next run uses 1980 onward. The new preparation CLI accepts `--start-year 1850..2000`; this is a configurable experiment boundary, not a statement that all selected stations have complete records from that year. The station inventory’s first/last year is only an eligibility filter; actual completeness is measured after decoding.

Use the official [GHCN format](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt): temperature and precipitation require tenths scaling, but snowfall and snow depth are already millimeters. Preserve missing sentinel and measurement/quality/source flags. Trace precipitation is a measured trace, not a missing value. Rejected or absent targets stay null; never fill them with a daily/feature mean to improve apparent evaluation coverage.

## Connector contracts

The [source registry](sources.json) enumerates data families, authority, coverage, role, access class and implementation status. No connector is called implemented merely because its URL is listed. Start with global government sources and add authoritative national refinements through the same interface; coverage is global by design, while completion of individual adapters is tracked explicitly.

Every connector must implement:

- Discovery/inventory with coverage polygon, supported variables, resolution, history, expected release delay, stable source IDs and source docs.
- Bounded fetch/backfill with timeouts, retry-after, backoff/jitter, conditional requests, checkpointed cursors, overlapping reconciliation windows and rate limits.
- Raw-payload retention/digest, fetched/issued/valid/available timestamps, parser version, units/aggregation, CRS/grid/station identifiers, source and license metadata.
- QC, duplicate detection, missingness, plausible-value and cross-field checks; quarantine invalid records without rewriting source truth.
- Health/freshness/coverage state; successful empty response is distinct from error and from unsupported coverage.
- Offline contract tests and a small explicit live probe. Public bulk/S3/FTP/WIS2 feeds are valid interfaces even when not REST. Registration-required access is not silently treated as keyless.

[NOAA NWS](https://www.weather.gov/documentation/services-web-api) provides forecasts, alerts and observations and requires an identifying User-Agent. NWS is not the whole world. [WIS2 discovery](https://wmo-im.github.io/wis2-guide/guide/wis2-guide-APPROVED.html) and national meteorological authorities supply the broader registry; data discovery still needs parser, terms and latency review per dataset.

## Immutable forecast record

At issuance, persist a run and values before returning success:

- Issue time and information cutoff; source run times and actual availability times.
- Model family/version, checkpoint hash, code revision, input/normalization/calibration manifests, data snapshot hashes and configuration.
- Requested/resolved coordinates, station/grid cell/elevation, IANA timezone and exact valid interval/aggregation semantics.
- Raw model output, independent baseline, any blended published output, blend weights and per-variable provenance. Do not archive only the corrected final signal and lose attribution.
- Mean/quantiles/ensemble or deterministic value, unit, missing flags and probability event definition/threshold.
- Product kind and trained/supported horizon. A seasonal anomaly is not an hourly temperature measurement.

Current implementation preserves the returned daily/hourly payload, model-version text and available checkpoint hash. It does not yet capture every upstream input artifact, raw preblend model output or blend weight; those are required production extensions.

## Matching and comparison rules

A verification pair is `(prediction value, observation version, matcher version, metric definition)`. It must retain station ID, great-circle distance, elevation difference, spatial interpolation method, temporal support, units and QC status.

1. Resolve the exact valid interval. Daily high/low and precipitation are not interchangeable across UTC day, local calendar day, station reporting day or precipitation accumulation window. Handle DST and non-hour offsets.
2. Use observed station values only where the station contract matches. Record differing time-of-observation semantics. A nearest station across a mountain or 100 km away does not establish local accuracy.
3. Evaluate gridded outputs with area-weighted grid metrics and separately matched stations. Do not pool these as one global score. Retain land/ocean and station-density strata.
4. Enforce maturity based on source release lag. Missing data stays pending; partial availability is tracked per variable. Preliminary reports can be superseded by curated data.
5. Append observation corrections. Publish both as-first-verified and latest-reconciled metrics where appropriate, recording changed inputs. Never rewrite the original forecast.
6. Use a fixed scheduled evaluation panel/grid, independent of website traffic. Requests alone would over-weight popular cities and weather-interest days. Deduplicate reissued/retried requests by the evaluation run/issue-cycle contract.
7. Compare candidates and baselines on **identical matched cases**. Score raw Lilith, reference NWP, persistence, empirical climatology, calibrated blend and each model version separately.

## Metrics and honest coverage

Temperature/wind/pressure: per-variable MAE, RMSE, bias and distributions. Precipitation amount: MAE/RMSE, accumulation-window skill, wet-day and intensity strata; occurrence probability: Brier score, Brier skill and reliability. Quantiles/ensembles: CRPS or quantile loss, empirical interval coverage/width, rank histograms and tail calibration.

For rare hazards retain hits, misses, false alarms and correct negatives with probability thresholds fixed on validation. Report precision/recall, probability of detection, false-alarm ratio, CSI/ETS and precision-recall curves; percentage accuracy alone can reward always predicting “no event.” Use spatial/temporal neighborhoods and object tracks appropriate to each phenomenon. Missing reports do not automatically become correct negatives.

Show lead hours/days, region/climate/elevation, month/season, land/ocean, source, model and sample/coverage counts. Use confidence intervals from block bootstrap by station/time/event, not independent-row assumptions. Retain minimum evidence thresholds and report suppressed/insufficient strata. Do not invent a score for every coordinate; interpolate validation evidence only as an explicitly separate research layer.

## Retained reports

- Daily operational report: verification watermark, intake health, matched/pending/excluded counts, missingness and product freshness.
- Weekly model report: model/reference/climatology/persistence scores by lead and regional cohort, calibration, coverage and changes from previous version.
- Monthly global report: station/grid-weighted regional/seasonal tables, uncertainty, rare-event performance, drift and coverage gaps.
- Event case report: every relevant pre-event issue, source observations, warning timeline, track/footprint/time errors and misses/false alarms. Include non-events sampled by the preregistered protocol.
- Training report: frozen manifests, data hashes, splits, hardware/software, seed, hyperparameter search, checkpoint selection and test scores; failed candidates remain recorded.

Each report has a stable ID, schema/version, generated time, data watermark, exact cohort/query, input object hashes, code/model/matcher versions, sample counts and limitations. Keep JSON for machines, readable Markdown/HTML and compact plots. Store reports in the object archive with an index in SQL. Changes produce a new report version and a diff rather than silently modifying a published result.
