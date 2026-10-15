# Repository audit

Reviewed origin/master at `10a3a00`, repository `consigcody94/L.I.L.I.T.H.`. Existing open [PR #2](https://github.com/consigcody94/L.I.L.I.T.H./pull/2) contains separate model/data work. It was inspected but is not merged or treated as verified implementation here. Reconcile both branches before integration.

## Existing behavior and implications

| Evidence at the reviewed revision | Finding | Consequence |
|---|---|---|
| `training/train_simple.py`, `GHCNProcessor.save_training_data` | Random sample split after fitting normalization over all samples; windows overlap | The reported validation RMSE does not establish future forecast performance |
| `training/train_simple.py` | Target sequence is 14 days | A 90-day decoder capacity is not evidence of 90-day training or skill |
| `data/training_stations.json` | 505 IDs are prefixed US; ordering begins with many Alabama stations | Station count is not globally representative sampling; inspect actual geographic distribution |
| `data/processing/ghcn_processor.py` | Rows are dropped before sequence construction, gaps can become adjacent days; linear interpolation can cross time boundaries | Training windows need an explicit calendar, missingness masks and as-of data rules |
| Same processor | SNOW and SNWD are divided by 10, although GHCN defines them in mm | Both the legacy processor and new decoder now use correct units; rebuild affected snow-derived artifacts |
| `models/simple_lilith.py` | Shared point Transformer exists and can train | Useful baseline; does not encode full atmospheric state or storm dynamics |
| `models/lilith.py` | Larger GAT/SFNO architecture exists | It is a research architecture, not a trained operational global weather engine |
| `inference/forecast_blend.py` | “Climatology” is a hand-coded latitude/region seasonal formula; fallback input is synthetic | It is a heuristic, not measured station normals; deterministic output can still be synthetic |
| `inference/simple_forecaster.py` | Model output is adjusted and blended with a third-party near-term forecast | Serving skill must be decomposed into raw model, reference, empirical climatology and blend skill |
| `web/api/main.py` | Daily predictions in process memory, seven leads retained, 5,000-row eviction | Restart loses forecasts; most of the claimed 90-day horizon cannot be audited |
| `data/providers/open_meteo.py` | Archive/recent forecast data returned under a `daily_observations` name | These are modeled historical estimates or old forecasts; not independent station truth |
| `web/api/main.py` | Hourly fallback uses random temperature and arbitrary humidity/wind; model path fills unmodeled fields with constants | User-facing output misrepresents model capability |
| `web/frontend/src/app/historical/page.tsx` and historical components | Mock climate trends, random precipitation, hard-coded records | Selecting a new place does not produce genuine historical records |
| API historical/ensemble endpoints | Both return 501 | README feature descriptions exceed implementation |
| Accuracy report | Requested `days_back` labels a period without filtering underlying records | Published accuracy can describe the wrong time period |
| Accuracy UI and uncertainty chart | Countdown implies station checks; heuristic bands are called 95% confidence | Verification provenance and probability calibration are overstated |
| Docker compose | Redis and Timescale containers exist but API is not wired to them | Declaring a database service does not make forecast storage durable |
| Packaging | Core install includes CUDA-oriented `xformers`; wheel includes only `lilith`; CLI entrypoint absent on master | Clean installation/packaging needs independent repair; new experiment instructions use explicit minimal dependencies |

## Changes in this branch

Daily forecasts now persist through a SQLite ledger before a successful response. Full returned payloads preserve heuristic ranges and per-lead source labels; hashes link loaded model weights. Issuances replay without recomputation. Hourly NWP reference payloads also persist but do not enter daily accuracy calculations.

Modeled comparison records are appended separately from immutable forecasts. Truth corrections retain their earlier versions. A report read does not trigger network calls. Reference comparisons remain explicitly labeled as modeled reference agreement, with zero independent-station verifications.

The historical endpoint fetches actual GHCN station files, retains the source digest and filters missing/QC-rejected values. The historical page renders available observations and completeness counts. Original mock historical components remain unused; replace/remove them when implementing final charts.

The new training path retains real dates, excludes incomplete windows rather than fabricating targets, separates time periods, excludes station holdouts from fitting and fits normalization on the training split only. It is a bounded research cohort, not the final global dataset.

## Known issues still requiring refinement

- Legacy SimpleForecaster can seed itself from synthetic history and uses heuristic serving climatology. Remove this fallback or explicitly expose it before production. The new experiments do not use that path.
- The full-model embedding shape, decoder axis/rollout stability and shared-graph batch offsets are fixed here and regression-tested. Reconcile overlap with PR #2 before integration. The decoder adds LayerNorm parameters; old full-model checkpoints need an explicit migration/retrain. No full-model weights were trained or promoted here.
- Existing naive server timestamps are interpreted as UTC by the pilot archive; serving and provider daily aggregation need explicit local-day contracts, including DST and station reporting windows.
- Existing API options such as start date, variable selection and ensemble member count are not all implemented. Unsupported options must be rejected or implemented.
- Per-variable partial verification, model/version-specific operational scores and bounded SQL aggregation need further work. Current daily summary metrics include only complete three-variable reference comparisons.
- SQLite pilot reports materialize matching records in memory. This is unsuitable for global volumes; implement SQL/columnar aggregations and the compact statistics design.
- Automatic comparison processes one modeled-reference version; provider correction reconciliation and watermarking are future jobs. Missing-data retry fairness needs a durable work queue so unavailable records cannot starve others.
- Public alert delivery has not been implemented. CAP parsing is not authorization, official-source authentication, spatial matching or lifecycle processing.
- The site and Docker deployment require production configuration and dependency review before exposure. Deployment is intentionally deferred until the user supplies hosting.
