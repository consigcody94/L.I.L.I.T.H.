# Research decisions for Lilith and Astra

Reviewed 2026-10-14. This synthesis connects the primary-source literature to the existing code and completed experiments. It separates decisions from demonstrated model capabilities. The detailed evidence, access limitations and citations are in [forecast science](RESEARCH_FORECAST_SCIENCE.md), [verification and hazards](RESEARCH_VERIFICATION_AND_HAZARDS.md), and [NAS/Proxmox storage](NAS_AND_PROXMOX_RESEARCH.md).

## Product and model decisions

| Requirement | Research conclusion | Concrete Lilith decision |
|---|---|---|
| Worldwide weather | Successful global models evolve multivariable atmospheric states; sparse daily land stations do not establish ocean, upper-air or severe-event skill. | Keep worldwide scope. Combine a global atmospheric reference/state with point correction, probabilistic calibration and dedicated hazard modules. Publish forecast, observation and warning coverage separately. |
| Detailed forecasts and 90 days | Medium-range trajectories and subseasonal/seasonal outlooks have different targets and evaluation protocols. | Serve detailed supported leads separately from weekly/monthly anomaly probabilities. A 90-element array is not evidence of useful day-specific skill three months ahead. |
| Older weather records | Available history predates 1980, but observation density, instruments and climatic regimes change. | Inventory earliest usable records per variable/platform; preserve source history. Compare longer-history and recent-era fits on identical protected cases before attributing improvement to more years. |
| Professional alerts | Official warnings carry issuing authority and lifecycle semantics; an ML risk probability is a separate product. | Implement authoritative CAP intake, references/update/cancel state, geography, freshness and audited delivery. Keep model risk messages attributed and separately qualified. |
| Accuracy everywhere | Forecast coverage can be global while measurement coverage is sparse. Average temperature error does not establish event skill. | Report per variable/lead/region/season, same-case baseline skill, coverage and uncertainty. Explicitly show missing truth and unsupported hazards. |
| Modest database and NAS storage | Relational metadata can stay compact while arrays dominate bytes. A dense global hourly archive can exceed available space. | Use local SQL for catalogs, lifecycle and queryable aggregates; immutable scientific arrays, source snapshots, checkpoints and reports on the NAS. Measure actual retention and compression before bulk ingestion. |

Primary examples supporting the first two decisions: [GraphCast](https://arxiv.org/abs/2212.12794), [GenCast](https://www.nature.com/articles/s41586-024-08252-9), [FuXi subseasonal research](https://www.nature.com/articles/s41467-024-50714-1). These papers establish their own experimental results; they do not establish Lilith's skill. Dataset/history support: [NOAA GHCN-Daily](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily), [Copernicus ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview). Verification protocol reference: [WeatherBench](https://weatherbench2.readthedocs.io/en/latest/).

## Changes the source review makes urgent

1. **Hourly ingestion must target current GHCNh.** NOAA identifies it as the replacement for ISD and offers PSV/Parquet bulk data. Pin the schema and variable-specific QC/accumulation semantics before adapting the legacy downloader. [NOAA product documentation](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly).
2. **Historical truth is versioned and delayed.** Preserve valid intervals, source availability, retrieval time and corrections. Retrospective final data can train a model, but cannot silently become an input available at an earlier simulated issue time. Observation-based and reanalysis-based evaluation remain separate. [ECMWF ERA5 documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=388500357).
3. **Hazard schemas need dated versions.** SPC added Conditional Intensity products in March 2026. Keep the older and current interpretations tied to product dates. Storm reports and disaster databases also have reporting/maturity biases; absent reports are not automatic negative labels. [NOAA SPC announcement](https://www.weather.gov/news/262402-spc), [NOAA Storm Events details](https://www.ncei.noaa.gov/stormevents/details.jsp).
4. **Ocean and upper-air validation need their own observations.** Add buoy/marine and radiosonde cohorts with platform-specific measurement support. GHCN land station scores cannot substantiate performance at sea or aloft. [NOAA ICOADS](https://www.ncei.noaa.gov/products/international-comprehensive-ocean-atmosphere-data-set), [NOAA IGRA](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive).
5. **Model availability must be pinned to actual artifacts.** Research now includes newer WeatherNext work as well as GraphCast, Pangu, FourCastNet, GenCast, NeuralGCM and FuXi. Pin source commit, inputs, checkpoint hash and artifact-specific license before any reproduction. New papers or public weights do not automatically make a model suitable for Lilith's deployment. See the detailed model review for current-generation sources and access boundaries.
6. **Live SQLite WAL stays off the NAS share.** The current pilot needs a true local filesystem; moving a share into a container does not change this. Use consistent database snapshots for archive copies. A NAS bind mount also needs a backup inventory separate from guest backups. [SQLite WAL](https://sqlite.org/wal.html), [SQLite backup API](https://www.sqlite.org/backup.html), [Proxmox container documentation](https://github.com/proxmox/pve-docs/blob/master/pct.adoc).

## What the completed experiments support

Seven real fits and their raw case predictions are retained. The raw 72-station Transformer experiments underperform their same-case climatology; the larger 1980-based cohort has small temperature gains, with precipitation still weak. Anomaly regression is the strongest temperature baseline among these recorded candidates. All remain `research_not_promoted`. See [the handoff](ASTRA_HANDOFF.md) and [measured results](training_runs.json).

The 144-station experiment changes both the station cohort and historical start date, so it does not isolate the value of pre-1991 history. Only 17, 14 and 42 stations contribute to the respective temporal test cohorts; selection through complete windows is severe. Held-out-location normals also need a fair cold-start comparison. The benchmark informed later choices, so another pass over it is development evidence, not a new untouched final test.

## Next implementation and training contract

1. Freeze the worldwide variable/horizon/hazard capability schema, observation intervals and real-time global input contract.
2. Build missingness-aware station preparation, issue-time NWP predictors/baselines and protected temporal, regional and whole-event folds. Register a fresh prospective audit before model selection.
3. Train local residual and distributional baselines first, using identical eligible cases. Report physical violations, paired baseline differences, proper probability scores and blocked uncertainty. Separate train/calibrate/evaluate data.
4. Reproduce a practical licensed global model with its exact supported fields, grid, initialization and hardware budget; compare frozen inference, adapters and fine-tuning rather than assuming training a global model from scratch is necessary.
5. Define hazard labels and non-events, then train/evaluate hazard-specific models. Qualify warning ingestion and offline delivery independently of ML skill.
6. Publish bounded, checksummed datasets and model/report manifests to the NAS; run outage, interrupted-copy and restore drills before sustained operation. Production deployment remains future work.

Astra should refine these contracts and inspect the actual arrays before launching a larger architecture search. The review supplies a scientific direction and specific implementation gates; unresolved empirical questions remain experiments to perform.
