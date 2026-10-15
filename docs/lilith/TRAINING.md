# Training and model qualification

## Work already executed

See [training_runs.json](training_runs.json) for measured results, manifests and hashes. All candidates remain **research_not_promoted**. Training completed is not a claim of operational readiness or uniformly global skill.

Experiments train the actual shared SimpleLILITH Transformer on NOAA GHCN daily values, not synthetic weather. Initial cohort: 72 geographically stratified station IDs, 1991–2025. Extended cohort: 144 station IDs with 1980–2025 history. The manifest records actual complete days and usable windows; many selected stations contribute no complete 30+target-day windows. Selected station count must not be presented as the count of independently validated locations.

Initial partitions: train 1991–2016, validation 2017–2020, test 2021–2025; the extended run starts training in 1980 with the same later boundaries. Entire input+target windows must remain in one time partition. Station holdouts are removed from training and validation; their later data is evaluated separately. They are a station holdout, not a rigorous contiguous-region exclusion.

Raw Transformer output is compared with persistence and **empirical training-period seasonal normals**, with no live NWP forecast blended into the experiment. The separately fitted anomaly-regression baseline uses regularization selected on validation only. Refinement after inspecting the original Transformer test results reuses a benchmark; a fresh prospective audit is still required.

Both 14- and 90-day Transformer candidates were trained on an NVIDIA L4 through private, bounded Modal jobs. Full checkpoints, per-case predictions, truths, baseline arrays and detailed reports were downloaded. Kaggle runs an independent reproduction against a frozen private benchmark snapshot. Colab availability was checked; the existing unrelated live session was not modified.

## Local and cloud commands

The existing full project dependencies are heavy and include packages unnecessary for these experiments. From a repository checkout, create an isolated environment and install a minimal set:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy==2.2.6 torch==2.8.0
.venv/bin/python -m training.prepare_global --output work/benchmark --cache work/noaa-cache --stations 144 --horizon 14 --start-year 1980
.venv/bin/python -m training.train_verified --data work/benchmark --output work/model-run --epochs 30
.venv/bin/python -m training.train_residual_baseline --data work/benchmark --output work/residual-run
```

On Modal:

```bash
uv run --with modal modal run training/cloud/modal_verified.py --horizon 14 --stations 144 --epochs 30 --start-year 1980 --run-name global-h14-1980-20261014
```

This creates no public endpoint or recurring job. Preparation uses CPU; fitting allocates one L4 and has explicit time/epoch limits and no application retries. The app stops after completion; the private research volume retains artifacts. Create the local destination directory before downloading a directory with the Modal CLI; the downloaded remote directory is nested inside it. Verify manifest hashes and array shapes after retrieval.

Use unique run names and snapshot dataset directories for all future runs. The original launcher reused `datasets/h14` for the two 14-day cohorts, so the original dataset is preserved locally and in the private Kaggle snapshot; its original reports/manifests remain immutable. The launcher now uses unique dataset directories and rejects completed-name reuse to prevent future collisions.

For Colab, run the same modules in a dedicated new session/notebook, export artifacts before stopping **only the session you created**. For Kaggle, use a private script kernel with GPU and a frozen dataset containing source plus the four prepared splits. Provider downloads and model fitting are independent stages, so a Kaggle reproduction needs no internet. Pin/record framework versions; bitwise equivalence across CUDA devices is not assumed.

## Final training data system

A station-only model cannot provide a physically complete atmospheric forecast over oceans or accurately localize every severe event. Use a layered model portfolio:

| Model family | Inputs/targets | Role |
|---|---|---|
| Empirical climatology and persistence | Historical station/grid values and real issue-time history | Mandatory baselines and honest low-skill fallback |
| Lilith point residual model | Station history, terrain, global/regional NWP fields and lead-specific seasonal normals | Improve calibrated local weather relative to reference NWP |
| Global atmosphere model or adapted existing weather model | Versioned multilevel atmospheric, land and ocean state | Globally complete dynamics; expensive research track with separate licensing/compute review |
| Subseasonal/seasonal model | Climate modes, SST/soil/snow state, reforecasts and weekly/monthly targets | Probabilistic 15–90 day outlooks |
| Hazard-specific models | Appropriate predictors and verified event labels | Heat/cold, heavy rain, wind, snow/ice, flood and convective/tropical risks |
| Calibration/blend layer | Out-of-fold model/reference forecasts and matured truth | Reliable distributions and location/lead/season-dependent model weighting |

Do not train a single temperature-only architecture and label its output tornado, hurricane, flood or storm-surge prediction. Each hazard has distinct labels, spatial/temporal support and observing limitations.

## Data and leakage controls

- Historical sources extend before 1980 where available. Compare full-history and recent-history fits with era-specific QA and climate drift handling. Keep enough decades for rare-event training without leaking future information.
- Reindex to complete real calendars. Missing inputs need explicit masks and a causal imputation policy fit on training data; missing labels never become target means. Complete-window filtering is acceptable for a pilot but its geographic selection bias must be reported.
- Fit scalers, seasonal normals, graph topology/data-derived covariates, feature selection, sampling weights, downsampling and calibration using allowed training folds only.
- Use observations actually **available at forecast issue time** for operational replay. Current GHCN final historical values are retrospective; some arrive too late for next-day operational seeds. Publication/ingestion lag is a required follow-up contract.
- For climate indices, satellite/reanalysis and NWP, enforce original release/version availability. A later revised ENSO value or an analysis containing future assimilation cannot enter an earlier simulated issue.
- Purge overlapping target windows at boundaries; group by issue cycle and event. Do not randomly split overlapping station sequences. Validate held-out locations and contiguous regions in addition to held-out years.
- Freeze immutable data and code snapshots before evaluation. Record source file hashes, selected/excluded station IDs, missingness, units, normalization, training windows and geographic/seasonal coverage.

## Evaluation stages

1. Offline parser/unit/calendar checks, one-batch CPU reference and GPU finite-loss/checkpoint-reload tests.
2. Baseline benchmark before architecture search, per variable/lead/region/season and same-case comparisons.
3. Rolling-origin backtests across multiple eras, recently withheld years, spatial holdouts, extreme-event holdouts and geographically balanced sampling.
4. Validation-only architecture/hyperparameter selection; then a frozen test. Maintain a fresh future shadow period once benchmark results have influenced design.
5. Probability calibration on separate held-out calibration data, followed by unseen validation of interval coverage, CRPS/Brier/reliability and rare-event false alarms.
6. Prospective shadow inference with real availability times, immutable issuance archive and delayed station verification.
7. Promotion decision based on reproducible skill/coverage, reliability, forecast freshness and serving cost. Maintain rollback to previous champion/reference.

Require improvements relative to **reference NWP**, not merely a hand-coded seasonal function, before claiming improved operational weather forecasts. Use block confidence intervals; small global-average improvements can conceal regional failures. No hard universal RMSE threshold is pre-approved here: establish tolerances per variable, region, horizon and application.

## Heavy compute progression

Use cheap pilots to expose leakage and weak baselines before large GPU allocations. The completed experiments reveal the raw Transformer is weak relative to climatology, especially at 90 days. More epochs or a bigger GPU alone do not prove the architecture is appropriate.

The next substantial jobs should prepare a broader, coverage-audited global station catalog; retain variable masks to avoid excluding most tropical/precipitation cases; add issue-time NWP predictors; train anomaly/residual and uncertainty models; and compare to archived same-issue reference forecasts. Climate-index and full-model work from PR #2 should be reconciled and scientifically audited before scaling.

Keep per-run limits on GPU type, wall time, attempts and artifact volume. Resume only from a recorded checkpoint. Record actual hardware/framework/precision and service run IDs in reports. A skill file’s listed credit balance is not an account reading; query actual availability before allocating work.
