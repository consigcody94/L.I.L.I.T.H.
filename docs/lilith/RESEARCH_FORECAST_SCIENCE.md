# Forecast science and model research for Lilith

Research date: 2026-10-14. Scope: scientific basis for a global weather service, its 14-day forecasts and 90-day outlooks, and a critique of the experiments already recorded in this repository. This is a source-backed research and implementation decision note, not certification of operational forecast or warning skill.

**Evidence labels:** **Fact** means a claim directly supported by the cited primary paper or provider documentation. **Repository finding** means an observation from the local source/manifests, which has not been independently reproduced in this research pass. **Inference** means a conclusion drawn from those facts. **Recommendation** means a proposed Lilith decision, not a result already achieved.

## 1. Scientific conclusion that should drive the product

**Inference:** Lilith should be a portfolio: a global atmospheric forecast/reference layer; learned local correction and calibration; dedicated event models; and a separate subseasonal/seasonal outlook layer. Global coverage and uniform accuracy are separate properties. Station forecasts can improve a point product, but their existence does not demonstrate skill over oceans, unobserved terrain, or every hazard.

**Fact:** The successful global models reviewed here learn evolution of multivariable atmospheric states rather than only a location's temperature/precipitation history. GraphCast predicts hundreds of variables on a global 0.25-degree grid through 10 days; Pangu uses pressure-level structure and multiple temporal steps; GenCast generates probabilistic global trajectories through 15 days. Their papers report benchmark-specific improvements, not universal superiority for every location, event or application. Sources: [GraphCast author paper](https://arxiv.org/abs/2212.12794), [Pangu author paper](https://arxiv.org/abs/2211.02556), [GenCast paper](https://www.nature.com/articles/s41586-024-08252-9).

**Recommendation:** Keep the user's worldwide scope. Implement global spatial coverage from gridded products while publishing a coverage/skill mask for each variable, horizon and hazard. Never substitute “no verified skill” or “no observations” with “no risk.” Site records should distinguish the forecast's spatial completeness from the availability of independent verification.

## 2. Evidence from the principal model families

| Model family and primary source | What the research establishes | Implication for Lilith |
|---|---|---|
| [GraphCast, Lam et al.](https://arxiv.org/abs/2212.12794) | A graph neural network trained on reanalysis predicts global atmospheric evolution and supports derived cyclone, atmospheric-river and temperature-extreme evaluations. The reported 90% improvement over 1,380 verification targets is specific to the study. | A graph is useful when it represents physically relevant global state and training targets. A graph joining a small set of daily land stations is a different experiment. Use matching initialization, variables and metrics before comparison. |
| [Pangu-Weather, Bi et al.](https://arxiv.org/abs/2211.02556) | The authors trained on 43 years of ERA5 and use a 3D Earth-specific Transformer with hierarchical temporal aggregation, evaluating forecasts from hours to about a week. | Vertical atmospheric structure and accumulated rollout error matter. A successful Transformer paper does not validate any Transformer applied to weather. |
| [FourCastNet, Pathak et al.](https://arxiv.org/abs/2202.11214) | Adaptive Fourier neural operators produce global 0.25-degree short/medium-range forecasts, with fast inference enabling ensemble experiments. | Spectral operators are a plausible global representation; reproduce input/target contracts and baselines before adopting an architectural label. Fast inference is not evidence that full training is inexpensive. |
| [Spherical Fourier Neural Operators, Bonev et al.](https://arxiv.org/abs/2306.03838) | The paper addresses geometric artifacts of flat Fourier transforms on the sphere and demonstrates long stable atmospheric rollouts. | Treat spherical geometry as a real requirement. A stable year-long simulation demonstrates numerical/statistical behavior, not an accurate prediction of each actual day a year ahead. |
| [Huge Ensembles Part I, Mahesh et al.](https://arxiv.org/abs/2408.03100) | SFNO ensembles use specific initial-condition and model-uncertainty designs, including bred vectors and parameter/checkpoint choices; evaluation includes spectra, reliability and extremes. | Ensemble diversity must have a validated uncertainty mechanism. Adding independent noise to final outputs does not establish calibrated or physically coherent event probabilities. |
| [GenCast, Price et al.](https://www.nature.com/articles/s41586-024-08252-9) | Stochastic trajectories are produced at 12-hour steps and 0.25-degree resolution; the study evaluates marginal and joint forecasts, extreme weather and cyclone tracks. The reported 97.2% of 1,320 targets concerns that experiment. | Forecast probabilities need coherent trajectories and proper scoring. A deterministic mean can look good in RMSE while losing realistic sharp weather structures. Test joint event behavior as well as pointwise errors. |
| [NeuralGCM, Kochkov et al.](https://www.nature.com/articles/s41586-024-07744-y) | A differentiable dynamical core plus learned parameterizations supports weather and climate simulations. The climate experiments prescribe sea-surface temperature and sea-ice concentration. | Hybrid physics is a separate research path. Simulating climate statistics with prescribed historical boundaries cannot demonstrate a coupled, issue-time 90-day local forecast. |
| [FuXi, Chen et al.](https://www.nature.com/articles/s41612-023-00512-1) | A cascade of models addresses different portions of the 15-day forecast and is compared with ECMWF ensemble-mean forecasts. | Lead-dependent models and training objectives are reasonable candidates. Comparison with an ensemble mean is not comparison with the full probabilistic ensemble distribution. |
| [FuXi-S2S, Chen et al.](https://www.nature.com/articles/s41467-024-50714-1) | Global daily-mean state forecasts extend to 42 days using 72 years of ERA5 statistics and flow-dependent perturbations. Evaluation focuses on weekly/multiweek averages; statistically significant improvements are not uniform across all variables. | A serious subseasonal path needs its own targets, inputs and uncertainty evaluation. This evidence does not establish 90-day daily local temperature or event forecasts. |
| [FourCastNet 3 author preprint](https://arxiv.org/abs/2507.12144) | A later probabilistic approach emphasizes spherical geometry and fast global rollouts, including a reported 60-day inference example. | Track newer work, but distinguish preprint findings, integration maturity and forecast-skill evidence. The length of a generated rollout does not establish useful prediction skill at its final lead. |

**Recommendation:** Start from an interoperable global model/reference interface, not a permanent commitment to one architecture. Compare off-the-shelf forecasts, fine-tuning/adapters, and smaller local models on the same frozen benchmark. Fresh training of a global atmospheric model is a later experiment after data pipelines, reference replay and cost estimates are proven.

### Current-generation supplement: WeatherNext 2, Cyclones, Mini and WeatherNext 3

**Fact — probabilistic mechanism:** The [FGN technical paper, Alet et al. (2025)](https://arxiv.org/abs/2506.10772) trains learned functional perturbations with a marginal CRPS objective, combining multiple independently trained models. It models successive global states conditional on the preceding two states, with six-hour steps through 15 days. Training uses ERA5 followed by operational HRES-fc0 fine-tuning. The authors test marginal calibration and joint spatial structure rather than assuming that good marginal scores establish coherent joint events. This is substantively different from Lilith's fixed independent output-noise ensemble.

**Fact — cyclone evidence:** [Operational tropical cyclone forecasting with AI, Alet et al., Nature (published August 2026)](https://www.nature.com/articles/s41586-026-10953-2) describes WeatherNext Cyclones: global atmospheric analysis plus historical cyclone labels, with direct cyclone tracking and predictions of track, intensity and wind radii. Its study covers 2023–2025, with the 2025 operational evaluation described separately. Cyclone-specific heads and tracking are part of the system; these results are not obtained by treating ordinary grid-cell wind speed as a complete cyclone prediction. The paper includes a smaller 1-degree Mini model and its own evaluation. None of those author results are measured Lilith results.

**Fact — initialization/output contract:** The [official WeatherNext 2 guide](https://developers.google.com/weathernext/guides/models-wn2) specifies HRES-fc0 initialization, 0.25-degree fields, six-hourly cycles, 15-day forecasts and a 64-member feed. It lists surface and 13-pressure-level variables. Total precipitation is a six-hour accumulation in metres; a conversion to millimetres does not make it a daily total. The public feed, a checkpoint's supported inference configuration, and a paper's experimental ensemble size should be recorded separately.

**Fact — downloadable artifact choice:** The [official repository artifact/quick-start guide](https://github.com/google-deepmind/weathernext#provided-pretrained-models) lists separate WN2/Cyclones cutoffs and 1-degree Mini checkpoints. Mini is smaller and is not expected to match full-model performance. The guide recommends TPUs, requires an attention-backend change on GPU, and indicates P100 inference for Mini versus H100 memory for full versions. Current code/notebooks are Apache 2.0; other repository materials are CC BY 4.0. Inputs and external feeds retain separate terms.

**Recommendation — bounded Mini research candidate:** Mini is worth evaluating as an integration candidate before committing to a full model. A future small, explicitly bounded reproduction can establish that Lilith can acquire/version inputs, initialize the architecture, run the tracker, serialize ensemble outputs, preserve units and verify the same cases against truth/reference forecasts. First validate a supplied example and checkpoint reload, then a protected multi-case benchmark. Keep checkpoint training cutoff, tracker version, member count, grid, backend and exact source release in the run manifest. Do not launch or budget a major training job from this literature finding alone.

**Inference — what Mini cannot establish:** A successful smoke test proves execution, not forecast skill. Mini skill cannot be substituted for the full-model paper's scores, and full-model scores cannot be assigned to Mini. A coarse cyclone-oriented candidate does not by itself establish street-level rainfall/flood effects, storm surge, tornado guidance, global warning authority, or 90-day event prediction. Those need their own inputs, impact/event labels, reference comparisons and protected evaluations. Adaptation on Lilith observations also needs fresh validation; it may change the pretrained model's behavior.

**Fact — newer provider generation:** The September 2026 [WeatherNext 3 author preprint](https://arxiv.org/abs/2609.03582) adds raw satellite/precipitation/station observations and higher-resolution prediction heads. The [current official guide](https://developers.google.com/weathernext/guides/models) identifies WeatherNext 3 as the newer model: live geostationary mosaics plus HRES analysis, hourly initialization, multi-resolution outputs, and different forecast horizons for main versus interim cycles. Its feed terms distinguish current/future data from historical data; the latter becomes CC BY 4.0 after the documented one-hour threshold. These are feed terms, not a blanket license for every checkpoint/input/archive.

**Recommendation:** Include WeatherNext 3 in the reference/provider comparison while treating the downloadable WN2/Cyclones/Mini artifacts as distinct reproduction candidates. Verify actual access, variables, latency, costs and accepted terms before selecting a feed. Do not infer that WN3 weights or training data are available merely because a predecessor has downloadable checkpoints. For any model generation, ingest authoritative national warnings independently and record which forecast or event record supported Lilith's own risk statement.

## 3. What 14 days and 90 days should mean

**Fact:** NOAA CPC separates short/extended weather guidance from monthly/seasonal outlooks; the latter describe probabilities of temperature and precipitation categories over the target period. CPC also provides 6–10 and 8–14-day outlook products. [CPC product explanation](https://www.cpc.ncep.noaa.gov/products/outlooks/). Its three-month maps use probabilities for above, near and below normal categories, rather than a sequence of exact daily outcomes. [CPC three-category explanation](https://www.cpc.ncep.noaa.gov/products/predictions/long_range/seasonal_info.php).

**Inference:** There is no defensible single universal cutoff at which all weather information vanishes. Skill depends on variable, region, season, event, target averaging and prediction system. Conversely, an architecture accepting `forecast_length=90` does not supply evidence for 90 useful daily forecasts. NOAA's product definitions and the separate medium-range and S2S evaluations support separating those claims.

**Recommendation — product contract:**

| Range | Lilith output contract | Necessary qualification |
|---|---|---|
| Hours to several days | Weather values/distributions and validated local hazards at the supported temporal resolution | Timely initialization, compatible observations, variable-specific calibration; do not imply hourly support from daily training |
| Through day 14/15 | Daily/interval weather distributions, lead-specific confidence and event probabilities where validated | Same-issue NWP/ensemble comparison; per-lead scorecards and event-specific limits |
| Weeks 3–6 | Weekly means/totals, anomaly probabilities, broader risk outlooks | Dedicated S2S reforecasts and weekly verification; publish regional/seasonal skill |
| Out to 90 days | Monthly and three-month anomaly/category probabilities with explicit target windows | Separate seasonal hindcasts, calibration and trend-aware climatology; useful skill must be demonstrated locally |

**Recommendation:** The site can retain an expandable 90-day calendar, but distant daily entries should expose a clearly identified climatological distribution or aggregated outlook, not a precise invented daily trajectory. Separate authoritative warnings from Lilith risk guidance in the schema and presentation. A potential-hazard outlook should preserve its horizon, affected spatial support, evidence, uncertainty and any authoritative alert references.

## 4. Why station-only forecasting cannot carry the global service

**Fact:** GHCN-Daily is a collection of land-station observations with heterogeneous elements and station records; it is not a globally complete four-dimensional analysis. [NOAA GHCN-Daily documentation](https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf). ERA5 combines model data and observations to create a globally complete reanalysis, including upper-air, land and ocean-related quantities. [Copernicus ERA5 documentation](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).

**Inference:** A local 30-day TMAX/TMIN/PRCP sequence omits much of the dynamical state controlling incoming weather: pressure, humidity, winds and their vertical/spatial structure. Sparse station interpolation cannot reconstruct unknown atmospheric information merely by assigning values to ocean cells. This limits the current model's scientific claim even when it beats persistence.

**Recommendation:** Use station observations for local correction and independent surface verification, global analyses/forecasts for atmospheric context, and hazard-specific observation/label systems for event targets. Include terrain, coast/land masks and point-to-grid representativeness in local calibration. For location scorecards, preserve the station-grid matching method, elevation difference, distance and valid observing interval.

## 5. Historical data before 1980 and availability at issue time

**Fact:** ERA5 currently extends to 1940, so an assumed 1980 global-data boundary would discard available historical material. ERA5's observing-system information content changes over time; ERA5T normally has about a five-day latency and can differ from final data released two to three months later. Reanalysis has more time to gather and improve observations than a live forecast does. [Copernicus ERA5 catalog](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview).

**Fact:** WeatherBench supplies cloud-optimized datasets at several spatial resolutions, but individual snapshots do not automatically cover the complete current provider archive. Its data guide lists versioned datasets and distinguishes their terms. [WeatherBench data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html).

**Recommendations:**

1. Inventory the actual earliest/latest usable date by station, grid variable, pressure level, provider and release, instead of applying one global cutoff. Retain older station history when the source and QC support it.
2. Compare recent-era and longer-history training on the **same** frozen tests. More years can increase regime/rare-event coverage while also introducing observation changes and climatic nonstationarity; the net forecast benefit is an empirical question.
3. Preserve measurement time, accumulation period, source publication/availability time, ingestion time and release/version. Final retrospective data can be used for research targets, but a hindcast must never receive data unavailable at its simulated issue time.
4. Distinguish an analysis-seeded research experiment from a production replay. A live model cannot normally initialize on today's final ERA5 state when its provider publishes that state days later. Use a supported real-time analysis/forecast input contract and measure that transition.
5. Retain source quality flags, explicit missingness masks and station metadata history. Causal imputation parameters and all climatology/calibration statistics must be fit within the permitted folds.

## 6. Direct critique of the completed Lilith experiments

**Repository evidence:** These numbers come from [training_runs.json](training_runs.json) and [TRAINING.md](TRAINING.md), not from independently rerunning the jobs. Feature order is TMAX °C, TMIN °C, PRCP mm. Entries report aggregated RMSE across target days and cases, so they are not “day-14 RMSE” or “day-90 RMSE.”

| Run | Actual test stations | TMAX RMSE: model / climatology | TMIN RMSE: model / climatology | PRCP RMSE: model / climatology |
|---|---:|---:|---:|---:|
| 14-day Transformer, 72 selected stations, 1991 onward | 17 | 4.578 / 4.134 | 4.168 / 3.865 | 6.728 / 6.663 |
| Kaggle reproduction of that 14-day experiment | 17 | 4.489 / 4.134 | 4.133 / 3.865 | 6.728 / 6.663 |
| 14-day Transformer, 144 selected stations, 1980 onward | 42 | 4.400 / 4.447 | 4.534 / 4.563 | 7.645 / 7.597 |
| 90-day Transformer, 72 selected stations | 14 | 5.241 / 4.256 | 4.663 / 3.986 | 7.190 / 7.069 |
| 14-day anomaly ridge, 144 selected stations, 1980 onward | 42 | 4.235 / 4.447 | 4.375 / 4.563 | 7.595 / 7.597 |
| 90-day anomaly ridge, 72 selected stations | 14 | 4.229 / 4.256 | 3.954 / 3.986 | 7.074 / 7.069 |

**Inferences and limitations:**

- The initial Transformer and its Kaggle reproduction are worse than empirical climatology on all three aggregate targets. Reproduction supports the broad experimental conclusion; it does not demonstrate operational accuracy or bitwise equivalence across devices.
- The 1980/144-station Transformer is modestly better on temperature but worse on precipitation. Since both station cohort and historical start changed, this is **not an isolated test proving that adding pre-1991 years improved the model**.
- The 90-day raw Transformer has negative MSE skill versus climatology for all three targets; TMAX MSE skill is approximately −51.7% and TMIN approximately −36.9%. Negative skill here means greater squared error relative to that baseline, not a negative probability or fraction of correct days.
- The simple anomaly regression is the strongest recorded aggregate temperature candidate on its matching station benchmarks. Its near-zero precipitation skill and small 90-day temperature gain do not establish useful rare-event or seasonal guidance.
- Selected station counts greatly exceed contributing test-station counts. Complete-window requirements preferentially exclude missing-data records; global sampling of station IDs is not worldwide validation.
- Held-out-station normals in `training/prepare_global.py` come from the nearest training station. The large held-out improvement over that transferred baseline may partly reflect a weak comparator. Future spatial evaluation should compare several legitimate cold-start climatology methods and archive the matching distances; do not use held-out future observations to improve the baseline.
- Aggregating overlapping target windows gives many dependent predictions. Count unique stations, issue cycles, valid days and independent event/temporal blocks. Quantify paired skill uncertainty using blocks, rather than treating each lead/case as independent.
- Existing temporal partitions and input/target boundary purging are valuable. However, inspected benchmark results have already informed later model choices. A fresh protected/prospective audit remains necessary; calling each subsequent evaluation “test once” would overstate independence.
- No same-issue global NWP comparison, calibrated probability qualification, wind/humidity/upper-air training, ocean verification or hazard-specific event qualification appears in these recorded runs. All recorded candidates correctly remain `research_not_promoted`.

## 7. Architecture and implementation findings requiring Astra review

**Repository findings:**

1. `models/simple_lilith.py` is a single-station encoder/decoder Transformer with metadata. It accepts a target length but does not ingest a global multilevel atmospheric state. The causal attention mask on decoder queries does not mean it feeds physically predicted weather states forward in the manner of GraphCast/GenCast rollouts.
2. `models/lilith.py` provides station-to-grid interpolation and a spectral/graph/temporal architecture, but that does not make its latent grid an assimilated global atmospheric state. Keep this architectural prototype separate from claims about the actual trained SimpleLILITH checkpoints.
3. `models/lilith.py::generate_ensemble` currently takes a deterministic forecast and appends `torch.randn_like(base_forecast) * 0.1` perturbations. The comment “calibrated noise” is unsupported by a fit/calibration procedure in that method. These members cannot be advertised as a qualified uncertainty product.
4. `models/components/sfno.py` falls back from spherical harmonics to planar FFT operations when `torch_harmonics` is unavailable. That changes the representation. Record the backend in every artifact and do not imply equivalence to the SFNO literature. The spherical branch's tensor layouts/weights should be checked against the actual `torch_harmonics` version before any expensive run; this research pass did not execute that branch.
5. The configured grid and interpolation boundary handling need explicit pole, periodic-longitude, coordinate-order and area-weighting tests. Grid dimensions alone do not specify physical resolution or useful effective forecast detail.
6. `training/train_residual_baseline.py` adds lead-specific fitted anomalies to empirical normals and constrains precipitation/temperature ordering. It is a useful benchmark. Its feature construction subtracts the first target-day normal from recent observations; test a scientifically cleaner ablation using each historical observation's own seasonal normal, holding all splits fixed.

**Recommendation:** Prioritize issues 1–3 and benchmark validity before scaling the prototype. Use uncertainty heads trained and evaluated against appropriate probabilistic objectives, followed by held-out calibration. Verify physical ordering and accumulation definitions on the exact served outputs. For precipitation, compare occurrence/amount or quantile/distributional models rather than relying only on mean-squared error across zero-inflated totals.

## 8. Verification and retained scientific records

**Fact:** WeatherBench evaluates deterministic and probabilistic forecasts with explicit baselines and multiple metrics; the current project points users toward WeatherBench-X while retaining its dataset guide. Its documentation includes RMSE/MAE/bias/ACC, precipitation SEEPS and ensemble CRPS. [WeatherBench framework](https://github.com/google-research/weatherbench2), [metric API](https://weatherbench2.readthedocs.io/en/latest/api.html).

**Recommendation — Lilith verification contract:**

- Archive every issued forecast immutably with issue time, valid interval, lead, spatial support, variable/unit, model/checkpoint/code hash, input dataset manifest, reference forecast cycle, ensemble/calibration version and output artifact checksum.
- Keep the first observed truth and subsequent revisions as separate records. Score both an agreed operationally available truth and a final-QC view where appropriate; do not quietly rewrite previous reports.
- Report per variable and lead: MAE/RMSE/bias, climatology-relative and same-issue NWP-relative skill, sample count and coverage. Add calibration/reliability, CRPS or quantile scores, interval coverage and sharpness for probabilistic forecasts.
- For threshold hazards, report Brier score/skill, reliability, precision/recall, false-alarm ratio and probability of detection using a declared event window, spatial support and matching tolerance. Retain all denominators. Rare-event skill cannot be inferred from average temperature RMSE.
- Add spatial/event diagnostics where displacement matters, plus regional/seasonal/extreme-case reports. Use area weighting for grid-global summaries and disclose station/network weighting separately.
- Define claim/promotion criteria before scoring the protected test. Preserve failures and sparse/no-truth regions on the public scorecard. Compare candidates on identical cases and publish block uncertainty on paired skill differences.

**Recommendation — storage implication:** Store forecast/observation metadata, run status and searchable score aggregates in a modest relational database; store chunked atmospheric arrays, immutable observation partitions, checkpoints and case-level predictions on the NAS/object-compatible archive. Parquet suits station/case tables; Zarr/NetCDF suit scientific multidimensional arrays. This is an architectural recommendation, not a measured capacity guarantee. Proxmox/NAS locality removes a transfer bottleneck but does not reduce the underlying data volume. Budget dimensions × variables × levels × time steps × members × retained issue cycles before ingesting full global archives.

## 9. Availability and licensing constraints

**Facts checked against first-party sources:**

| Artifact/source | Verified constraint | Lilith action |
|---|---|---|
| [Pangu official repository](https://github.com/198808xc/Pangu-Weather#license) | Trained parameters are BY-NC-SA 4.0; the repository explicitly forbids commercial model use. | Use only in a compatible research scope unless separately licensed. Do not assume public download permits deployment. |
| [NeuralGCM official repository](https://github.com/neuralgcm/neuralgcm) | Code is Apache 2.0; weights are CC BY-SA 4.0. | Track code and weight obligations separately. |
| [GraphCast repository URL, now redirecting to WeatherNext](https://github.com/google-deepmind/graphcast) | The inspected current repository distinguishes code and other materials. Repository identity/content has changed since the original GraphCast paper. | Pin an exact implementation commit, checkpoint URL/hash and accompanying license. Do not apply a current repository license retroactively to an unidentified legacy checkpoint. |
| [WeatherBench dataset guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) | Dataset-specific licenses differ; some datasets permit only research use. The guide flags TIGGE ensemble material as research-only. | Copy exact per-dataset license files and provenance into the NAS catalog; keep research benchmarks distinct from deployable inputs. |
| [Copernicus licensing announcement](https://forum.ecmwf.int/t/cc-by-licence-to-replace-licence-to-use-copernicus-products-on-02-july-2025/13464) | CDS/ADS/EWDS changed from the prior Copernicus license to CC-BY in July 2025. | Record the catalog/license accepted for the actual source/version and provide its required attribution. Do not infer that every mirror/baseline has identical terms. |

**Recommendation:** “Free government API” should mean verified access terms, availability and coverage, not a blanket assumption that every useful global input is government-run, unrestricted, timely or homogeneous. Dataset access accounts, historical archives, live reference forecasts and pretrained model weights are four separate acquisition contracts. Training on a provider archive does not authorize arbitrary redistribution of that archive.

## 10. Prioritized research and training roadmap

This order builds a worldwide product and evidence system without shrinking the geographic requirement. Phases are model/data qualification stages, not permission to claim that pilot regions represent all locations.

| Priority | Work to perform | Required deliverable and advancement gate |
|---|---|---|
| P0: freeze evidence | Preserve existing manifests, raw-case predictions, training runs, code and the negative results. Build a capability matrix by variable/horizon/region/hazard. | Reconstructable run catalog; no promoted model or unsupported alert/90-day claim. |
| P0: fix benchmark contract | Add issue-time availability, real-time/reference initialization, independent protected time/region folds, per-lead metrics, baseline cold-start audit and block uncertainty. | One reproducible paired reference replay before another architecture search. |
| P1: global inputs and NAS catalog | Inventory NOAA and other agency observations plus chosen multilevel global analysis/reference products, with coverage, releases and licenses. Stage selected variables/levels/resolutions and observation partitions. | Validated worldwide input contract, explicit no-truth masks, measured bytes/throughput and finite retention budget. |
| P1: local correction | Fit seasonal normals, anomaly ridge, simple tree/linear baselines, then lead-specific NWP residual and probabilistic models using the same cases. Add elevation/coast context and missingness masks. | Temperature/precipitation skill and calibration relative to the same-issue reference on unseen folds, with geographic failures exposed. |
| P1: uncertainty | Fit quantile/distributional models and held-out calibration. Qualify ensemble-member and aggregate behavior. | Reliability, coverage/sharpness and proper-score evidence; fixed output noise is retired as a scientific uncertainty claim. |
| P2: global model adaptation | Reproduce a suitable licensed global model at supported initialization/resolution; compare frozen outputs, local adapters and fine-tuning. | Checkpoint/license manifest, multiple-case physical sanity, equal-case skill and an actual measured compute/storage budget before scaling. |
| P2: hazard modules | Define and source independent event labels; train separate heat/cold/heavy-rain/wind/etc. models with fit-for-purpose spatial and temporal predictors. | Event-specific verification and alert provenance; no hazard promoted solely from temperature skill. |
| P2: S2S | Use archived reforecasts, initial ocean/land/snow state and released climate predictors; fit weekly anomaly probability baselines before complex models. | Weeks 3–6 lead/region/season scorecards and calibration, with simple/reference baselines. |
| P3: seasonal 90-day | Train distinct monthly/three-month probabilistic products with hindcasts and drift/trend controls. | Protected seasonal skill and reliability; explicit target averaging window and climatology reference. |
| Continuous: shadow operations | Archive issuance, source lags, realized truth and revisions; generate retained reports, failures and regressions. | Prospective evidence showing the offline result survives actual issue-time data and service outages. |

**Recommendation for Colab/Modal/Kaggle:** Keep preparation and training separate, freeze a portable data snapshot and source commit, use explicit accelerator/time/retry/output limits, and return checkpoints plus complete case-level evaluations to the NAS. Use independent provider reproduction to test portability. No listed cloud provider or available GPU changes the scientific qualification gate, and no additional cloud job was launched for this research note.

## 11. Astra refinement handoff

Decisions to challenge/refine before substantial new training:

1. Choose the initial real-time global state/reference contract and verify it matches any reused model's expected fields, levels, normalizations, timestep and grid.
2. Redesign the site/API capability schema so weather distributions, weekly/monthly outlooks, Lilith hazard probabilities and authoritative alerts carry distinct provenance and validation status.
3. Audit spatial holdout baselines, missing-data selection and overlap-dependent uncertainty; retain all existing results as development evidence.
4. Select a licensed global-model reproduction/adaptation candidate on practical cost and reproducibility grounds, without claiming the paper's benchmark skill transfers to Lilith.
5. Remove the inference-time fixed-noise ensemble claim and specify calibrated probabilistic training plus separate calibration data.
6. Specify NAS dataset layout, immutable lineage, retention, checksums, download recovery and local/cloud subset staging before large archive acquisition.
7. Pre-register model-promotion claims and verification gates by variable/horizon/region/hazard, and identify a genuinely fresh prospective audit period.

Open scientific questions remain: gains from older history versus climate drift; effective geographic coverage with missingness-aware training; benefit of a learned atmospheric model beyond reference-based local calibration; and which S2S/seasonal products retain useful local skill. These require experiments and prospective verification. This literature pass supplies a defensible research direction; it cannot resolve them by assertion.
