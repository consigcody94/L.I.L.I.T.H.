# Lilith research: observations, independent verification, extremes and warnings

Research date: **2026-10-14 (America/Chicago)**. Scope: global land/ocean weather, retrospective records including periods before 1980, honest comparison to observed conditions, and a professional alert service. Sources below are NOAA/NWS/NCEI, ECMWF, WMO, standards owners, original papers, or the papers' authors. Links sit next to the claims they support. This is a research/design note, not evidence that a listed connector or verification pipeline is implemented.

## 1. Conclusions that change implementation priorities

**Evidence.** NOAA now identifies **GHCN hourly (GHCNh) as the replacement for ISD**, aligns station identifiers with GHCN daily where possible, and supports bulk PSV and Parquet. An hourly ingestion design should therefore use current GHCNh documentation, with ISD retained as a historical source lineage rather than treated as the only current hourly interface. [NOAA GHCNh product](https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly)

**Evidence.** Historical data substantially precede 1980: GHCN daily has nineteenth-century records, GHCNh metadata lists a collection envelope beginning in 1750, and ERA5 covers 1940 onward. These are collection bounds, **not complete records at every location or for every variable**. [NOAA daily documentation](https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf), [NOAA hourly metadata](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C01688/html), [ECMWF ERA5 overview](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)

**Evidence.** ERA5 is produced by combining observations and a modeling/data-assimilation system. It is useful for complete global fields but is a model-based reference. Its preliminary ERA5T release has about five days' latency and can differ from the final product released two to three months later. [ECMWF ERA5 time-series description](https://www.ecmwf.int/en/forecasts/datasets/era5-hourly-time-series-data-single-levels-1940-present)

**Lilith decision proposal.** Build three parallel evaluation streams: direct station verification, grid-reference agreement, and hazard/event verification. Keep observation revisions and the original forecast immutable. Archive upstream availability, not just valid times. A forecast issued now cannot legitimately use a later published record for its initial state in an operational replay.

**Evidence.** Operational forecast centers evaluate many aspects of quality and continue developing specific severe-weather evaluation methods; one global headline score does not describe all uses. [ECMWF forecast evaluation](https://www.ecmwf.int/en/research/modelling-and-prediction/forecast-evaluation)

**Lilith decision proposal.** The global product remains the scope. Scientific qualification is tracked independently by variable, lead, region, season, reference type and hazard. A map covering Earth is not a certificate of independently observed accuracy at every coordinate.

## 2. What the existing Lilith runs actually establish

**Repository evidence, read on the research date.** [Training registry](training_runs.json), [training documentation](TRAINING.md) and [current verification contract](DATA_AND_VERIFICATION.md) describe retrospective daily land-station experiments using TMAX, TMIN and PRCP. The 1980-based 14-day experiment selected 144 station IDs, but ordinary test cases came from **42 stations**, plus **8 held-out stations**. Selection count must not be reported as validated location count.

The registry reports a climatology-anomaly ridge candidate with test RMSE approximately **4.24°C TMAX, 4.37°C TMIN, and 7.59 mm precipitation**, versus **4.45°C, 4.56°C, and 7.60 mm** for its empirical climatology. These are aggregated experiment scores, not day-specific error promises. The same registry says no candidate is promoted and reference NWP/calibration are absent. Its 90-day Transformer loses substantially to climatology. These facts support retaining a strong simple baseline and changing the data/evaluation system before expensive scale-up; they do not support operational severe-event claims.

**Required qualification distinction.** All those experiment targets are final historical GHCN records; the current benchmark is not a replay of what was available at each historical issue time. A frozen retrospective test remains valuable, but after results influence design, a fresh temporal or prospective audit is required. This is a Lilith evaluation requirement, not a claim that the previous runs are invalid.

## 3. Observations: preserve what was measured, where, when and how

### GHCN daily: format, quality and the reporting day

**Evidence.** The native daily format separates measurement, quality and source flags. Missing values use `-9999`. TMAX/TMIN are tenths of °C; PRCP is tenths of mm; SNOW/SNWD are mm. A blank quality flag means the value did not fail a QA check, not a guarantee of perfect measurement. Trace and multiday values carry distinct semantics. Source-derived TAVG and extrema can differ from daily local-observer measurements. [NOAA daily README, sections III–IV](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt)

**Evidence.** Yearly CSVs include `OBS-TIME` populated from NOAA's Historical Observing Metadata Repository; the full period is updated daily. The native `.dly` layout does not include that timestamp field. [NOAA by-year format](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme-by_year.txt)

**Evidence.** NOAA's QA methodology applies a sequence of consistency and outlier checks and documents the meaning of the flags. This is screening, distinct from climatic homogenization. [Durre et al. 2010, original QA paper](https://journals.ametsoc.org/view/journals/apme/49/8/2010jamc2375.1.xml)

**Evidence.** GHCN daily does **not** adjust historical biases from instrument or observing-practice changes. Its use for long-period trends or climatologies therefore needs application-specific treatment. [NOAA daily bias-adjustment section](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily), [Menne et al. 2012, original dataset paper](https://www.ncei.noaa.gov/pub/data/ushcn/papers/menne-etal2012.pdf)

**Evidence.** HOMR tracks station locations, equipment, observation times and reporting methods; histories are most extensive for the U.S. Cooperative Observing Program. It does not imply equally complete metadata worldwide. [NOAA HOMR](https://www.ncei.noaa.gov/access/homr/)

**Lilith requirements.** Store the original integer/flags beside decoded units; exclude rejected targets while retaining their rejection reasons. Distinguish trace, true zero, absent report and multiday total. Match a model's daily target to the station's actual accumulation/reporting interval. Use historical station coordinates/elevation where available, with an explicit unknown-metadata state. Keep `(TMAX + TMIN)/2` distinguishable from a reported daily mean. Never create a target by filling missing weather with an average.

### GHCN hourly: use variable- and source-aware QC

**Evidence.** The current documentation identifies GHCNh v1.1.0 and describes variable-level measurement, quality, report-type, source-code and original-station-ID fields. Common QA is applied to temperature, dew point, station/sea-level pressure, wind direction and wind speed; other variables preserve component-source QC. Legacy numeric flags therefore cannot be interpreted using a single universal rule. Its precipitation section explains accumulation durations and incomplete accumulations. [NOAA GHCNh documentation, sections III, V, VI and IX](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf)

**Lilith requirements.** Version the field dictionary/QC mapping. Keep exact report timestamps, original report type and variable-specific accumulation duration. An hourly record can contain a six-hour precipitation amount; do not sum overlapping amounts as if each were one-hour rain. Resolve duplicate source reports explicitly. Prefer column projection and bounded station/year partitions when preparing training batches from the NAS.

### Marine and upper-air measurements extend the validation domain

**Evidence.** ICOADS contains individual surface-marine observations from ships, buoys and other platforms, with an archive envelope from 1662 onward and separate monthly summary products. Its final and provisional streams have different release/version semantics. Collection age does not imply dense global historical coverage. [NOAA ICOADS product and access](https://www.ncei.noaa.gov/products/international-comprehensive-ocean-atmosphere-data-set)

**Evidence.** NDBC real-time files generally retain 45 days and use `MM` missing markers, whereas archived files use variable-specific nines and have post-processing. Real-time automated QC and archival checks differ. [NDBC measurement definitions](https://www.ndbc.noaa.gov/faq/measdes.shtml), [NDBC quality-control procedures](https://www.ndbc.noaa.gov/faq/qc.shtml)

**Evidence.** IGRA contains globally distributed radiosonde/pilot-balloon observations, with some records reaching 1905. Its current product page identifies v2.2, variable temporal/vertical coverage and possible instrument/practice discontinuities despite QC. [NOAA IGRA](https://www.ncei.noaa.gov/products/weather-balloon/integrated-global-radiosonde-archive)

**Lilith proposal.** Add marine and upper-air cohorts instead of extrapolating land-station verification to oceans or pressure levels. Preserve platform location at observation time, measurement height/depth, sampling duration, reporting convention and quality flags. Distinguish sea-surface from air temperature, observed from standardized-height wind, and instantaneous from averaged quantities. Match atmospheric model variables at compatible pressure/height, and account for balloon displacement when coordinates exist. Keep source overlaps visible: using a different file format for the same buoy report does not create a second independent observation.

### Latency and revision are part of truth quality

**Evidence.** The older daily documentation describes one-to-two-day availability where possible; current collection update frequency is daily, which is not a universal per-station delivery guarantee. GHCNh metadata likewise promises a daily update process rather than a universal real-time observation feed. [NOAA daily documentation](https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf), [NOAA hourly metadata](https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C01688/html)

**Lilith proposal.** Learn arrival-lag distributions from ingest logs per source/network/variable; publish maturity watermarks. Use a separate near-real-time source when a daily/hourly archive arrives too late to initialize the product. Historical records whose original availability is unknown remain a retrospective cohort, with any conservative assumed lag reported explicitly. Snapshot raw payload/hash, retrieval time, source version and parser version; append later corrections without deleting earlier verification evidence.

## 4. Reanalysis and independent evaluation

**Evidence.** ERA5's assimilation system incorporates observations across assimilation windows. ECMWF explains that analysis construction combines a previous forecast with observations and that the observing system changes across history. Reanalysis initialization can therefore convey information not available to a same-clock operational forecast, and reference uncertainty varies by era and location. [ECMWF ERA5 description](https://www.ecmwf.int/en/forecasts/datasets/era5-hourly-time-series-data-single-levels-1940-present), [ECMWF ERA5 documentation](https://confluence.ecmwf.int/pages/viewpage.action?pageId=388500357)

**Evidence.** ECMWF also verifies forecasts directly against observations using suitable temporal/spatial interpolation and observation operators. That is a distinct evaluation from verifying against an analysis field. [ECMWF observations-based verification](https://www.ecmwf.int/en/newsletter/149/meteorology/use-forecast-departures-verification-against-observations)

**Lilith requirements.** Record both `valid_time` and `available_at`; preserve model initialization, forecast lead and source run. A future target observation may mature after issuance—that is normal. A predictor may not. Score a forecast initialized from finalized reanalysis as a retrospective reanalysis-initialized experiment unless a fair operational initialization has been demonstrated. Keep final reanalysis useful for training but prohibit it from silently entering a real-time backtest's inputs.

“Independent station verification” means comparison with measured station values rather than another forecast API. It does not automatically mean that the observing network was withheld from all assimilation systems or from a pretrained model's training. Record network lineage and, when known, assimilation/training overlap. Fit point corrections without the held-out station/year; do not call interpolation onto assimilated stations completely independent system validation.

## 5. Evaluation design that resists accidental optimism

**Evidence.** Structured-data cross-validation research shows random splitting can underestimate prediction error when spatial/temporal dependence is ignored; blocking must match whether interpolation or extrapolation is intended. [Roberts et al. 2017, authors' paper page and abstract](https://florianhartig.github.io/publications/roberts2017cross/)

**Evidence.** WeatherBench 2 supplies reproducible medium-range evaluation, baseline datasets and code; its paper distinguishes whole forecast-system comparisons and acknowledges limitations of headline metrics for spatial/temporal structure and extremes. [Rasp et al. 2024, original paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023MS004019), [official benchmark data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)

**Lilith proposed protocol.** Freeze an experiment manifest before scoring and use the same valid cases for all candidates:

1. **Rolling temporal backtests.** Fit only before each cutoff, calibrate in a later separate block, then evaluate the next unseen block. Include distinct historical eras and the latest available years; never learn scalers, seasonal normals, thresholds or weights from future targets.
2. **Station holdouts.** Exclude held-out station IDs from model fitting and calibration. Define what local climatology is permitted at a new station; a model with local historical normals is a different task from zero-history transfer.
3. **Contiguous region holdouts.** Hold out entire regions with spatial buffers appropriate to grid spacing and correlation. Nearby-station exclusion alone does not establish cross-region transfer. Record overlap with pretrained global-model data separately.
4. **Event-group holdouts.** Place an entire cyclone, heatwave, synoptic storm system or flood episode in one fold. Purge overlapping target windows across time boundaries. Neighboring stations from the same storm do not provide independent event samples.
5. **Prospective shadow issuance.** Run the frozen candidate on a fixed global evaluation schedule with actual input arrival logs; archive before targets occur. Preserve a champion/reference forecast on the same schedule.
6. **Coverage accounting.** Retain eligible, matched, pending, missing, rejected and unsupported counts. Report how complete-window selection changes latitude, terrain, climate and precipitation distributions.

**Proposed uncertainty method.** Estimate paired score differences using time/event blocks with station or region clustering; choose blocks from dependence diagnostics and show sensitivity to block choice. State sample count and effective event count. The WMO subseasonal verification implementation uses block bootstrap for skill significance, providing an operational precedent. [WMO lead centre verification](https://confluence.ecmwf.int/spaces/WLS/pages/394992254/Verification)

## 6. Metrics: choose the target before the score

**Evidence.** Proper scoring rules reward honest predictive distributions; CRPS evaluates a complete forecast CDF and generalizes absolute error. [Gneiting and Raftery 2007, original paper](https://doi.org/10.1198/016214506000001437), [authors' full paper](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)

For each variable and lead, the following is Lilith's proposed minimum reporting contract:

| Target | Required measures and interpretation |
|---|---|
| Continuous point quantity | MAE, RMSE, signed bias in physical units; same-case improvement over climatology, persistence and archived reference NWP |
| Predictive distribution | CRPS, interval coverage and width, quantile loss, rank/PIT diagnostics; calibration split kept separate |
| Binary event probability | Brier score, Brier skill, reliability bins with counts and uncertainty, probability-threshold curves |
| Rain amount/occurrence | Separate occurrence from intensity; match accumulation intervals; report dry/wet and high-intensity cohorts |
| Spatial event field | Threshold/neighborhood FSS across specified scales plus physical magnitude and displacement errors |
| Cyclone/other coherent object | Track, timing, intensity and footprint errors; preserve object matching and missed/spurious objects |

**Metric definitions.** With hits `H`, misses `M`, false alarms `F` and correct negatives `C`: detection probability is `H/(H+M)`, false-alarm ratio is `F/(H+F)`, false-alarm rate is `F/(F+C)`, and critical success index is `H/(H+M+F)`. These last two false-alarm quantities are different. Binary Brier score is `mean((p-y)^2)`; lower is better. Skill against a fixed reference is `1 - score_model/score_reference`, when the denominator is defined. Undefined denominators stay null with counts. [MET verification measures, maintained by the verification-tool developers](https://metplus.readthedocs.io/projects/met/en/latest/Users_Guide/appendixC.html)

**Evidence.** Finite ensemble size affects scores such as CRPS; fair scores are appropriate for certain comparisons under exchangeability assumptions. A larger ensemble is not automatically a better underlying model. [Leutbecher, ECMWF 2018 research explanation](https://www.ecmwf.int/en/newsletter/157/news/how-many-ensemble-members-are-desirable), [original ensemble-size paper](https://doi.org/10.1002/qj.3387)

**Lilith requirements.** Store member count, weights and whether ordinary or fair CRPS was used. Quantile forecasts require a documented CRPS approximation. A good CRPS does not establish calibration of a rare hazard by itself. Do not compare per-station arithmetic RMSE to area-weighted global-grid RMSE as if they were identical experiments.

## 7. Extreme weather needs false alarms, spatial context and non-events

**Evidence.** Evaluation only on occasions where disasters occurred creates the forecaster's dilemma: an always-alarm strategy can appear artificially good. Properly designed weighted scores can emphasize high-impact outcomes while preserving meaningful evaluation. [Allen et al. 2023, original weighted-verification paper](https://journals.ametsoc.org/view/journals/wefo/38/3/WAF-D-22-0161.1.xml)

**Evidence.** Small spatial displacements can incur both a miss and a false alarm under gridpoint error scores. ECMWF uses fractions skill score across thresholds and neighborhood scales to examine useful spatial skill. Positive FSS alone does not certify usefulness. [ECMWF spatial-verification research explanation](https://www.ecmwf.int/en/about/media-centre/science-blog/2023/verifying-high-resolution-forecasts); original method: [Roberts and Lean 2008, DOI](https://doi.org/10.1175/2007MWR2123.1)

**Evidence.** An original comparison of IFS HRES, Pangu-Weather and GraphCast finds the best extreme-weather performance depends on region and type of extreme. This is evidence against assuming that a winning global-average model wins every hazard everywhere. [2024 GMD comparison paper](https://gmd.copernicus.org/articles/17/7915/2024/)

**Lilith proposal.** Register event definitions before scoring: threshold, accumulation/duration, spatial radius/footprint, season, reference and minimum evidence. Report physical thresholds and locally derived climatological percentiles separately. Learn percentile cutoffs from allowed historical training data; never estimate a test event's threshold using that same future period.

Maintain the complete scheduled sample of events **and non-events** for proper-score evaluation. Keep event-only case reports as diagnostics. If training oversamples extremes or uses class weights, recalibrate probabilities to deployment prevalence on held-out data. Store all alert thresholds chosen on validation, missed events, false alarms, first useful lead, escalation timing and stale-data intervals. Use no universal heat/flood threshold across all climates.

## 8. Hazard truth: what the major archives can and cannot label

| Evidence source | Verified source facts | Lilith implication/proposal |
|---|---|---|
| NOAA Storm Events | U.S. and adjacent-water significant events; data begin 1950; search warns event types are limited before 1993 and appearance takes about three months. County-based versus forecast-zone collection depends on event scale. [Current NOAA search](https://www.ncei.noaa.gov/access/storm-events-database/search) | Preserve event taxonomy/version, episode ID, zone/county and geometry precision; do not equate absence of a report with a measured non-event. Apply maturity delay and era-specific eligibility. |
| Storm Events historical era details | NOAA's local explanation lists tornadoes since 1950, severe thunderstorms since 1955 and other types since 1996; publication/database availability is 90–120 days. [NWS storm-report records](https://www.weather.gov/unr/storm_reports) | Pre-1996 missing heat/flood/winter rows cannot be trained as universal negative labels. Use source-specific historical rules. |
| SPC local storm reports | Preliminary reports are shown as received; SPC explicitly omits non-thunderstorm LSRs such as hurricane wind reports from this summary. [SPC report disclaimer](https://origin-west-www-spc.woc.noaa.gov/climo/online/) | Use for early provisional corroboration, not a complete multi-hazard final truth table. Deduplicate reports and reconcile later. |
| IBTrACS | Merges global agency best tracks; current product is v4r01. Agency wind averaging and historical differences matter; pre-1950 duplicates/mismatches are known. [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) | Track agency, wind averaging convention and provisional/final status. Keep a whole storm in one fold; do not simply pool one-minute and ten-minute wind thresholds. |
| IBTrACS source caveats | Source best tracks are not uniform quality; there is no single simple global conversion that resolves all reporting differences. [Technical details, section 6](https://www.ncei.noaa.gov/sites/g/files/anmtlf171/files/2024-07/IBTrACS_version4r01_Technical_Details.pdf) | Design basin/agency-specific intensity verification and separate track uncertainty from exposure/impact estimates. |
| MRMS precipitation/hazard estimates | Blends radars, gauges, satellite, lightning and forecast-model information; products are estimates with regional domains. [NOAA NSSL MRMS](https://www.nssl.noaa.gov/projects/mrms/) | Keep estimator/version and dependencies; a gauge-corrected field and the same gauge are correlated verification references. Do not represent MRMS as global independent measurements. |
| Radar/gauge mismatch | Radar samples aloft; evaporation can make detected precipitation fail to reach the ground. Gauges sample points. [NSSL research explanation](https://inside.nssl.noaa.gov/mrms/2018/03/when-radar-observed-precipitation-does-not-reach-the-ground/) | Preserve missing/blocked/low-quality coverage; compare amount, support and accumulation, not merely two raster values. |
| USGS hydrology | Telemetry is provisional until reviewed and may be materially revised. [USGS provisional-data statement](https://water.usgs.gov/data/provisional.html) | Keep approval/status codes and versions. Rainfall alone does not label river flooding: verify stage/discharge against site/basin thresholds. |

**Operational discovery update.** NOAA moved the Storm Events interface in August 2026; connector docs should use the current `/access/storm-events-database` location rather than depend on a legacy redirect. [NOAA service-change notice](https://www.nesdis.noaa.gov/news/location-change-storm-events-database)

**Operational discovery update.** USGS publishes modernized Water Data APIs and describes migration from legacy `waterservices.usgs.gov` clients. Implement against current documentation and verify endpoint availability during adapter development. [USGS API announcement/documentation entry](https://www.usgs.gov/tools/usgs-water-data-apis)

**Uncertainty.** No source above supplies complete worldwide labels for every hailstone, tornado, flash flood, surge or human impact. Underreporting and changing surveillance can alter apparent event frequency. Lilith must explicitly distinguish observational coverage, reported events, physical hazard estimates and impacts. A zero damage report is not a universal “no physical hazard” observation.

## 9. Official products: respect their horizon and probability meaning

These are source-owned products to ingest/archive and compare where their scopes match. They are not target observations.

| Authority/product | Current documented contract | Product behavior proposed for Lilith |
|---|---|---|
| NWS point/hourly/grid forecast | `/points` resolves office/grid mapping; forecast products cover the next seven days. [NWS API documentation](https://www.weather.gov/documentation/services-web-api) | Preserve issued/updated timestamps, grid mapping and variable intervals; archive each actually available run. U.S. product scope is explicit. |
| SPC convective outlooks | Days 1–3 plus days 4–8, continental U.S.; probabilities describe severe weather within 25 miles of a point. Day 1/2 separate tornado/wind/hail, day 3 combines them. Current March 2026 documentation introduces Conditional Intensity Groups replacing SIGN labeling. [SPC current descriptions](https://origin-west-www-spc.woc.noaa.gov/about/outlooks/) | Version schema by effective date; retain probability radius, event class and conditional intensity separately. Parse source valid intervals instead of inferring a local calendar day. |
| WPC Excessive Rainfall Outlook | Days 1–5; rainfall exceeding flash-flood guidance within 40 km/25 miles of a point. [WPC ERO](https://www.wpc.ncep.noaa.gov/qpf/excess_rain.shtml) | Label as the defined regional excessive-rainfall risk, not a calibrated household inundation probability. |
| NHC tropical outlook/advisory | Tropical formation chances for 48 hours and seven days, with basin-specific scope. [NHC text product descriptions](https://www.nhc.noaa.gov/aboutnhcprod.shtml) | Separate disturbance formation probability from track, wind, rain and surge; preserve source authority per basin. |
| NHC cyclone cone | Shows center-track uncertainty through five days; it does not define all affected weather. [NHC cone explanation](https://www.nhc.noaa.gov/templates/graphics_statcone_inc.shtml) | Display wind/surge/rain layers alongside center track; never use cone-outside as a safe-weather classification. |
| CPC outlooks | 6–10 day, 8–14 day, monthly and seasonal products express departures from normal over periods; seasonal outlooks use three-month periods. [CPC outlook descriptions](https://www.cpc.ncep.noaa.gov/products/outlooks/) | Render probabilistic period-average/accumulated anomalies separately from hourly/day-specific forecasts. Evaluate against matching aggregation/climatology. |

**Lilith inference.** A single station-temperature network cannot scientifically inherit the capabilities of all these distinct forecast systems simply by producing a long output vector. Each supported hazard requires appropriate state inputs, target definitions, data latency, calibration and evaluation. Keep global capability in the architecture while allowing per-product status to remain research, shadow or qualified.

## 10. A professional global alert service

**Evidence.** CAP is an international message format supporting event, severity, urgency, certainty, areas and instructions. WMO maintains resources and an alert-authority registry for identifying authoritative warning issuers. [WMO CAP overview](https://wmo.int/site/wmo-common-alerting-protocol), [WMO alerting-authority registry](https://alertingauthority.wmo.int/), [WMO CAP implementation resources](https://wmo.int/site/wmo-common-alerting-protocol/cap-resources)

**Evidence.** The CAP 1.2 standard defines message identifiers/senders/timestamps; Actual, Test and Exercise statuses; update/cancel message types with references; effective/onset/expiration fields; and geometry/geocodes. CAP conformance alone does not make an issuer an official government authority. [OASIS CAP 1.2 standard](https://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2-os.html)

**Lilith proposed alert contract.** Keep the complete original payload and a normalized searchable record. Validate issuer/feed against the authority registry and configured source domains. Use source identifier + sender + sent time for idempotency, preserve references, process cancellation/update/expiry, and keep Test/Exercise events out of public warning channels. Do not assume every alert has a polygon; resolve supported official geocodes while preserving source precision. Keep language, instruction text and authoritative links.

Distinguish government warnings from Lilith-generated forecast/risk notifications in data type, visual treatment and outbound text. A Lilith “no elevated risk” result must not hide an active authoritative warning. Product decisions include opt-in places/hazards, spatial matching, delivery retries, deduplication, acknowledgment, escalation, expiration, delayed messages and a delivery audit. Archive notification decisions and payloads for later event reports. These are implementation proposals, not claims of existing delivery capability.

**Evidence with retention consequence.** NWS `/alerts` returns approximately the past seven days of alerts, not a full research archive; its active endpoint is not radar imagery. [NWS alerts and radar API details](https://www.weather.gov/documentation/services-web-api)

**Lilith requirement.** Ingest and persist messages continuously once the service is running. Treat a successful empty feed, feed error, stale feed and unsupported region as different states. Track issue-to-ingest and ingest-to-delivery latency; showing stale warnings without their status is an alert-system failure even when the weather model is accurate.

## 11. Global exchange and what “everywhere” can honestly mean

**Evidence.** WIS2 became operational on 1 January 2025 and supports international Earth-system data exchange. Its guide describes discovery catalogs, MQTT notifications, HTTP resource access, global caches and access controls for recommended data. Core and recommended data have different exchange/usage conditions. WIS2 is not a uniform REST weather API and does not promise every national dataset is keyless or identically licensed. [WMO operational announcement](https://wmo.int/media/news/wmo-information-system-20-will-transform-sharing-of-earth-system-data), [approved WIS2 guide](https://wmo-im.github.io/wis2-guide/guide/wis2-guide-APPROVED.html)

**Lilith proposal.** Maintain source registry records for coverage, variables, vertical levels, time support, history, release lag, source IDs, authentication and actual license. Discover national/regional authorities through WMO mechanisms and review each feed. Integrate authoritative warnings by territory and retain language/provenance; NWS, SPC, WPC and NHC alone do not constitute worldwide alert coverage.

Report global gridded forecast availability separately from direct verification availability. A verification dashboard should distinguish at least: measured station match; radar/gauge/satellite estimate comparison; grid-analysis reference agreement; unsupported target; insufficient records; pending truth; and stale inputs. Ocean/polar/sparse-land areas may have global forecasts but limited near-surface independent observations. Do not fabricate a numeric local accuracy score where evidence is absent.

## 12. Reports and storage contracts for the NAS/Proxmox design

The following is an **architecture proposal** informed by the source revision/retention facts above. The NAS removes the old assumption that all records must fit a tiny application database; it does not remove the need for bounded retention, indexing or measured capacity.

- **Raw source archive:** compressed source objects keyed by source/version/date/hash; immutable retrieval metadata and actual availability logs. Store in NAS/object storage. Reconcile source corrections into new versions.
- **Analysis-ready observations:** partitioned Parquet for station observations and event labels, preserving flags/provenance; chunked arrays for gridded products. Keep physical support, units, valid times and quality masks.
- **Forecast archive:** raw Lilith outputs, calibrated/blended outputs and reference forecasts separately, with issue/cutoff/model/input hashes, lead, spatial support and distribution representation. A returned forecast is persisted before being counted as successfully issued.
- **Small operational SQL index:** run metadata, source health, stations, object pointers, alert lifecycle, verification jobs and report summaries. Avoid putting all dense global grids in transactional rows.
- **Verification pair archive:** forecast/reference object IDs, observation version, match method/distance/elevation difference, variable/interval, QC decision, weighting and exclusion reason. Persist sufficient statistics and/or paired arrays to reproduce results.
- **Retained report objects:** stable report ID, generation time, data watermark, exact cohort/query, model/code/data/matcher/metric versions, sample/coverage counts, uncertainty, raw metrics and readable narrative. Revisions make new report versions.

Proposed cadence: daily data-health/maturity report; weekly model-and-reference skill report; monthly global/regional calibration and coverage report; event case report covering every relevant issue, false alarms and misses. Aggregate cached reports from versioned objects; do not recalculate the only copy of historical scores from mutable live tables.

Capacity should be estimated per data family from actual sample compression, grid dimensions, variables, levels, ensemble members, issue frequency, lead steps and retention. Global gridded ensembles can dominate storage even when station Parquet and report summaries are modest. Keep CPU/GPU working shards near the compute; use the NAS as the durable archive with independently verified backups and restore tests. NAS access method and mounted paths remain deployment facts to validate; this note has not mounted or changed storage.

## 13. Prioritized checklist for Astra refinement and implementation

Priority indicates dependency/order of heavy work, **not a reduced global product scope**.

### P0 — before larger training jobs or public skill claims

- [ ] Freeze/version the current retrospective artifacts and preserve the fact that only 42 ordinary + 8 held-out stations contributed to the 1980-based test.
- [ ] Add GHCNh as the current hourly target and explicit ISD lineage/mapping; pin v1.1.0 field/QC definitions in adapter tests.
- [ ] Produce a global station/variable/era coverage census, including pre-1980 records, actual valid fractions and complete-window exclusion bias.
- [ ] Implement exact interval/unit/flag handling and station metadata epochs, with separate daily observation-time and hourly accumulation semantics.
- [ ] Add `available_at`, first-seen payload hashes, source versions and maturity/revision state to observation and upstream forecast contracts.
- [ ] Freeze rolling temporal, region and event-group partitions, calibration blocks and a new prospective audit period before candidate refinement.
- [ ] Archive same-issue reference NWP and evaluate on identical cases. Keep raw Lilith, calibrated result and blend scores separate.
- [ ] Write a metric contract covering physical-unit scores, CRPS/Brier/reliability, coverage, blocks/uncertainty and versioned baselines.

### P1 — global hazard and alert qualification

- [ ] Build versioned hazard-label specifications; implement Storm Events maturity/era rules, IBTrACS agency conventions and hydro/radar QA.
- [ ] Register both event and non-event samples, evaluation footprints/radii and validation-only thresholds; track misses and false alarms.
- [ ] Support the March 2026 SPC Conditional Intensity Group schema alongside historical schemas.
- [ ] Preserve CAP lifecycle, trusted-issuer provenance and notification delivery audit; archive feeds beyond their public rolling retention.
- [ ] Publish a territory/variable/hazard matrix of forecast, observation and authoritative-warning coverage and source freshness.
- [ ] Add retained machine-readable and readable reports, with reproducible queries and revision diffs.

### P2 — scale and operational acceptance

- [ ] Evaluate global gridded forecasts against both analysis references and withheld observation cohorts; add probability and spatial/event diagnostics.
- [ ] Compare broad/full-history fits against recent-era fits and drift-aware variants without contaminating final test selection.
- [ ] Verify NAS object throughput, compression/capacity, dataset manifests, backups and Proxmox restore path before sustained training ingestion.
- [ ] Run prospective shadow inference with real availability constraints and explicit per-product promotion/rollback gates.

## 14. Known uncertainty and source-reading boundaries

The sources establish data/product behavior and evaluation methods; they do not establish Lilith's skill, universal data availability, a guaranteed warning delivery SLA, or that decades-old observations are homogeneous. Current adapter implementations still need live bounded probes, source-license review and reproducible parser examples.

NOAA's GHCNh PDF was retrieved directly and read using PDF extraction because web rendering rejected its `application/octet-stream` content type. Some legacy SPC pages returned 403; the current NOAA-operated `origin-west-www-spc.woc.noaa.gov` descriptions were accessible and read. The original Roberts–Lean FSS DOI/publisher page rejected full-text access; operational explanations here therefore rely on the accessible first-party ECMWF research account, with the original DOI retained for follow-up. The Roberts et al. cross-validation source was read on a coauthor's publication page; its abstract supports the general blocking rationale, while the concrete Lilith fold choices above are proposals requiring empirical validation.

Astra should resolve concrete protocol choices—event labels, sampling, climatology period, station matching, baseline availability, regions, probability calibration and promotion thresholds—before interpreting another training run as a material advance toward the global production system.
