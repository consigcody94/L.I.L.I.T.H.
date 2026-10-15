# Current CONUS training campaign

The user's latest steering focuses Lilith on CONUS. Prior global design/literature remains useful background, but the current modeling and verification panel is the contiguous 48 states and DC.

## Observation registry and actual data

Use the official [NOAA HOMR station reports](https://www.ncei.noaa.gov/access/homr/reports) and their published fixed-width layouts. The reviewed snapshot contains 881 ASOS entries and 1,031 AWOS entries in those states. These are registry entries, not necessarily distinct stations across both programs or guarantees of historical completeness. NOAA describes approximately 950 ASOS sites nationwide; the larger automated network includes AWOS. [NWS ASOS](https://www.weather.gov/asos/).

`training/conus_catalog.py` retains all registry entries and matches daily station identity using an explicit GHCND field where available, otherwise WBAN/COOP plus a <=10 km coordinate crosscheck. The first snapshot yields 1,082 unique GHCN daily stations and 828 unmatched registry entries. Unmatched stations remain an hourly/METAR acquisition backlog. Do not count them as sites without weather, guess identities from names, or label COOP-only stations as ASOS hardware.

Historical NOAA daily files can predate ASOS commissioning. Preserve that history and source flags without claiming it came from instrumentation installed later. GHCN reporting-day intervals, QC, missing targets, changing availability and source revisions remain material qualifications.

## Fixed experiment

[The protocol](experiments/conus-round2/PROTOCOL.md) was written before fitting or viewing new audit skill. It specifies 30 input days, a two-day assumed observation embargo and 14 targets; fit 1980–2016, validation 2017–2020, calibration 2021–2023, inspected development 2024–2025, then a separate 2026 retrospective audit through September. Partition purging means the first month is input/gap support rather than scored targets. Missing truth never becomes an imputed target.

Changes driven by the literature:

- Each input anomaly uses its own seasonal normal, with a leap-aligned calendar.
- Missing inputs use training-period seasonal values plus explicit masks; partial target variables retain separate denominators.
- Seasonal normals use historical local observations for both model and baseline. Station holdouts exclude neural/regression fitting, but allow pre-2017 local normals; this is not zero-history or contiguous-region transfer.
- Corrected multivariable ridge plus a small residual MLP and residual Transformer; fixed validation-selected blends and champion choices, with raw outputs reported separately.
- Wet-day probability evaluation and separate residual interval calibration; measured coverage is required, not a blanket calibrated-uncertainty claim.
- Per-case arrays, per-lead/station/latitude/season scores, reference comparisons and approximate clustered uncertainty are retained. No automatic production promotion or severe-event claim.

## Execution boundary

`training/cloud/modal_conus.py` downloads NOAA records directly on Modal, builds/verifies arrays there, runs the new regression tests and exports frozen data/source/raw-record packages to a private Modal volume. The Mac's interrupted local preparation is not the final dataset and must not resume. Existing raw cache files remain intact.

The private Kaggle script in `experiments/conus-round2/kaggle` requires CUDA, records the actual virtual GPU, fits the two neural candidates and evaluates measured NOAA targets. Statistical ridge and model input transformations run on Kaggle's host CPU; no substantive fitting or data preparation should run on the Mac. Bounded epochs/timeouts and unique run names prevent uncontrolled retries or overwriting finished evidence.

Prepared datasets, original NOAA snapshots, source hashes, tests, checkpoints, case predictions and reports are copied to a new immutable NAS research directory with remote checksum verification, retaining independent copies. Bulk archives belong on the NAS; a future live SQL/WAL database belongs on local Proxmox guest storage. This does not deploy an API, recurring worker or notification service.

Results and actual execution IDs will be added to the Astra handoff after completed jobs. Never replace measured results with expectations about whether the revised architecture ought to win.
