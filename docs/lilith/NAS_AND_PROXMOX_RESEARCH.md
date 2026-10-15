# NAS and Proxmox storage design — evidence and decisions

Research date: 2026-10-14 (user’s local date). User constraint: existing NAS storage is available now; the eventual Proxmox deployment will access the NAS on the local network. This supersedes any implication that paid cloud object storage must be selected before proceeding.

## Decision

Use the NAS as the bulk archive for source snapshots, normalized station/forecast arrays, model checkpoints and retained reports. Keep the hot transactional database on a local Proxmox guest disk and back it up consistently to the NAS. Continue GPU training on the available remote accelerators; the NAS is the data home, not the training accelerator. This is an engineering recommendation for the user’s topology, not a claim that any service has already migrated.

A read-only check confirmed that the NAS is reachable and Proxmox already has an active CIFS mount. Private operational capacity/host details are retained only in the local outputs, not in this public repository. The data volumes currently have one member each, so the NAS must not be the only copy of irreplaceable models/reports. Existing Proxmox guest backups were observed; their presence does not imply that newly added archive directories are independently protected or restore-tested.

## Database placement and filesystem semantics

**Evidence:** SQLite documents that WAL needs shared memory among clients on one host and does not work across a network filesystem. A network share mounted at a local-looking directory remains a network filesystem. A bind mount into an LXC does not turn CIFS/NFS into local storage. [SQLite WAL documentation](https://sqlite.org/wal.html).

**Decision:** point `LILITH_LEDGER_PATH` at a real local guest filesystem during the pilot. Never point the live WAL ledger directly at the NAS CIFS/NFS directory or open it concurrently from the Mac and Proxmox. For production, use PostgreSQL/PostGIS over its database protocol; preserve immutable array/report files on the NAS.

**Evidence:** PostgreSQL can operate on appropriately configured NFS, but its documentation requires hard mounts and emphasizes correct server durability/fsync behavior and outage monitoring. That does not establish that a specific appliance/share is qualified for database storage. [PostgreSQL filesystem and NFS documentation](https://www.postgresql.org/docs/current/creating-cluster.html#CREATING-CLUSTER-NFS).

**Decision:** local database storage is the initial design because it isolates transactional latency and share outages. CIFS can remain the archive transport already present; do not convert the storage protocol, change exports or remount production shares simply to follow a generic NFS example. A later protocol change requires workload measurements and an explicit operational change plan.

## Compact storage can work; full atmospheric mirrors are different

Raw float32 values only, excluding flags, station/grid metadata, indexes, revisions, ensembles, compression and backups. The 1980–2025 interval contains 16,802 calendar days. Calculations are illustrative datasets, not claims about actual source completeness or sizes.

| Hypothetical retained data | Calculation | Raw decimal bytes |
|---|---|---:|
| 100,000 stations, daily, three variables, 1980–2025 | 100,000 × 16,802 × 3 × 4 | 20.16 GB |
| 20,000 stations, hourly, ten variables, same period | 20,000 × 16,802 × 24 × 10 × 4 | 322.60 GB |
| 0.25°-like 720×1440 grid, hourly, ten surface variables, same period | 720 × 1440 × 16,802 × 24 × 10 × 4 | 16.72 TB |

**Inference:** long historical station data and compact prediction records can fit comfortably within a multi-terabyte archive when their real overhead and duplication are controlled. Mirroring every global hourly grid/atmospheric level indefinitely is a different storage class. Free government access does not remove those bytes. Preserve full source history for the actual station/variable cohorts and use indexed/chunked subsets of massive gridded sources, with versioned manifests and explicitly chosen retention.

Proposed first operational budget: measure a representative month and historical backfill sample, then establish a weather archive soft limit and growth alerts. A 256 GiB pilot budget is a sizing starting point, not a quota applied to the NAS and not permission to prune existing user data. Keep a separate retention decision for high-resolution ensembles and event cases. No automatic deletion is enabled here.

## Files and chunking

Store immutable Parquet partitions for station observations and verification samples; retain standardized typed units, missing masks, original flags, source IDs, valid/available times and revisions. Use gridded chunks/shards sized for the actual temporal/spatial queries. Cache frequently accessed metadata and extraction results on the Proxmox guest.

**Evidence:** Zarr’s sharding specification exists to group multiple chunks under one storage key, reducing the problems of enormous numbers of small filesystem/object entries. [Zarr v3 sharding specification](https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html).

**Decision:** do not create one NAS file for each coordinate/variable/hour. Benchmark chunk/shard sizes and compression against local extraction, region maps and training reads. A provisional 32–256 MiB shard range can be tested; it is not a universal optimum. Record actual compression, inode count, sequential throughput and random-query latency. Keep SQL catalogs small by referencing shared arrays rather than inserting every grid value into relational rows.

A plain filesystem content-addressed archive is sufficient initially. An S3-compatible service is optional only if useful for client interoperability; the logical object-key/manifest design does not require deploying an extra server on the NAS. The NAS appliance’s limited resources are a reason to keep database/API/model work in the compute guest, subject to measured Proxmox memory headroom.

Suggested namespace inside a dedicated Lilith share/directory:

```text
Lilith/
  raw/<source>/<snapshot-sha256>/...
  observations/<source>/<schema>/year=YYYY/region=R/...
  forecasts/<model>/<issue-cycle>/<product>/...
  models/<model-id>/<checkpoint-sha256>/...
  reports/<report-id>/<version>/...
  manifests/<dataset-or-run-id>.json
  backups/database/<timestamp>/...
  staging/<worker-run-id>/...
```

## Write publication and outages

These are proposed invariants, not an implemented NAS worker:

1. Fetch/prepare into a bounded local spool. Record raw hashes, expected lengths and manifest content.
2. Write to an isolated NAS staging directory. Validate lengths/checksums after transfer; publish completed objects under immutable names. Same-share rename/fsync behavior must be validated for the actual protocol/appliance.
3. Publish a completed manifest only after all referenced objects are present and durable; then transactionally expose its catalog reference. Never expose half-written arrays as complete forecasts.
4. On share outage, pause archive publication and report storage degradation. Preserve already committed local issuance evidence within a bounded spool. Stop new work before the spool fills; do not silently spill a missing mount’s writes onto the guest root disk.
5. On recovery, reconcile by content hash, retry idempotently and advance watermarks only for complete datasets. User-facing forecast freshness and archive durability are separate states.

Mount readiness, expected filesystem identity and free-space checks belong in service startup/preflight. A path existing on disk is not proof the NAS is mounted. Run disk-full, network-loss, interrupted-copy, duplicate-publish and recovery tests before deployment.

## Backups and recoverability

**Evidence:** Proxmox’s container documentation distinguishes storage-backed mount points and bind mounts; the latter need a separate backup mechanism. A container backup alone should not be assumed to include the NAS directory bind-mounted into it. [Proxmox-owned container documentation](https://github.com/proxmox/pve-docs/blob/master/pct.adoc).

**Decision:** keep the existing guest backups, add an explicit archive backup/replication inventory, and restore-test the model/report/manifests and database together. The local database and NAS archive are separate failure/backup units. A snapshot on a data disk does not provide an independent copy after that disk fails.

**Evidence:** SQLite’s online backup API makes a consistent database snapshot and avoids copying a live main database file while overlooking active state. [SQLite backup API](https://www.sqlite.org/backup.html).

**Decision:** create consistent local pilot snapshots (or PostgreSQL database-aware backups in production), then copy those closed artifacts to the archive. Retain at least one independent copy of irreplaceable checkpoints/reports/manifests outside the NAS; reproducible raw public data can have a different policy from unique forecast issuances and research evidence. Keep credential material out of public manifests and GitHub.

## Proxmox deployment preparation

Do not install the weather service directly into the hypervisor management environment. Use an isolated VM or appropriately constrained unprivileged container, local application/database disk, an explicit archive mount/bind path, read-only consumer access where practical and a restricted archive-writer identity. Verify UID/GID/ACL mappings without granting broad privileges simply to bypass a permission error. This is a deployment recommendation; no guest, share, service, protocol or network configuration was changed.

Separate web/API, bounded data workers and database resource budgets. The current Proxmox host has other active workloads; do not allocate its apparent total RAM as if Lilith were alone. Whole-atmosphere GPU training remains remote unless the user later supplies suitable dedicated compute. Copy prepared inputs/checkpoints using private transfer paths; do not expose NAS storage protocols to cloud runtimes on the public internet.

Before rollout, measure current RAM/swap, database fsync latency, archive transfer/metadata throughput, backup restore time and failure behavior. Then choose guest resources and retention targets. Local NAS access reduces cloud archive dependency; it does not remove observation sparsity, model predictability limits or the need for calibration and scientific evaluation.
