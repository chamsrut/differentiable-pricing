# Task 9F: private object storage and entitlement-aware vendor ingestion

## Status

`On hold` — **a plan only. Nothing in this file has been built, and nothing in
it may be built without a separate decision to start it.**

It was written as the deferred half of task 9D
([task-9d-data-holdings-audit.md](task-9d-data-holdings-audit.md)) and records
what remote storage and vendor ingestion would have to look like, so that the
question is answered on paper before anyone reaches for credentials.

**Cloud storage and vendor ingestion are deferred reproducibility work, not
prerequisites for the first CRR learnability experiment**
([../../decision-log.md](../../decision-log.md) DEC-030, DEC-031). Phase 1 of
the locked roadmap runs entirely against local holdings. This task exists so
that phase 2 — the XSP/SPY study — has a written plan to start from, and so
that nobody improvises one under time pressure.

**Resumption condition:** a recorded decision that the project needs
machine-to-machine access to its own archive, which in practice means the
XSP/SPY phase running on more than one machine or needing an auditable
retrieval path. Not before.

## Objective

If this project's authorized holdings had to live somewhere other than one
developer's disk, what would that have to satisfy? A positive answer is a
design that an authorized user can follow to reconstruct the archive
bit-for-bit; a negative answer — that some part cannot be done without
redistributing licensed content, or without holding credentials the project
should not hold — is equally valid and stops that part here.

## Motivation

Task 9D established that the archive is a single, indivisible, never-committed
unit of 120 files and 747,750,918 bytes, verified by an in-tree checksum
manifest, plus roughly 2.8 GiB of derived Parquet
([../../data-holdings-catalogue.md](../../data-holdings-catalogue.md)). All of it
exists in exactly one place. There is no second copy, no integrity check that
runs anywhere but this machine, and no way for a second authorized user to
obtain it except by re-purchasing it from the vendor.

That is acceptable for a single-machine feasibility study and not acceptable
for a phase that other people are expected to reproduce.

## Authoritative inputs

- [../../data-holdings-catalogue.md](../../data-holdings-catalogue.md) — what
  actually exists, and its measured integrity state.
- [../../market-state-reconstruction-contract.md](../../market-state-reconstruction-contract.md)
  — what the archive can and cannot supply.
- [../../architecture.md](../../architecture.md) — artifact identity and provenance
  rules any stored object must satisfy.
- [../../decision-log.md](../../decision-log.md) DEC-015 (the archive and
  everything derived from it stay out of Git; CI never sees it), DEC-011,
  DEC-028, DEC-030, DEC-031.

## The redistribution boundary

This is the constraint every other section is subordinate to.

**Licensed OPRA and Databento content may not be redistributed publicly, and
nothing in this plan may be built in a way that would allow it.** Concretely:

- No bucket, prefix, URL, or credential described here is ever public, and no
  public-read policy or unauthenticated endpoint is acceptable at any point.
- No signed URL is ever committed, logged, printed to a report, or shared.
- "Reproducible" in this task means **reproducible by an authorized user who
  independently holds the required vendor entitlements** — never "downloadable
  by anyone with the repository."
- A user without the entitlement must be able to read every manifest,
  checksum, schema, and row count, and must be unable to obtain a single byte
  of licensed content. The metadata layer and the content layer are separated
  for exactly this reason.
- Public, freely redistributable inputs (the FRED series) are **not** moved to
  a different regime by this separation; they simply stay with the archive so
  the archive remains one unit.

If any design choice would blur that boundary, the correct outcome is to record
that it cannot be done, not to weaken the boundary.

## Scope of the plan

### 1. Private object storage of authorized holdings

- One private S3 bucket (or equivalent), no public access at any level, object
  ownership enforced, versioning on, and a lifecycle policy that never
  silently deletes a referenced object.
- Layout mirrors the on-disk archive exactly — provider / dataset / product /
  schema / session — so that a local tree and a remote prefix are the same
  namespace. Raw vendor payloads (`.dbn.zst`) are stored **unchanged**; the
  archive's rule that payloads are never decompressed or transcoded is
  preserved end to end.
- Derived Parquet is stored under a separate prefix from raw, never mixed, and
  is always reconstructible from raw plus tracked code — so that losing it is
  an inconvenience, not a data loss.

### 2. Immutable versioned manifests and checksum verification

- Every stored set carries a manifest listing each object's relative path,
  byte size, and SHA-256 — the same discipline the local archive already
  passes (119/119 verified).
- Manifests are immutable once written and versioned by content digest. A
  change produces a new manifest; an existing one is never edited. This mirrors
  the repository's frozen-evidence rule.
- Verification is on **retrieval as well as upload**, and it fails closed: an
  object whose digest does not match its manifest entry is an error, never a
  warning.
- A manifest must be readable, and verifiable as internally consistent, by
  someone who cannot read the objects it describes.
- Manifests must be **scrubbed of machine-local absolute paths** before they
  leave the machine. Task 9D found the existing local manifests carry Windows
  source paths containing a user name; those fields are excluded or replaced,
  never uploaded as-is.

### 3. Access control, encryption, credential handling

- Least privilege, per-principal: a read-only principal for retrieval, a
  separate write principal for ingestion, and no principal that can delete a
  versioned object without an explicit administrative action.
- Encryption in transit and at rest, with a documented key regime.
- **No credential ever enters this repository** — not in code, not in a config,
  not in a test fixture, not in a committed environment file, and not in any
  report or artefact. Credentials come from the environment or an external
  secret store, and the ingestion path must fail with a clear error when they
  are absent rather than falling back to anything.
- Access is logged. An audit trail of who retrieved what is part of the design,
  not an optional extra.
- CI never receives any of these credentials and never reaches the bucket
  (DEC-015).

### 4. Retrieval and query support

- Retrieval by logical selector — provider, product, schema, session — rather
  than by opaque key, so that a caller expresses what it needs rather than
  where it happens to sit.
- Retrieval is verifying by default: bytes are checked against the manifest
  before they are handed to a caller.
- A dry-run/plan mode that reports exactly what would be fetched, its size, and
  its digests, without fetching, so cost and scope are visible first.
- Partial retrieval must be unmistakable as partial: an interrupted fetch can
  never present as a complete set, following the same manifest-written-last
  discipline the harvesting work already uses.

### 5. Entitlement-aware Databento ingestion

- Every request is expressed as a versioned request specification — dataset,
  symbols, `stype_in`, schema, UTC window, encoding, compression — matching the
  fields the existing local request manifests already record.
- **Entitlement is checked before a request is submitted**, and an
  un-entitled request fails with an explicit message naming what is missing.
  The pipeline never silently downgrades a request, substitutes a different
  schema, or falls back to a sample.
- Downloaded payloads are stored byte-for-byte, with the vendor's own sidecar
  metadata retained alongside them, exactly as the local archive does.
- Cost and volume are estimated and reported before download; a download is a
  manual, deliberate act.
- Transformation to Parquet is a separate, deterministic step over stored raw
  bytes, never fused into the download, so that raw and derived can always be
  reconciled independently.

### 6. Reproducible reconstruction by an authorized user

The target property, stated exactly: *a user who independently holds the
required vendor entitlements can, from this repository plus their own
entitlements, reconstruct an archive that matches the recorded manifests
byte-for-byte, and can verify that it does.*

Everything needed for that — request specifications, manifests, checksums,
schemas, and the transformation code — is tracked in Git. Nothing licensed is.

## Out of scope

- Implementing any of the above. This file is a plan.
- Creating a bucket, an IAM role, a key, or any cloud resource.
- Calling Databento or any other vendor API.
- Uploading, moving, copying, or deleting any local holding.
- Choosing a cloud vendor as a binding decision — "S3" here means private
  object storage with these properties; an equivalent service satisfying them
  is not excluded.
- Any change to what the archive is allowed to be used for. This task moves
  bytes around; it grants no new analytical permission, and
  [../../market-state-reconstruction-contract.md](../../market-state-reconstruction-contract.md)
  continues to bound what may be derived from them.
- Redistributing licensed content in any form, under any justification.

## Predeclared conventions

Fixed before any implementation begins, and **all still `OPEN`** — none is
invented here:

- **Storage vendor and region**, and whether one bucket or several. `OPEN`.
- **The retention and versioning policy**, including what may ever be deleted
  and by whom. `OPEN`.
- **The credential source** — environment, profile, or external secret store.
  `OPEN`.
- **The manifest schema version** for remote sets, and its relationship to the
  local `sha256sums.txt` convention already in use. `OPEN`.
- **The entitlement-check mechanism** — what is queried, and what a negative
  answer looks like. `OPEN`.
- **Cost ceiling and alerting**. `OPEN`.

Do not invent a value for any of these during implementation. Resolve and
version it first, and record the resolution in the decision log.

## Acceptance gates

Stated now so that a future implementation cannot define its own success after
the fact:

- **Entry:** a recorded decision to start this work, and every `OPEN`
  convention above resolved and versioned.
- **Redistribution gate:** no artefact produced by this task allows an
  unentitled party to obtain licensed content. Checked by inspection of the
  access policy, not by assertion.
- **Integrity gate:** an upload followed by a retrieval reproduces every
  object's SHA-256, and a deliberately corrupted object is rejected on
  retrieval.
- **Reconstruction gate:** an authorized user, starting from the repository and
  their own entitlements, reproduces a manifest-matching archive.
- **Secret gate:** no credential, signed URL, or machine-local absolute path
  appears in any committed file or any published report.
- **Exit:** the gates above pass and the design is recorded in a contract, not
  only in code.
- **Failure:** any gate that cannot be met is recorded as a negative result
  with its cause. It is not worked around by relaxing the redistribution
  boundary.

## Manual-run protocol

**None.** No numerical study is involved. Any future ingestion or upload is a
manual, deliberate, terminal-invoked act by a human with credentials: no hook,
no CI job, and no agent may launch, schedule, or gate on one, and no agent may
hold or request credentials.

## Artifact policy

Manifests, request specifications, schemas, and transformation code are tracked
in Git. Licensed payloads and everything derived from them stay out of Git and
out of CI (DEC-015). No report produced by this task may embed vendor content,
a credential, a signed URL, or a machine-local absolute path.

## Review requirements

`code-reviewer` for any storage, retrieval, or ingestion code, with explicit
attention to credential handling and failure modes. A **material approval** in
a fresh top-level session is required before any bucket is created or any
vendor request is submitted — the first irreversible act, not the last.

## Stop conditions

- Any step would make licensed content reachable without an entitlement: stop.
- A credential would have to be written into the repository, a test fixture, or
  a report: stop.
- An entitlement check cannot be performed before a paid request: stop; do not
  submit the request and find out afterwards.
- A manifest would have to be edited rather than superseded: stop.
- The work is being started without a recorded decision that it is needed:
  stop. This task is `On hold` by default.

## Completion report

If and when this task is ever run, record: which gates passed and failed; the
resolved value of every `OPEN` convention; the exact storage layout and access
policy; evidence that upload-and-retrieve reproduces digests; evidence that an
unentitled party cannot obtain content; an update to
[../../project-state.md](../../project-state.md); and a decision-log entry.

State explicitly whether any licensed content was made reachable to any party
that did not already hold the entitlement for it. The expected answer is no.
