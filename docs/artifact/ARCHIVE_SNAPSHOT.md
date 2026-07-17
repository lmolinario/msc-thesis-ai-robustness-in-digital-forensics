# Historical Repository Snapshot

## Authoritative research artifact

The `main` branch contains the current curated, documented, and publicly
presentable version of the research artifact associated with the MSc thesis:

> *Evaluating the Robustness of AI-Based Forensic Tools under Adversarial and
> Anti-Forensic Attacks*

Reviewers, the thesis supervisor, and the examination committee should use
`main` as the primary source for the current methodology, documentation,
software, and frozen experimental results.

## Preserved pre-cleanup snapshot

For provenance and audit purposes, the repository preserves the historical
branch:

[`archive/pre-commission-cleanup-2026-07-16`](https://github.com/lmolinario/msc-thesis-ai-robustness-in-digital-forensics/tree/archive/pre-commission-cleanup-2026-07-16)

The same repository state is also identified by the annotated tag:

[`snapshot/pre-commission-cleanup-2026-07-16`](https://github.com/lmolinario/msc-thesis-ai-robustness-in-digital-forensics/tree/snapshot/pre-commission-cleanup-2026-07-16)

Both references resolve to the immutable commit:

```text
309a4580537ebc3bb7950f29c090bb2729fc603b
```

The tag annotation records the snapshot as the repository state immediately
before public-release cleanup and data minimization.

This snapshot records the repository state immediately before the
pre-commission public-release cleanup, data minimization, and structural
reorganization.

## Contents and interpretation

The historical snapshot may contain materials subsequently removed, minimized,
or reorganized on `main`, including:

- historical working documentation and progress records;
- intermediate organizational material;
- less-minimized commercial-tool exports;
- image artifacts or Git LFS pointers later excluded from public distribution;
- redundant files retained at the time for operational traceability.

Their presence in the archive documents the research process. It does not mean
that every archived file is intended for current public reuse, redistribution,
or operational deployment.

## Intended use

The snapshot is retained for:

- provenance verification;
- methodological and repository audit;
- comparison with the curated public artifact;
- recovery of historical research material;
- examination by the supervisor or thesis commission when necessary.

It is **not** the current source of truth for the final documentation or
experimental reporting. Where the archive and `main` differ, `main` is the
authoritative version unless the research question specifically concerns the
historical repository state.

## Immutability

The branch is protected by the active GitHub ruleset
`Immutable pre-commission snapshot`, configured to:

- restrict updates;
- restrict deletions;
- block force pushes;
- allow no bypass actors.

The annotated tag provides a second stable reference to the same commit. A
separate tag ruleset should restrict updates and deletion of that tag.

No further commits are expected on the archive branch, and the snapshot tag
must not be moved to another commit.

## Data-access boundary

The archive supports provenance but does not replace the repository's
controlled-access policy. Availability of a historical reference, manifest,
export, Git LFS pointer, or path does not by itself grant authorization to use
or redistribute the corresponding underlying data.

For current access and restoration instructions, consult the data-access
documentation on `main`.