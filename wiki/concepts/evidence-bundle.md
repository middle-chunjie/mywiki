---
type: concept
title: Evidence Bundle
slug: evidence-bundle
date: 2026-04-20
updated: 2026-04-20
aliases: [audit bundle, evidence manifest]
tags: [auditability, verification]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Evidence Bundle** (证据包) — the minimally sufficient package of artifacts, checks, hashes, and version identifiers used to justify skill promotion, reconstruct rewards, and support independent replay.

## Key Points

- The paper includes tool schemas, arguments, outputs or output hashes, contract checks, and verifier identity in the bundle.
- Evidence bundles tie promotion, reward computation, and reproducibility to the same trace instead of separate opaque processes.
- Hashes and canonical manifests are proposed to make evidence tamper-evident across asynchronous pipelines.
- Periodic reruns of bundle-linked tests are used to detect regressions after promotion.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2025-audited-2512-23760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2025-audited-2512-23760]].
