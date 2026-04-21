---
type: concept
title: Replay-Based Verification
slug: replay-based-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [replay verification, verifier replay]
tags: [verification, agents]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Replay-Based Verification** (基于回放的验证) — validating an agent behavior or candidate skill by rerunning it under a controlled harness and checking contracts, tool schemas, and deterministic outcomes from logged artifacts.

## Key Points

- ASG-SI promotes skills only after replay passes held-out tasks and interface checks.
- Replay decouples verification from the internal state of the model that originally generated the trajectory.
- The same replay artifacts are reused for reward reconstruction and later regression testing.
- The paper treats verifier versioning and harness specification as part of what makes replay evidence meaningful.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[huang-2025-audited-2512-23760]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[huang-2025-audited-2512-23760]].
