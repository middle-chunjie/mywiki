---
type: concept
title: Inter-Consistency
slug: inter-consistency
date: 2026-04-20
updated: 2026-04-20
aliases: [cross-perspective consistency, inter consistency, 跨视角一致性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Inter-Consistency** (跨视角一致性) — the degree of agreement between outputs from different reasoning perspectives that are intended to describe the same latent functionality.

## Key Points

- In [[unknown-nd-enhancing-2309-17272]], inter-consistency is measured between solution, specification, and test-case vertices in a 3-partite graph.
- Edge weights are computed by deterministic execution, for example checking whether a solution satisfies a generated test case or whether a specification accepts a candidate input-output pair.
- The paper encodes these agreements through `L_inter = f^T L f`, encouraging connected vertices with strong agreement to receive similar scores.
- Reported gains from MPSC-Uniform show that inter-consistency alone already improves the GPT-3.5-Turbo baseline substantially.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-enhancing-2309-17272]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-enhancing-2309-17272]].
