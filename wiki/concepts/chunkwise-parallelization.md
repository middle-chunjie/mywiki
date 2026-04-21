---
type: concept
title: Chunkwise parallelization
slug: chunkwise-parallelization
date: 2026-04-20
updated: 2026-04-20
aliases: [chunk-wise parallelization]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Chunkwise parallelization** — a computation strategy that partitions a sequence into chunks so recurrent state updates can be parallelized within each block while preserving causal dependence across blocks.

## Key Points

- Kimi Linear derives a chunkwise form of KDA that separates inter-chunk recurrent state propagation from intra-chunk parallel attention computation.
- The method uses WY-style packing and a UT transform to compress repeated rank-1 state transforms into matrix operations that map well to Tensor Cores.
- The paper's FLOPs discussion assumes chunk size `C = 64` and shows that KDA removes several matrix multiplications compared with a more general DPLR implementation.
- This chunkwise formulation is central to the claim that KDA can beat prior fine-grained gated linear attention in kernel efficiency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[team-2025-kimi-2510-26692]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[team-2025-kimi-2510-26692]].
