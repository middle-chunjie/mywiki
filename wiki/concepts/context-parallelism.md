---
type: concept
title: Context Parallelism
slug: context-parallelism
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context parallelism** (上下文并行) — a long-context training strategy that shards sequence positions across devices while keeping the model computation coordinated across those shards.

## Key Points

- Composer 2 uses Context Parallelism as its main long-context scaling axis because it requires less communication than Tensor Parallelism and preserves full hidden dimensions locally.
- In the paper's MLA implementation, local KV latent vectors are computed first, then all-gathered across CP ranks before KV projections are applied.
- To balance causal-attention work, the sequence is split into `2 * CP` chunks and each rank processes chunk `i` together with chunk `2 * CP - 1 - i`.
- The reported training setup uses `CP = 2` during continued pretraining and `CP = 8` during RL, with the CP dimension folded into the FSDP dimension for memory savings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[research-2026-composer-2603-24477]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[research-2026-composer-2603-24477]].
