---
type: concept
title: Mean Reciprocal Rank
slug: mean-reciprocal-rank
date: 2026-04-20
updated: 2026-04-20
aliases: [MRR, 平均倒数排名]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Mean Reciprocal Rank** (平均倒数排名) — a retrieval metric that averages the reciprocal of the rank assigned to the first correct result for each query.

## Key Points

- The paper uses MRR as the main evaluation metric across six programming-language subsets of CodeSearchNet.
- MRR is defined as `MRR = (1 / |Q|) Σ_i 1 / Rank_i`, where `Rank_i` is the position of the true code snippet for query `i`.
- Soft-InfoNCE improves MRR by about 1-2 points over vanilla InfoNCE across CodeBERT, GraphCodeBERT, and UniXCoder.
- The metric is suitable here because code search is evaluated by how highly the correct snippet is ranked in a fixed candidate pool.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-rethinking-2310-08069]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-rethinking-2310-08069]].
