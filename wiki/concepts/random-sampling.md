---
type: concept
title: Random Sampling
slug: random-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [RS, 随机采样]
tags: [decoding, generation, efficiency]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Random Sampling** (随机采样) — a stochastic decoding strategy that samples next tokens from a model's predictive distribution instead of always selecting the highest-scoring continuation.

## Key Points

- The paper benchmarks random sampling as one of three decoding strategies for synthetic question generation.
- RS is the most efficient decoding method in the study, producing accepted questions far faster than beam search or contrastive search.
- Although RS has lower average `hitsR` than beam search, it still yields strong downstream reranking results when paired with an appropriate small generator.
- The authors suggest that RS may preserve useful diversity, which could explain why some RS-generated datasets perform better than expected in downstream evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
