---
type: concept
title: Rejection Sampling
slug: rejection-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [拒绝采样]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Rejection Sampling** (拒绝采样) — a sampling procedure that generates multiple candidate trajectories or outputs and keeps only those that satisfy a scoring or acceptance criterion.

## Key Points

- CoRAG uses rejection sampling to turn QA-only data into supervised retrieval chains without manual annotation.
- For each instance, the paper samples up to `16` chains and scores them by `log P(A | Q, Q_{1:L}, A_{1:L})`, where `A` is the gold final answer.
- The maximum chain length during data generation is randomly chosen from `[1, 5]`, with sub-query sampling temperature `0.7` and sub-answer temperature `0`.
- Chain generation stops early if an intermediate sub-answer exactly matches the gold answer or if the average conditional log-likelihood exceeds `-0.05`.
- The paper shows that better sampled chains matter: distilling chains from GPT-4o improves downstream CoRAG performance over using weaker generators.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2025-chainofretrieval-2501-14342]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2025-chainofretrieval-2501-14342]].
