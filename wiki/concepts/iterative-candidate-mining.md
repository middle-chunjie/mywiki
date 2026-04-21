---
type: concept
title: Iterative Candidate Mining
slug: iterative-candidate-mining
date: 2026-04-20
updated: 2026-04-20
aliases: [self-guided mining, 迭代候选挖掘]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Iterative Candidate Mining** (迭代候选挖掘) — a training strategy that repeatedly refreshes candidate sets using the current retriever so the model can discover stronger positives and harder negatives over time.

## Key Points

- UDR does not keep a fixed candidate pool based only on surface similarity or target overlap.
- At each iteration, it retrieves `top-K` candidates from the full task dataset under the current retriever, rescoring them with the LM afterward.
- This lets the model surface useful demonstrations whose outputs may look dissimilar but still help the task.
- The paper reports a clear ablation gap: removing self-guided mining reduces average performance from `58.4` to `56.5`.
- The procedure balances quality and cost by rescoring `K = 50` candidates but sampling only `l = 8` per training step.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-unified]].
