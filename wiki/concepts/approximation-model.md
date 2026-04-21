---
type: concept
title: Approximation Model
slug: approximation-model
date: 2026-04-20
updated: 2026-04-20
aliases: [draft model, proposal model, 近似模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Approximation Model** (近似模型) — a cheaper model used to approximate a slower target model closely enough to propose candidate outputs that can later be verified or corrected.

## Key Points

- In speculative decoding the approximation model is denoted `M_q`, while the exact target model is `M_p`.
- Its usefulness is governed by the agreement statistic `α` and cost ratio `c`, not by accuracy alone.
- The paper finds that draft models roughly two orders of magnitude smaller than the target often give the best latency trade-off.
- Approximation models need not share the same architecture as the target; unigram, bigram, copy heuristics, and non-autoregressive models are all discussed as viable drafts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[leviathan-2023-fast-2211-17192]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[leviathan-2023-fast-2211-17192]].
