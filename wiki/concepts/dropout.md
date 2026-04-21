---
type: concept
title: Dropout
slug: dropout
date: 2026-04-20
updated: 2026-04-20
aliases: [随机失活]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Dropout** (随机失活) — a stochastic regularization technique that randomly zeroes a subset of hidden activations during training, here functioning as the only augmentation used to form positive pairs.

## Key Points

- SimCSE uses standard Transformer dropout, without any extra augmentation module, to create two different views of the same sentence.
- The unsupervised objective relies on independently sampled masks `z` and `z'`; if the same mask is reused, performance collapses from `82.5` to `43.6` on STS-B dev.
- The default dropout rate `p = 0.1` is the best setting among the tested values, outperforming `p = 0`, `0.01`, `0.05`, `0.15`, `0.2`, and `0.5`.
- The paper interprets dropout as minimal augmentation that preserves sentence semantics while preventing representation collapse.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gao-2022-simcse-2104-08821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gao-2022-simcse-2104-08821]].
