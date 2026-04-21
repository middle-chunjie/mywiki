---
type: concept
title: InfoNCE
slug: infonce
date: 2026-04-20
updated: 2026-04-20
aliases: [InfoNCE loss, 信息噪声对比估计]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**InfoNCE** — a contrastive objective that classifies one positive sample against multiple negatives and yields a lower bound on mutual information between context and target variables.

## Key Points

- CPC defines InfoNCE as `` `L_N = -E_X log( f_k(x_{t+k}, c_t) / sum_{x_j in X} f_k(x_j, c_t) )` `` over one positive and `N-1` negatives.
- The paper shows that the optimal scoring function estimates the density ratio `` `p(x_{t+k}|c_t) / p(x_{t+k})` ``.
- Under this derivation, minimizing InfoNCE maximizes the bound `` `I(x_{t+k}; c_t) >= log(N) - L_N` ``.
- Larger negative sets make the mutual-information lower bound tighter, which is why in-batch negatives are important in CPC.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[oord-2019-representation-1807-03748]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[oord-2019-representation-1807-03748]].
