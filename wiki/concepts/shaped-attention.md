---
type: concept
title: Shaped Attention
slug: shaped-attention
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

**Shaped Attention** — a self-attention parameterization that adds an identity component and subtracts a centering matrix so skipless Transformers preserve informative signals more faithfully at initialization.

## Key Points

- The paper adopts the modification `` `A(X) <- alpha * I + beta * A(X) - gamma * C` `` instead of plain self-attention when the attention skip is removed.
- Query weights are initialized to `` `W^Q = 0` ``, so the attention logits start at zero and the identity term dominates the effective attention map.
- Trainable head-specific scalars `` `alpha_h` ``, `` `beta_h` ``, and `` `gamma_h` `` provide a small empirical gain over a shared-scalar formulation.
- Shaped Attention slightly outperforms the earlier Value-SkipInit attention variant while supporting skipless SAS and SAS-P blocks.
- In the paper's interpretation, Shaped Attention replaces part of the stabilizing role normally played by residual paths in deep Pre-LN Transformers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2023-simplifying-2311-01906]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2023-simplifying-2311-01906]].
