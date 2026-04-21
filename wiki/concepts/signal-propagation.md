---
type: concept
title: Signal Propagation
slug: signal-propagation
date: 2026-04-20
updated: 2026-04-20
aliases: [信号传播]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Signal Propagation** (信号传播) — a theoretical framework for tracking how activation norms and inter-input correlations evolve across depth, especially at initialization, to predict whether deep networks remain trainable.

## Key Points

- The paper uses signal propagation as the main theoretical lens for explaining why standard Pre-LN residual branches help deep Transformer optimization.
- Removing the attention skip connection is framed as dangerous because it induces degeneracies such as rank collapse unless the attention matrix is biased toward identity.
- The authors use signal-propagation arguments to justify Shaped Attention, zero-initialized query weights, and downweighted MLP branch gains in skipless blocks.
- The theory successfully motivates several simplifications, but the paper stresses that it does not fully explain observed optimization-speed differences under AdamW during training.
- The work positions architectural simplification as a way to narrow the gap between tractable theory and the more complex Transformer blocks used in practice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2023-simplifying-2311-01906]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2023-simplifying-2311-01906]].
