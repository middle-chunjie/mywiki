---
type: concept
title: Next-Token Prediction
slug: next-token-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [NTP, 下一词预测]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Next-Token Prediction** (下一词预测) — an autoregressive objective that trains a model to predict token `x_l` from the prefix `x_<l` of a sequence.

## Key Points

- REVELA starts from the classical objective `P(x_l^i | x_{<l}^i)` and extends it to condition on other documents in the batch.
- The paper uses next-token prediction as the supervision signal that indirectly trains the retriever.
- Because cross-document context is weighted by retriever similarity, lower NTP loss encourages better retrieval structure.
- The framework keeps the standard decoder-only causal setup while expanding the conditioning context beyond the local sequence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cai-2026-revela-2506-16552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cai-2026-revela-2506-16552]].
