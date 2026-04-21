---
type: concept
title: Ordered Mask
slug: ordered-mask
date: 2026-04-20
updated: 2026-04-20
aliases: [nested mask, 有序掩码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Ordered mask** (有序掩码) — a differentiable masking scheme that preserves an importance ordering by allowing leading units to survive while progressively dropping lower-ranked units.

## Key Points

- [[unknown-nd-improving-2401-02993]] applies ordered masks over the top-`k` retrieved sentence representations on each embedding dimension.
- The method assumes the original retriever order is meaningful and refines it with trainable, differentiable masks rather than fully reranking all retrievals.
- The paper implements the mask with a chain of Bernoulli variables and a Gumbel-Softmax reparameterization.
- Masked retrieval representations are averaged and added to the sentence representation in the same way as the reranker-based fusion module.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-improving-2401-02993]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-improving-2401-02993]].
