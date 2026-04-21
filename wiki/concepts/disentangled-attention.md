---
type: concept
title: Disentangled Attention
slug: disentangled-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [disentangled self-attention, 解耦注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Disentangled Attention** (解耦注意力) — an attention design that models token-content and positional information with separate interactions inside the attention score instead of mixing them by simple input-level embedding addition.

## Key Points

- The paper adopts disentangled attention as the backbone for injecting tree structure into Transformer self-attention.
- Absolute tree position is represented by a dedicated position-position interaction `` `beta_ij = (a_i W_a^Q)(a_j W_a^K)^T` `` rather than by summing position vectors with node embeddings.
- Local tree relations are also encoded through separate relative-position interactions `` `gamma_ij` ``, keeping content and structure partially factorized.
- This design is motivated by prior NLP work arguing that naive addition of token and position embeddings creates noisy mixed correlations.
- The paper's results suggest that disentangled attention remains effective when the positional objects are tree coordinates instead of linear token offsets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-nd-rethinking]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-nd-rethinking]].
