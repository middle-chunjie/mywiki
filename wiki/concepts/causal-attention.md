---
type: concept
title: Causal Attention
slug: causal-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [causal mask, uni-directional attention, 单向注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Causal Attention** (单向注意力) — an attention pattern in which each token can attend only to earlier positions, enforcing autoregressive next-token prediction.

## Key Points

- The paper identifies causal attention as the reason classical embeddings from decoder-only models miss information that appears later in the sequence.
- Under mean pooling, early token states cannot encode later evidence, so late discriminatory content gets diluted in the pooled vector.
- Under last-token pooling, the representation overemphasizes the tail of the sequence and performs poorly in zero-shot embedding extraction.
- Echo embeddings work around this constraint without removing the causal mask by making the second occurrence attend to the first occurrence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[springer-2024-repetition-2402-15449]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[springer-2024-repetition-2402-15449]].
