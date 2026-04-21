---
type: concept
title: In-Batch Attention
slug: in-batch-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [batch attention, 批内注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**In-Batch Attention** (批内注意力) — an attention mechanism that allows a sequence to attend to other sequences in the same training batch, rather than only to its own tokens.

## Key Points

- REVELA adds an in-batch branch `h_i^l` alongside standard self-attention output `e_i^l` in each transformer layer.
- Cross-document attention is computed from the current sequence's queries to cached keys and values from other sequences in the batch.
- Retriever-produced similarity scores weight the contribution of each external sequence before aggregation.
- The final in-batch representation is the sum of within-sequence self-context and similarity-weighted cross-document context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cai-2026-revela-2506-16552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cai-2026-revela-2506-16552]].
