---
type: concept
title: Word Embedding Space
slug: word-embedding-space
date: 2026-04-20
updated: 2026-04-20
aliases: [word embedding space, 词嵌入空间, embedding space]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Word Embedding Space** (词嵌入空间) — the vector space in which discrete vocabulary items are represented as continuous embeddings used to compute language-model token scores.

## Key Points

- LM-Switch operates directly in embedding space by learning a linear matrix `W` that perturbs token embeddings.
- The paper's core formula is `e'_v = e_v + εWe_v`, connecting condition control to embedding geometry.
- For runtime efficiency, the implementation applies the same transformation on contextual vectors before the LM head instead of explicitly rewriting every embedding.
- The transfer method treats `W` as a relation in embedding space and ports it across models with the aligned transform `H^TWH`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[han-2024-word-2305-12798]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[han-2024-word-2305-12798]].
