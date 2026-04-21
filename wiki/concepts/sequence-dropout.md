---
type: concept
title: Sequence Dropout
slug: sequence-dropout
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 序列丢弃
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sequence Dropout** (序列丢弃) — a training regularization method that randomly replaces entire contextual sequence inputs with a null representation so the model remains robust when some context is missing.

## Key Points

- In CDE, contextual document embeddings `M_1(d*)` are replaced by a null token `v_empty` with uniform probability `p = 0.005`.
- This prevents the contextual encoder from overfitting to the assumption that a full context set is always available.
- The same mechanism lets the model degrade gracefully to a non-contextual biencoder at test time by replacing all contextual inputs with null tokens.
- The paper uses this as a practical bridge between corpus-aware indexing and settings where no target-corpus context can be supplied.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[morris-2024-contextual-2410-02525]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[morris-2024-contextual-2410-02525]].
