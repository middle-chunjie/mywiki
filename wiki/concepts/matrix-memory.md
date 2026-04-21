---
type: concept
title: Matrix Memory
slug: matrix-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [matrix-valued memory, 矩阵记忆]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Matrix Memory** (矩阵记忆) — a recurrent state representation that stores information in a matrix rather than a scalar or vector cell, enabling key-value style retrieval with higher capacity.

## Key Points

- mLSTM upgrades the LSTM cell state from a scalar memory to a matrix `C_t` so stored information can be retrieved by a query vector.
- The matrix memory is updated with gated outer products `v_t k_t^T`, making the recurrent state resemble associative-memory and fast-weight mechanisms.
- The paper uses matrix memory to address LSTM's poor rare-token behavior and limited storage capacity.
- The tradeoff is higher computation, since the memory state and updates scale as `d x d` even though the recurrence can still be parallelized across sequence positions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[beck-2024-xlstm-2405-04517]].
