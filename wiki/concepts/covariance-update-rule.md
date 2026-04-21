---
type: concept
title: Covariance Update Rule
slug: covariance-update-rule
date: 2026-04-20
updated: 2026-04-20
aliases: [covariance memory update, 协方差更新规则]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Covariance Update Rule** (协方差更新规则) — an associative-memory update that stores a key-value pair by adding an outer product to a memory matrix.

## Key Points

- In mLSTM the recurrent state update is `C_t = f_t C_{t-1} + i_t v_t k_t^T`, where the forget gate acts like decay and the input gate acts like a learned update rate.
- The paper connects this rule to classic associative memories, bidirectional associative memory, and fast-weight programmers.
- Retrieval uses a query vector `q_t` together with a normalizer `n_t` so the output is scaled by `max(|n_t^T q_t|, 1)`.
- This mechanism lets xLSTM approximate attention-like key-value storage while remaining a recurrent architecture with linear sequence processing.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[beck-2024-xlstm-2405-04517]].
