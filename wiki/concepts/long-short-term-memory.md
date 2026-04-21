---
type: concept
title: Long Short-Term Memory
slug: long-short-term-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [LSTM, long short-term memory, 长短期记忆]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Long Short-Term Memory** (长短期记忆) — a gated recurrent neural network that maintains an additive cell state to preserve gradients and store sequence information over long horizons.

## Key Points

- The paper treats the classic LSTM update `c_t = f_t c_{t-1} + i_t z_t` and output gating as the starting point for all xLSTM variants.
- Vanilla LSTM is argued to suffer from three limitations in modern language modeling: weak revision of past storage decisions, scalar memory capacity, and limited parallelizability.
- xLSTM keeps the LSTM intuition of gated recurrent memory but replaces standard sigmoid-gated scalar storage with exponential gating, normalization, and in mLSTM a matrix-valued state.
- The paper frames LSTM as historically important for early language modeling and asks how far it can be pushed with modern residual backbones and scaling practice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[beck-2024-xlstm-2405-04517]].
