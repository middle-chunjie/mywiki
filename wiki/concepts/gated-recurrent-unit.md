---
type: concept
title: Gated Recurrent Unit
slug: gated-recurrent-unit
date: 2026-04-20
updated: 2026-04-20
aliases: [GRU, 门控循环单元]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Gated Recurrent Unit** (门控循环单元) — a recurrent neural network cell that uses gating to control information flow across sequence positions with fewer parameters than an LSTM.

## Key Points

- The paper uses GRUs for both the encoder and decoder of the VAE-based semantic filter.
- The encoder is bidirectional so the final representation aggregates left and right context from the query text.
- The decoder autoregressively reconstructs the original query from latent variable `z`.
- GRUs are chosen as the sequence model that captures query semantics for anomaly-style filtering.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-importance-2202-06649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-importance-2202-06649]].
