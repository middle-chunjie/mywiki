---
type: concept
title: Sampling Decoding
slug: sampling-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [stochastic decoding, sampled decoding, 采样解码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sampling Decoding** (采样解码) — a generation strategy that samples next tokens from a model distribution instead of deterministically taking the highest-probability continuation, in order to produce diverse candidate outputs.

## Key Points

- [[weng-2023-large-2212-09561]] uses sampling decoding in forward reasoning to produce multiple candidate answers for later reranking.
- The paper sets `K = 5` candidate conclusions per input so that backward verification has a non-trivial answer set to compare.
- No top-k truncation is used in the reported implementation, emphasizing diversity over deterministic decoding.
- The method's effectiveness depends on sampling producing at least one correct candidate answer among the generated conclusions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weng-2023-large-2212-09561]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weng-2023-large-2212-09561]].
