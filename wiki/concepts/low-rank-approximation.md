---
type: concept
title: Low-Rank Approximation
slug: low-rank-approximation
date: 2026-04-20
updated: 2026-04-20
aliases: [rank-k approximation, truncated approximation, 低秩近似]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Low-Rank Approximation** (低秩近似) — the replacement of a matrix by another matrix of smaller rank that preserves its dominant singular directions as well as possible under a chosen norm.

## Key Points

- LASER applies low-rank approximation after training by replacing a selected Transformer weight matrix with a rank-`r` truncation.
- The retained rank is set as `r = floor(ρ · rank_max(W))`, where `ρ` can be as small as `0.01` in the paper's search.
- In several later-layer MLP matrices, aggressive approximation preserves behavior and can even improve downstream accuracy.
- The paper interprets this as keeping lower-order signal while removing higher-order components that may encode noisy alternatives.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sharma-2023-truth-2312-13558]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sharma-2023-truth-2312-13558]].
