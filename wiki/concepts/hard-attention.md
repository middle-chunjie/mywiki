---
type: concept
title: Hard Attention
slug: hard-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [Hard Attention, 硬注意力]
tags: [attention, transformer, interpretability]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hard Attention** (硬注意力) — an attention mechanism that makes discrete selection decisions, typically by choosing a single position with an argmax rather than averaging over many positions with soft weights.

## Key Points

- [[friedman-2023-transformer-2306-01128]] uses hard attention to ensure that categorical attention heads output discrete variables that can be decompiled into programs.
- The categorical attention output is defined as `` `A_i = One-hot(argmax_j S_{i,j})` ``, making each query attend to exactly one key position.
- When no key matches, the implementation defaults to the beginning-of-sequence token; when multiple keys match, it selects the nearest match.
- The paper also relaxes this discrete argmax during training with Gumbel-Softmax so the model remains trainable end to end.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-transformer-2306-01128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-transformer-2306-01128]].
