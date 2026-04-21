---
type: concept
title: Dynamic NTK Scaling
slug: dynamic-ntk-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [dynamic ntk interpolation, 动态NTK缩放]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dynamic NTK Scaling** — a RoPE extrapolation method that rescales positional frequencies as sequence length grows, preserving short-context behavior while extending usable context length at inference time.

## Key Points

- The paper applies Dynamic NTK scaling at inference so a model trained with `L = 2048` can be used at up to `L' = 8192`.
- Its presentation builds on NTK-aware scaling by introducing a hyperparameter `α` into the base-rescaling rule.
- The method keeps original position embeddings for inputs within the pretrained context length and scales them gradually only when the sequence exceeds that range.
- For `nomic-embed-text-v1`, the evaluation setting uses `α = 2` on texts longer than `2048` tokens.
- This extrapolation strategy is central to the paper because the model is never trained natively at 8k context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[nussbaum-2025-nomic-2402-01613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[nussbaum-2025-nomic-2402-01613]].
