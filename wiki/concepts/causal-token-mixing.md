---
type: concept
title: Causal Token Mixing
slug: causal-token-mixing
date: 2026-04-20
updated: 2026-04-20
aliases: [causal mixing, 因果式token混合]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Causal Token Mixing** — a token interaction regime in which the representation at position `t` can only aggregate information from positions up to `t`, not from future positions.

## Key Points

- The paper formalizes causal mixing as `y_t = f(x_1, ..., x_t)` and contrasts it with fully visible mixing `y_t = f(x_1, ..., x_T)`.
- It argues that Mamba's recurrent SSM is inherently causal, while attention can operate in fully visible mode by default and be switched to causal mode with masks.
- The authors claim causal mixing is appropriate for autoregressive generation but mismatched to standard visual understanding tasks where the whole image is available at once.
- Their ViT causalization experiment is used as qualitative evidence that forcing causal mixing on ImageNet harms classification performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-mambaout-2405-07992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-mambaout-2405-07992]].
