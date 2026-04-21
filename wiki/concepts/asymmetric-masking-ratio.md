---
type: concept
title: Asymmetric Masking Ratio
slug: asymmetric-masking-ratio
date: 2026-04-20
updated: 2026-04-20
aliases: [非对称掩码比例]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Asymmetric Masking Ratio** (非对称掩码比例) — a masking strategy that applies different corruption levels to different modules so each module is forced to solve a distinct subproblem.

## Key Points

- RetroMAE masks the encoder input moderately at about `15%-30%`, preserving enough content to build a sentence embedding.
- The decoder input is masked much more aggressively at about `50%-70%`, making reconstruction depend on the encoder embedding.
- With enhanced decoding, the best decoder masking ratio in ablation is `0.5`; without enhanced decoding, the best reported value is `0.7`.
- For the encoder, increasing the masking ratio from `0.15` to `0.3` helps retrieval, but a very aggressive ratio such as `0.9` damages performance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2022-retromae-2205-12035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2022-retromae-2205-12035]].
