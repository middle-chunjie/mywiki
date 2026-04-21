---
type: entity
title: T5-XXL
slug: t5-xxl
date: 2026-04-20
entity_type: model
aliases: [T5 XXL, T5-XXL 11B]
tags: []
---

## Description

T5-XXL is the `11B`-parameter T5 v1.1 encoder-decoder model used as the target model `M_p` for the paper's walltime experiments on translation and summarization.

## Key Contributions

- Serves as the main production-scale target model demonstrating `2.3x-3.4x` exact decoding speedups.
- Anchors the empirical trade-off between acceptance rate `α`, draft cost `c`, and choice of `γ`.

## Related Concepts

- [[sequence-to-sequence]]
- [[speculative-decoding]]
- [[transformer]]

## Sources

- [[leviathan-2023-fast-2211-17192]]
