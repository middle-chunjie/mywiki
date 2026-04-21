---
type: concept
title: Self-Training
slug: self-training
date: 2026-04-20
updated: 2026-04-20
aliases: [self training, 自训练]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Self-Training** (自训练) — a semi-supervised learning strategy in which a model generates pseudo-labels or auxiliary supervision that is reused to improve the model or a downstream pipeline.

## Key Points

- S2M adopts RGX, a cooperative self-training framework, as its candidate single-turn QA generator.
- The generated QA pairs are treated as noisy intermediate supervision rather than as final conversational data.
- The paper adds score-based filtering and redundancy merging on top of the self-training outputs before reassembly.
- This design lets S2M reuse a strong single-turn generator while moving beyond the limitations of direct single-turn augmentation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-sm-2312-16511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-sm-2312-16511]].
