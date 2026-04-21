---
type: concept
title: Multi-Modal Learning
slug: multi-modal-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [Multimodal Learning, Multi-view Learning, 多模态学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Modal Learning** (多模态学习) — a learning paradigm that combines complementary signals from multiple modalities or views into joint or coordinated representations for downstream prediction.

## Key Points

- This paper treats code tokens, ASTs, and CFGs as three complementary modalities of the same source code snippet.
- MMAN uses joint representation for the three code modalities through attention-weighted fusion, then uses coordinated representation to align code and natural-language descriptions in a shared semantic space.
- The paper argues these modalities are complementary rather than conflicting, because tri-modal fusion outperforms every unimodal ablation.
- In this work, multi-modal learning improves retrieval accuracy while also making the representation more interpretable via modality-specific attention weights.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wan-2019-multimodal-1909-13516]]
- [[jiang-2022-crossmodal-2211-09623]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wan-2019-multimodal-1909-13516]].
