---
type: concept
title: Multimodal Contrastive Learning
slug: multimodal-contrastive-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [cross-modal contrastive learning, 多模态对比学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multimodal Contrastive Learning** (多模态对比学习) — a representation learning objective that aligns matched examples from different modalities while separating mismatched pairs in a shared embedding space.

## Key Points

- CoCoSoDa applies the objective to code snippets and natural-language queries rather than to two code views only.
- The training loss combines inter-modal terms for code-query pairing with intra-modal terms for code-code and query-query geometry.
- Cosine similarity with temperature `τ = 0.07` is used inside InfoNCE-style losses on both the code side and the query side.
- The paper credits this objective for improving both pairwise alignment and unimodal uniformity, which correlates with higher code-search accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2023-cocosoda-2204-03293]]
- [[unknown-nd-code-2402-01935]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2023-cocosoda-2204-03293]].
