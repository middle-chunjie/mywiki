---
type: concept
title: Adaptive Retrieval
slug: adaptive-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval on demand, conditional retrieval]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Adaptive Retrieval** (自适应检索) — a retrieval strategy that triggers evidence access conditionally based on model state or predicted need instead of using a fixed retrieval schedule.

## Key Points

- SELF-RAG predicts whether retrieval is needed before each segment and can also continue using already fetched evidence.
- Retrieval can be controlled by thresholding the normalized probability of `Retrieve=Yes`.
- The method uses adaptive retrieval to reduce unnecessary evidence injection on creative or self-contained prompts.
- Ablation results show that replacing adaptive selection with naive `top-1` retrieval hurts performance on PopQA and ASQA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[asai-2023-selfrag-2310-11511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[asai-2023-selfrag-2310-11511]].
