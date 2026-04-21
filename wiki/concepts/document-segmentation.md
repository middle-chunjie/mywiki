---
type: concept
title: Document Segmentation
slug: document-segmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [document splitting, text segmentation, 文档分段]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Document Segmentation** (文档分段) — the process of partitioning a document into contiguous units that preserve coherence and are suitable for downstream indexing, retrieval, or generation.

## Key Points

- LumberChunker treats segmentation as a retrieval-critical design choice rather than a preprocessing detail.
- The paper focuses on long-form narrative books, where naive paragraph or fixed-window segmentation often cuts across semantic scene boundaries.
- Paragraph IDs make segmentation explicit so an LLM can point to a precise boundary without rewriting the text.
- The proposed method uses local context windows and repeated boundary prediction to segment the full document incrementally.
- Better segmentation improves both passage retrieval metrics and downstream RAG QA performance in the reported experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[duarte-2024-lumberchunker-2406-17526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[duarte-2024-lumberchunker-2406-17526]].
