---
type: concept
title: Context Utilization
slug: context-utilization
date: 2026-04-20
updated: 2026-04-20
aliases: [上下文利用率]
tags: [rag, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Utilization** (上下文利用率) — the proportion of retrievably supported ground-truth claims that a generator actually uses in its response.

## Key Points

- RAGChecker defines context utilization as the ratio between ground-truth claims supported by retrieved chunks and repeated in the response, and all ground-truth claims supported by retrieved chunks.
- The metric isolates whether a generator can convert available evidence into correct output instead of merely receiving good retrieval.
- Among generator-side metrics in the paper, context utilization shows the strongest empirical correlation with overall F1.
- Context utilization is comparatively stable across BM25 and E5-Mistral, suggesting that stronger retrieval mainly improves recall by surfacing more usable evidence.
- Prompt changes can raise context utilization (`59.2 -> 63.7` in one diagnosis setting), but this often worsens noise sensitivity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ru-2024-ragchecker-2408-08067]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ru-2024-ragchecker-2408-08067]].
