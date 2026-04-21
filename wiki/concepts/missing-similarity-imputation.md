---
type: concept
title: Missing Similarity Imputation
slug: missing-similarity-imputation
date: 2026-04-20
updated: 2026-04-20
aliases: [imputed missing similarity, 缺失相似度插补]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Missing Similarity Imputation** (缺失相似度插补) — an inference heuristic that substitutes an estimated similarity for query-document token pairs that were not retrieved, allowing approximate late-interaction scoring without loading all document tokens.

## Key Points

- XTR uses imputation because candidate documents often expose only a subset of their relevant token matches after first-stage retrieval.
- For a missing query-token match, the paper sets an imputed value `m_i` bounded by the last retrieved top-`k'` score for that query token.
- This lets XTR approximate the sum-of-max score while reusing retrieval scores instead of recomputing all pairwise token similarities.
- Ablation results show the method is effective for XTR but not sufficient on its own for a model trained with the original ColBERT objective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-nd-rethinking]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-nd-rethinking]].
