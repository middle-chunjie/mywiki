---
type: concept
title: Overlapped Clustering
slug: overlapped-clustering
date: 2026-04-20
updated: 2026-04-20
aliases: [重叠聚类, overlapping clustering]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Overlapped Clustering** (重叠聚类) — a clustering scheme that allows one document to belong to multiple clusters instead of enforcing a mutually exclusive partition.

## Key Points

- JTR uses overlapped clustering to relax the assumption that a document has only one semantic placement in the tree.
- The paper formulates cluster assignment with matrices `Y`, `M`, and `C`, then solves it approximately with `C* = Proj(Y_bar^T M)`.
- A document can appear in at most `lambda` leaf nodes, which increases recall by exposing multi-topic documents through multiple search paths.
- The ablation study shows that adding overlapped clustering raises doc-dev performance from `0.303` to `0.327` MRR@100 and from `0.678` to `0.743` R@100, with latency increasing from `5 ms` to `8 ms`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-constructing-2304-11943]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-constructing-2304-11943]].
