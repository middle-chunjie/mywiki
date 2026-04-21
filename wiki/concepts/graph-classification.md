---
type: concept
title: Graph Classification
slug: graph-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [graph classification, 图分类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Graph Classification** (图分类) - a supervised task that assigns a label to an entire graph rather than to individual nodes or edges.

## Key Points

- RAGraph includes graph classification in the same unified task formulation as node classification and link prediction.
- For graph-level tasks, the paper adds a full-link virtual center node inside the query graph to receive retrieved information.
- The fused output vector is compared against class prototypes in a `5`-shot setting.
- The method reports strong graph-classification results on PROTEINS, COX2, ENZYMES, and BZR, especially on COX2 and BZR.
- Retrieved toy graphs are intended to supply transferable context when the target graph differs from training graphs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jiang-2024-ragraph-2410-23855]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jiang-2024-ragraph-2410-23855]].
