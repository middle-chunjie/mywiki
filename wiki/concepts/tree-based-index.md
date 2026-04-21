---
type: concept
title: Tree-Based Index
slug: tree-based-index
date: 2026-04-20
updated: 2026-04-20
aliases: [树形索引, tree index]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tree-Based Index** (树形索引) — a hierarchical retrieval index that groups documents into coarse-to-fine clusters so search can prune large parts of the corpus instead of scoring every document.

## Key Points

- JTR builds the index by recursively applying `k`-means until each leaf contains at most `gamma` documents.
- Each node stores a trainable embedding, and retrieval scores nodes with `e_c^T Phi(q)` before descending to children.
- The paper argues that tree quality depends on preserving a maximum-heap-like ordering so beam search keeps the correct path alive.
- Relative to prior tree baselines such as Annoy and FLANN, the JTR tree is optimized with supervised retrieval signals instead of only unsupervised clustering criteria.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-constructing-2304-11943]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-constructing-2304-11943]].
