---
type: concept
title: Recall@K
slug: recall-at-k
date: 2026-04-20
updated: 2026-04-20
aliases: [R@K, recall at rank k, 召回率@K]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Recall@K** (召回率@K) — a retrieval metric that measures whether at least one relevant item appears within the top `K` ranked results returned by a system.

## Key Points

- The paper evaluates both text-to-video and video-to-text retrieval with `R@1`, `R@5`, and `R@10`.
- It additionally reports `R@Sum`, the sum of `R@1`, `R@5`, and `R@10`, to summarize top-rank retrieval quality.
- All major comparisons are averaged over three random seeds (`0`, `42`, `123`) to reduce variance.
- The method's main empirical claim is that strong `R@K` can be preserved or improved while training only a small adapter parameter set.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-mvadapter-2301-07868]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-mvadapter-2301-07868]].
