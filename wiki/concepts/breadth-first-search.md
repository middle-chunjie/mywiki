---
type: concept
title: Breadth-First Search
slug: breadth-first-search
date: 2026-04-20
updated: 2026-04-20
aliases: [BFS, 广度优先搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Breadth-First Search** (广度优先搜索) — a search strategy that expands all candidate states at the current depth before moving deeper in the search tree.

## Key Points

- CoRAG implements a BFS-style tree-search decoder to explore multiple retrieval-chain branches at test time.
- At each step, the current state is expanded by sampling several sub-queries, then each expanded state is evaluated through rollout-based continuation.
- The paper scores each expanded state by the average penalty over its rollouts and keeps the state with the lowest average penalty for further expansion.
- In the reported implementation, the expansion size is `4`, the number of rollouts is `2`, and each rollout is limited to at most `2` additional steps.
- BFS-based tree search improves some QA results but is significantly more expensive than greedy decoding or best-of-`N` sampling.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2025-chainofretrieval-2501-14342]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2025-chainofretrieval-2501-14342]].
