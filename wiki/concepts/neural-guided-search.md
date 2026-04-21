---
type: concept
title: Neural-Guided Search
slug: neural-guided-search
date: 2026-04-20
updated: 2026-04-20
aliases: [neural guided search, 神经引导搜索]
tags: [search, deep-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Neural-Guided Search** (神经引导搜索) — a hybrid inference strategy in which a learned model predicts search-relevant signals that are then used to prioritize or constrain a symbolic search procedure.

## Key Points

- [[balog-2017-deepcoder-1611-01989]] predicts which DSL functions are likely to appear in the target program instead of predicting source code directly.
- These predicted marginals are used to order DFS expansions and to expand active function sets in sort-and-add search.
- The paper integrates the same guidance signal into multiple back-end solvers, including enumeration, [[sketch]], and [[lambda-squared]].
- Empirically, the largest gains come from sort-and-add style procedures, where the predicted ranking of functions directly reduces the effective search space.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[balog-2017-deepcoder-1611-01989]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[balog-2017-deepcoder-1611-01989]].
