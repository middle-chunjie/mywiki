---
type: concept
title: Greedy Search
slug: greedy-search
date: 2026-04-20
updated: 2026-04-20
aliases: [greedy substitution search, 贪心搜索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Greedy Search** (贪心搜索) — a search strategy that iteratively selects the locally best candidate according to a scoring objective without exhaustively exploring all combinations.

## Key Points

- [[jha-2023-codeattack-2206-00052]] uses a constrained greedy search to test substitutes for high-influence tokens one by one.
- The search terminates as soon as a candidate satisfies the required quality drop, similarity threshold, and perturbation budget.
- Greedy replacement is paired with vulnerability ranking, which keeps the black-box attack query-efficient.
- The paper uses greedy search over both operator-level and token-level perturbation candidates rather than full combinatorial optimization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jha-2023-codeattack-2206-00052]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jha-2023-codeattack-2206-00052]].
