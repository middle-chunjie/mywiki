---
type: concept
title: Neural Type Recommendation
slug: neural-type-recommendation
date: 2026-04-20
updated: 2026-04-20
aliases: [learned type suggestion, 神经类型推荐]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Neural Type Recommendation** (神经类型推荐) — the use of trained machine learning models to propose candidate types for unresolved program variables.

## Key Points

- HiTYPER can consume Top-1, Top-3, or Top-5 predictions from Typilus or Type4Py during the recommendation phase.
- The paper treats neural outputs as recommendations rather than final predictions because they may violate typing rules.
- Recommendations are only requested for hot slots, reducing the number of places where learned guesses can introduce errors.
- A similarity-based correction step repairs explicitly invalid user-defined type predictions before rejection rules validate them.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
