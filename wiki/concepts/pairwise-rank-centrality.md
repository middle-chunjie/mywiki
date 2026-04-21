---
type: concept
title: Pairwise Rank Centrality
slug: pairwise-rank-centrality
date: 2026-04-20
updated: 2026-04-20
aliases: [rank centrality, pairwise rank aggregation]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Pairwise Rank Centrality** — a ranking method that converts pairwise comparisons into a Markov chain over items and uses the stationary distribution of that chain as the aggregate preference score.

## Key Points

- PRISM converts model ratings into pairwise battle outcomes between LLMs and estimates transition probabilities from empirical win fractions.
- The paper uses regularization `alpha = 1` and tie threshold `t = 5` when building the comparison graph.
- The resulting stationary distribution is intended to be more order-robust and tournament-comparable than online Elo-style updates.
- Even with this aggregation method, model rankings remain sensitive to topic mix, geography, and cohort composition.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kirk-2024-prism-2404-16019]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kirk-2024-prism-2404-16019]].
