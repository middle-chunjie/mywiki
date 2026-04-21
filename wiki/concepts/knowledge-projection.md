---
type: concept
title: Knowledge Projection
slug: knowledge-projection
date: 2026-04-20
updated: 2026-04-20
aliases: [KP, knowledge projection, 知识投影]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Projection** (知识投影) — a set-valued operator `R(V)` that returns all entities connected to a seed set `V` through relation subspace `R`, serving as the atomic unit of WebShaper's information-seeking formalization.

## Key Points

- WebShaper defines a KP as `R(V) = \{u \mid \exists v \in V, (u,v)\in R \text{ or } (v,u)\in R\}` over the universal entity set.
- The paper treats unions of projections with the same relation as mergeable through `R(S_1) \cup R(S_2) = R(S_1 \cup S_2)`.
- KP triplets `[X, r, S]` provide a prompt-friendly representation for recursive information-seeking tasks.
- Entire questions are built by intersecting or recursively composing multiple KPs around target variable `T`.
- The Expander uses KP structure to decide what new evidence to retrieve and how to validate expanded sub-questions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tao-2025-webshaper-2507-15061]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tao-2025-webshaper-2507-15061]].
