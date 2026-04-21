---
type: concept
title: kNN-LM
slug: knn-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [kNN-LM, k-nearest neighbors language model, k近邻语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**kNN-LM** (k近邻语言模型) — a retrieval-augmented language model that interpolates a base LM distribution with a distribution formed from the `k` nearest datastore entries for the current context.

## Key Points

- The paper uses `kNN-LM` as the retrieval baseline that RETOMATON accelerates and improves.
- In `kNN-LM`, every context vector queries an external datastore and weights retrieved values by negative distance.
- The paper keeps the same interpolation structure as `kNN-LM`, but replaces many repeated searches with automaton traversal.
- Full search retrieves `k_neigh = 1024` neighbors in the reported setup.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[alon-2022-neurosymbolic-2201-12431]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[alon-2022-neurosymbolic-2201-12431]].
