---
type: concept
title: Molecular Graph
slug: molecular-graph
date: 2026-04-20
updated: 2026-04-20
aliases: [分子图]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Molecular Graph** (分子图) — a graph representation of a molecule in which nodes correspond to atoms and edges encode inter-atomic relations such as bonds or geometric distances.

## Key Points

- The paper formalizes each molecule as a weighted graph `G = (V, E)` where node features encode atom types and edges carry distance information.
- Edge weights are defined from 3D atomic coordinates, allowing geometry to influence message passing.
- This graph view is the basis for both node-level reconstruction and graph-level pooling in ASGN.
- The paper treats the chemical space as a collection of such molecular graphs, split into labeled and unlabeled subsets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hao-2020-asgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hao-2020-asgn]].
