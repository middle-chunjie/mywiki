---
type: concept
title: Grammar Splitter
slug: grammar-splitter
date: 2026-04-20
updated: 2026-04-20
aliases: [grammar splitter, 文法切分器]
tags: [parallelism, grammar]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Grammar Splitter** (文法切分器) — a parallelization method that partitions a PCFG into balanced, disjoint subgrammars so multiple workers can search without overlapping candidate programs.

## Key Points

- [[fijalkow-2022-scaling]] introduces grammar splitting as a generic wrapper that can parallelize any underlying search algorithm operating on a PCFG.
- A valid split must both partition the program space and keep probability mass balanced across workers so that each CPU contributes comparable search effort.
- The paper measures balance with `alpha = max mass / min mass` and uses hill climbing with swaps and refinements to approach `alpha_desired = 1.05`.
- In random-PCFG experiments, grammar splitting yields near-linear speedups, especially for SQRT Sampling and Heap Search.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fijalkow-2022-scaling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fijalkow-2022-scaling]].
