---
type: concept
title: Best-of-N Search
slug: best-of-n-search
date: 2026-04-20
updated: 2026-04-20
aliases: [best-of-N, BoN search]
tags: [reasoning, search, inference]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Best-of-N Search** — an inference strategy that samples `N` candidate solutions and selects the one ranked highest by a scorer, verifier, or reward model.

## Key Points

- The paper evaluates both ORMs and PRMs by ranking uniformly sampled candidate solutions from a fixed generator.
- Reliability is measured by the fraction of problems solved after selecting the top-ranked candidate among `N` sampled traces.
- The PRM outperforms both the ORM and majority voting across all tested values of `N`, with the gap widening as `N` increases.
- The headline result uses best-of-`1860` search on a held-out subset of the MATH test set.
- The setup isolates reward-model quality from generator improvement because the generator is fixed throughout the experiments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lightman-2023-lets-2305-20050]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lightman-2023-lets-2305-20050]].
