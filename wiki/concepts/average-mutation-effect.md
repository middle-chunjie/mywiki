---
type: concept
title: Average Mutation Effect
slug: average-mutation-effect
date: 2026-04-20
updated: 2026-04-20
aliases: [AME]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Average Mutation Effect** — the expected change in task correctness induced by a specific semantics-preserving mutation, used as a scalar measure of concept understanding.

## Key Points

- [[hooda-2024-do-2402-05980]] defines mutation effect as the absolute difference between attribution outcomes on original and counterfactual inputs.
- AME averages this per-example effect over all valid problem instances for a predicate-specific mutation.
- Lower AME indicates better robustness to the targeted predicate perturbation.
- The paper reports AME values above `30%` for some models and mutations, showing large concept-level brittleness despite reasonable baseline accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
