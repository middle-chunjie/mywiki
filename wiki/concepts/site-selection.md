---
type: concept
title: Site Selection
slug: site-selection
date: 2026-04-20
updated: 2026-04-20
aliases: [perturbation site selection]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Site Selection** — the optimization problem of choosing which program locations should be perturbed under a fixed attack budget.

## Key Points

- The paper models site choice with a binary vector `z`, where `z_i = 1` means the `i`th perturbable site is selected.
- The perturbation budget is enforced by `1^T z <= k`, making `k` the attacker's strength.
- If a selected symbol occurs multiple times in the program, all of its occurrences are marked active in `z`.
- A central empirical claim of the paper is that optimizing sites, rather than picking them randomly, materially improves attack success.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[srikant-2021-generating-2103-11882]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[srikant-2021-generating-2103-11882]].
