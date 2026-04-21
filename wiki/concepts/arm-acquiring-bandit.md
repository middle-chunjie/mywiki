---
type: concept
title: Arm-Acquiring Bandit
slug: arm-acquiring-bandit
date: 2026-04-20
updated: 2026-04-20
aliases: [dynamic-arm bandit]
tags: [bandits, search]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Arm-Acquiring Bandit** — a bandit setting in which new candidate arms appear over time instead of being fixed in advance.

## Key Points

- [[tang-2024-code-2405-17503]] frames program refinement as an arm-acquiring bandit because each LLM repair creates a brand-new program that can later be refined again.
- In this view, the current set of candidate programs is the active arm set, and refining any one of them can expand the search space further.
- This formulation matches the infinite-width, infinite-depth refinement tree better than standard fixed-arm bandit models.
- The paper uses this framing to justify Beta-Bernoulli Thompson Sampling over partial programs.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-code-2405-17503]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-code-2405-17503]].
