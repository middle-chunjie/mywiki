---
type: concept
title: Evolutionary Algorithm
slug: evolutionary-algorithm
date: 2026-04-20
updated: 2026-04-20
aliases: [EA, evolutionary algorithm, 进化算法]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Evolutionary Algorithm** (进化算法) — a population-based optimization method that iteratively applies variation and selection operators such as mutation, crossover, and survival filtering.

## Key Points

- EvoAgent maps agent generation onto an evolutionary process in which each agent is treated as an individual and its settings are the variables to evolve.
- The paper operationalizes the core EA steps as LLM-mediated crossover, mutation, selection, and result-update operators over agent populations.
- It uses selection not only to prefer stronger agents but also to preserve diversity relative to parent agents and earlier generations.
- The method is nonparametric in the sense that it does not require retraining the backbone LLM; it wraps an existing agent framework with search over agent configurations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yuan-2024-evoagent-2406-14228]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yuan-2024-evoagent-2406-14228]].
