---
type: concept
title: Bidirectional Pruning
slug: bidirectional-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [two-sided pruning, 双向剪枝]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Bidirectional Pruning** (双向剪枝) — pruning search branches both before execution with predictive signals and after execution with grounded feedback to improve accuracy per unit compute.

## Key Points

- ToolTree combines pre-pruning with threshold `tau_pre` and post-pruning with threshold `tau_post` inside one planning loop.
- Pre-pruning reduces wasted tool calls by discarding branches that look schema-incompatible or low-yield before execution.
- Post-pruning prevents further expansion of branches that fail after actual tool interaction, which improves credit assignment over purely hypothetical search.
- The ablation results show that removing either pruning mechanism increases token cost, and removing both hurts both cost and accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2026-tooltree-2603-12740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2026-tooltree-2603-12740]].
