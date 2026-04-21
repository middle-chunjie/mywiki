---
type: concept
title: Targeted Structured Pruning
slug: targeted-structured-pruning
date: 2026-04-20
updated: 2026-04-20
aliases: [target-aware structured pruning, targeted pruning, 目标结构化剪枝]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Targeted Structured Pruning** (目标结构化剪枝) — a structured pruning strategy that searches for a high-performing subnetwork while enforcing explicit architectural constraints for a predetermined target shape.

## Key Points

- Sheared-LLaMA extends CoFiPruning so the pruned model matches a specified dense architecture instead of only a sparsity ratio.
- The target shapes are borrowed from existing efficient small models, including Pythia-1.4B-like and INCITE-Base-3B-like configurations.
- Constraint terms with Lagrange multipliers directly enforce target counts for layers, heads, hidden dimensions, and FFN widths.
- The resulting models trade a small amount of immediate perplexity for better practical throughput than irregularly pruned baselines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2024-sheared-2310-06694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2024-sheared-2310-06694]].
