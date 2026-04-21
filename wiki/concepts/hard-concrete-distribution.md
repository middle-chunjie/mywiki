---
type: concept
title: Hard Concrete Distribution
slug: hard-concrete-distribution
date: 2026-04-20
updated: 2026-04-20
aliases: [hard concrete, Hard Concrete]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hard Concrete Distribution** — a sparse gating distribution used to approximate discrete on/off decisions while remaining trainable with gradient-based optimization.

## Key Points

- Sheared-LLaMA parameterizes pruning masks with hard-concrete distributions so mask values concentrate near `0` or `1`.
- This lets the method optimize discrete keep/prune choices for architectural blocks within a continuous training procedure.
- The distribution is paired with explicit architectural constraints rather than used only to induce overall sparsity.
- In this paper, hard-concrete masks support end-to-end search over dense target subnetworks during the pruning stage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2024-sheared-2310-06694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2024-sheared-2310-06694]].
