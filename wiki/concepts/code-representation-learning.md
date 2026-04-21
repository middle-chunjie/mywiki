---
type: concept
title: Code Representation Learning
slug: code-representation-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [code representation learning, 代码表示学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Representation Learning** (代码表示学习) — the process of mapping code into vector representations that preserve semantic or structural information useful for downstream tasks.

## Key Points

- The paper decomposes code models into representation learning `r` and predictive learning `p`, making symmetry guarantees explicit at both stages.
- It argues that generalizable code representations should be equivariant to semantics-preserving transformations rather than fully invariant.
- SymC realizes this with Transformer embeddings augmented by PDG-aware graph-biased attention.
- The learned representations are evaluated indirectly through downstream tasks on both binary and source code.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
