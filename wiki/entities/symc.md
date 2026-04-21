---
type: entity
title: SymC
slug: symc
date: 2026-04-20
entity_type: tool
aliases: [SymC model, symmetry-preserving code model]
tags: []
---

## Description

SymC is the symmetry-preserving Transformer-based code model introduced in [[pei-2024-exploiting-2308-03312]]. It injects graph-structured program semantics into attention so predictions remain invariant to semantics-preserving permutations.

## Key Contributions

- Uses graph-aware biased self-attention built around PDG-derived distance matrices.
- Delivers stronger generalization than pre-trained baselines on unseen transformations while using far less training compute.
- Provides a concrete architecture matching the paper's group-equivariant and group-invariant theory.

## Related Concepts

- [[transformer]]
- [[group-equivariance]]
- [[program-dependence-graph]]

## Sources

- [[pei-2024-exploiting-2308-03312]]
