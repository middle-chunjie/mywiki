---
type: concept
title: Manifold Projection
slug: manifold-projection
date: 2026-04-20
updated: 2026-04-20
aliases: [projection onto a manifold, 流形投影]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Manifold Projection** (流形投影) — constraining a learnable parameter to lie on a structured geometric set so that optimization respects desired invariants such as stability or conservation.

## Key Points

- mHC frames its main contribution as projecting the HC residual mapping onto a specific manifold rather than leaving it unconstrained.
- The chosen manifold is the set of doubly stochastic matrices, which restores stable signal propagation while still allowing cross-stream interaction.
- The input and output mappings are also constrained to be non-negative, which the paper treats as a simpler form of manifold restriction that avoids sign cancellation.
- This viewpoint separates topology design from the exact computational block and makes the framework compatible with future alternative constraints beyond double stochasticity.
- The conclusion explicitly presents manifold choice as an open research direction for balancing plasticity and stability in future architectures.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2026-mhc-2512-24880]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2026-mhc-2512-24880]].
