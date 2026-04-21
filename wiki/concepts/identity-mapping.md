---
type: concept
title: Identity Mapping
slug: identity-mapping
date: 2026-04-20
updated: 2026-04-20
aliases: [identity map, 恒等映射]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Identity Mapping** (恒等映射) — a propagation property in which a shallower representation is passed to deeper layers without distortion, preserving stable forward activations and backward gradients.

## Key Points

- The paper treats identity preservation as the core reason standard residual connections remain stable at scale.
- In Hyper-Connections, the composite residual mapping `prod H_l^res` is unconstrained, so it no longer behaves like an identity path and can amplify or attenuate signals.
- mHC restores a generalized identity property by constraining `H_l^res` to the doubly stochastic manifold, which preserves feature means and regularizes signal norms.
- Because doubly stochastic matrices are closed under multiplication, the identity-preserving behavior is meant to hold across arbitrary depth rather than only at one layer.
- The observed HC loss spike and gradient instability in the 27B run are presented as empirical evidence for what happens when identity mapping is broken.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2026-mhc-2512-24880]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2026-mhc-2512-24880]].
