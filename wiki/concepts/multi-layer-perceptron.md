---
type: concept
title: Multi-Layer Perceptron
slug: multi-layer-perceptron
date: 2026-04-20
updated: 2026-04-20
aliases: [MLP, 多层感知机]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Multi-Layer Perceptron** (多层感知机) — a feedforward neural network that alternates learned linear transformations with fixed nonlinear activations applied at nodes.

## Key Points

- The paper uses MLPs as the main baseline family and contrasts them with KANs in terms of both parameterization and interpretability.
- In the paper's notation, an MLP has the form `W_{L-1} \circ \sigma \circ ... \circ W_0`, separating linear maps from fixed activations, unlike KANs which merge both into edge functions.
- For width `N` and depth `L`, the paper highlights the standard `O(N^2 L)` MLP parameter scaling against KAN's `O(N^2 L G)` spline-augmented scaling.
- Across toy fitting, PDE solving, and several science benchmarks, the paper reports that MLPs usually need many more parameters than KANs to reach similar accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-kan-2404-19756]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-kan-2404-19756]].
