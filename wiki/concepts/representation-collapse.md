---
type: concept
title: Representation Collapse
slug: representation-collapse
date: 2026-04-20
updated: 2026-04-20
aliases: [Feature Collapse, 表示坍缩, 特征坍缩]
tags: [neural-architecture, training-pathology, deep-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Representation Collapse** (表示坍缩) — a failure mode in deep networks where the hidden representations at successive layers become highly similar to one another, effectively reducing the functional depth of the network as deeper layers contribute diminishing transformations.

## Key Points

- Empirically diagnosed by measuring cosine similarity between the inputs of adjacent layers; high similarity (approaching 1.0) across the depth of the network signals collapse.
- In Pre-Norm Transformer language models, this pathology emerges at scale: OLMo-1B with Pre-Norm shows median inter-layer cosine similarity near 1.0 in deeper layers, with narrow percentile bands.
- Representation collapse reduces the value of depth: adding more layers offers diminishing returns because each layer performs a near-identity transformation.
- [[hyper-connections]] directly address this by allowing the network to learn low residual weights, forcing each layer to transform representations more substantially; DHC-trained models show significantly lower and more variable inter-layer cosine similarity.
- Distinct from gradient vanishing: [[pre-norm]] suppresses vanishing gradients but induces collapse; [[post-norm]] suppresses collapse but reintroduces vanishing gradients (the seesaw effect).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2025-hyperconnections-2409-19606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2025-hyperconnections-2409-19606]] as a central motivation for hyper-connections.
