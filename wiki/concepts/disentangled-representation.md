---
type: concept
title: Disentangled Representation
slug: disentangled-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [解耦表示, disentangled embedding]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Disentangled Representation** (解耦表示) — a representation in which different latent factors are encoded in separable components rather than being conflated in a single entangled embedding.

## Key Points

- DCCF models user and item preferences through multiple latent intents instead of one coarse-grained embedding.
- The paper introduces `K` global intent prototypes for users and items and computes soft assignments from node embeddings to those prototypes.
- Disentangled representations are injected directly into message passing through the global refinement term `R`, not only used as an auxiliary head.
- The authors argue that disentanglement is necessary to make self-supervised signals more informative under preference diversity.
- Ablation `-Disen` shows a clear performance drop, supporting the utility of disentangled representations in recommendation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2023-disentangled]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2023-disentangled]].
