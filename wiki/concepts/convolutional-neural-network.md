---
type: concept
title: Convolutional Neural Network
slug: convolutional-neural-network
date: 2026-04-20
updated: 2026-04-20
aliases: [卷积神经网络, CNN]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Convolutional Neural Network** (卷积神经网络) — a neural architecture that applies learned convolutional filters over local regions to derive hierarchical feature representations.

## Key Points

- The paper uses a two-layer CNN to encode patent title, abstract, and claims.
- The first convolutional layer operates at the word level, while the second models sentence-level or claim-order structure.
- Pooling is used after each convolution stage to compress long patent text into a fixed-size representation.
- In this work, CNN is one half of the hybrid CTF model and is responsible for content-based patent features.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2018-patent]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2018-patent]].
