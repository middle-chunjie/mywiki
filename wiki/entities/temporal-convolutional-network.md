---
type: entity
title: Temporal Convolutional Network
slug: temporal-convolutional-network
date: 2026-04-20
entity_type: tool
aliases: [TCN]
tags: []
---

## Description

Temporal Convolutional Network is the backbone encoder used by TimesURL in [[liu-2023-timesurl]]. In the paper it supplies the sequence encoder on which contrastive and reconstruction objectives are jointly optimized.

## Key Contributions

- Provides the base architecture that maps `x_i ∈ R^{T×F}` into timestamp-level latent representations.
- Makes TimesURL comparable to prior time-series SSL work such as TS2Vec under a similar encoder family.

## Related Concepts

- [[representation-learning]]
- [[self-supervised-learning]]
- [[time-reconstruction]]

## Sources

- [[liu-2023-timesurl]]
