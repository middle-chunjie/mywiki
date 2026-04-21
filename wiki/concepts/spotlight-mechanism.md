---
type: concept
title: Spotlight mechanism
slug: spotlight-mechanism
date: 2026-04-20
updated: 2026-04-20
aliases: [spotlight attention, spotlight handle]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Spotlight mechanism** — a localized visual focusing mechanism that parameterizes attention by a center and radius, then extracts context by weighting nearby image features more heavily than distant ones.

## Key Points

- STN represents the spotlight at step `t` with `s_t = (x_t, y_t, sigma_t)^T`, where `(x_t, y_t)` is the center and `sigma_t` controls radius.
- Weights over the feature map are induced by a truncated Gaussian-like score, making the focus location explicit and differentiable.
- The spotlight context `sc_t` is a weighted sum of spatial encoder features and is passed to the decoder together with output history and spotlight position.
- The paper argues this is more structure-aware than standard soft attention because it explicitly models a reading path over structural images.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yin-2018-transcribing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yin-2018-transcribing]].
