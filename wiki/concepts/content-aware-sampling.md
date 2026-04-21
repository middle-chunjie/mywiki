---
type: concept
title: Content-aware sampling
slug: content-aware-sampling
date: 2026-04-20
updated: 2026-04-20
aliases: [content-aware inhomogeneous sampling, 内容感知采样]
tags: [sampling, vision]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Content-aware sampling** (内容感知采样) — resampling an input according to learned spatial importance so discriminative regions receive more sampling density than less relevant regions.

## Key Points

- DPP treats the discriminative projection map as a probability mass function over sampling locations.
- Spatial mappings for the prompted image are computed globally, not from purely local neighborhood rules.
- A Gaussian distance kernel regularizes the mapping to avoid degenerate collapses where many pixels map to one location.
- The final prompted image is produced with differentiable bilinear interpolation, preserving end-to-end training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-finegrained-2207-14465]].
