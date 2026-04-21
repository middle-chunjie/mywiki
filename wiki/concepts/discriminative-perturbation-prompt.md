---
type: concept
title: Discriminative perturbation prompt
slug: discriminative-perturbation-prompt
date: 2026-04-20
updated: 2026-04-20
aliases: [DPP]
tags: [prompting, vision]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Discriminative perturbation prompt** — an image-space prompt that amplifies discriminative object regions by content-aware spatial resampling before a frozen backbone processes the image.

## Key Points

- DPP is the sample-prompting component of FRPT and operates directly on pixels rather than latent tokens.
- It predicts a discriminative projection map from low-level features and normalizes the map spatially with softmax.
- The prompt image is generated through content-aware inhomogeneous sampling followed by bilinear interpolation.
- The paper argues that DPP exaggerates decision-boundary-relevant details instead of merely cropping foreground objects.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-finegrained-2207-14465]].
