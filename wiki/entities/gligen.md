---
type: entity
title: GLIGEN
slug: gligen
date: 2026-04-20
entity_type: tool
aliases: [GLIGEN]
tags: []
---

## Description

GLIGEN is the layout-to-image diffusion model used in [[cho-2023-visual-2305-15328]] as the final rendering stage of VPGen. It consumes the original prompt together with region descriptions and bounding boxes.

## Key Contributions

- Provides the image-generation backend for the proposed step-by-step pipeline.
- Makes VPGen comparable to Stable Diffusion v1.4 because GLIGEN is built on a frozen Stable Diffusion backbone with spatial adapters.
- Reveals a key bottleneck in the analysis: final images can still fail even when the predicted layout is correct.

## Related Concepts

- [[diffusion-model]]
- [[layout-control]]
- [[text-to-image-generation]]

## Sources

- [[cho-2023-visual-2305-15328]]
