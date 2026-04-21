---
type: concept
title: Image Inpainting
slug: image-inpainting
date: 2026-04-20
updated: 2026-04-20
aliases: [masked image completion, 图像修复]
tags: [computer-vision, generative-models, self-supervised-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Image Inpainting** (图像修复) — a computer vision task in which a model fills in missing or masked regions of an image to produce a complete, visually coherent result.

## Key Points

- In Bar et al. (2022), image inpainting is used as a pre-training objective on grids of academic figures and infographics, producing a model capable of visual in-context learning.
- The model is trained to fill missing patches in grid-like images; this surprisingly enables zero-shot generalization to structured vision tasks like segmentation and colorization.
- A visual prompt is constructed as a grid where in-context examples occupy cells and the query occupies a designated cell with the output region masked.
- The inpainting pre-training objective does not explicitly expose the model to segmentation or detection; generalization emerges from the grid-completion structure.
- Performance at inpainting on held-out tasks (colorization) is weaker than on segmentation, suggesting the base model has uneven task coverage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-nd-what]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-nd-what]].
