---
type: concept
title: Image Restoration
slug: image-restoration
date: 2026-04-20
updated: 2026-04-20
aliases: [图像复原, image restoration]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Image Restoration** (图像复原) — the task of reconstructing a high-quality image from a degraded observation while preserving scene-faithful structure and texture.

## Key Points

- ReFIR treats restoration as an ill-posed problem where diffusion-based LRMs can generate realistic but scene-inaccurate details under severe degradation.
- The paper argues that faithful restoration requires supplementing a model's internal prior with external visual knowledge from retrieved high-quality images.
- Its mechanistic analysis separates structure reconstruction in ControlNet from texture restoration in the UNet decoder, guiding where retrieval information should be injected.
- The method targets both fidelity and perceptual quality, reporting gains on PSNR, SSIM, LPIPS, NIQE, FID, MUSIQ, and CLIPIQA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guo-2024-refir-2410-05601]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guo-2024-refir-2410-05601]].
