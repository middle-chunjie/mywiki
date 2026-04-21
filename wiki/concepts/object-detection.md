---
type: concept
title: Object Detection
slug: object-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [目标检测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Object Detection** (目标检测) — the task of locating and classifying object instances within an image, typically with bounding boxes and class labels.

## Key Points

- The paper treats COCO detection as a long-sequence visual task because typical inference resolution yields about `4K` tokens under ViT-style patching.
- MambaOut is evaluated as the backbone of Mask R-CNN with ImageNet pretraining, AdamW, `lr = 1e-4`, batch size `16`, and the standard `1x` schedule.
- `MambaOut-Tiny` reaches `45.1 AP^b`, which is better than some visual Mamba baselines but worse than `VMamba-T` at `46.5 AP^b`.
- These results support the paper's narrower claim that SSM may still be useful in dense visual tasks even if it is unnecessary for ImageNet classification.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-mambaout-2405-07992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-mambaout-2405-07992]].
