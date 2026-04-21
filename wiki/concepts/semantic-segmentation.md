---
type: concept
title: Semantic Segmentation
slug: semantic-segmentation
date: 2026-04-20
updated: 2026-04-20
aliases: [语义分割]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Segmentation** (语义分割) — the task of assigning a semantic category label to each pixel in an image.

## Key Points

- The paper evaluates ADE20K segmentation as another long-sequence visual task, since high-resolution inputs produce around `4K` tokens.
- MambaOut is used as the backbone for UperNet with ImageNet initialization, AdamW `lr = 1e-4`, batch size `16`, and `160000` training iterations.
- `MambaOut-Tiny` reports `47.4` single-scale mIoU and `48.6` multi-scale mIoU, slightly above `VMamba-T` but below `LocalVMamba-T`.
- The segmentation results reinforce the paper's claim that SSM may still help dense vision workloads that are longer-sequence than classification.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-mambaout-2405-07992]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-mambaout-2405-07992]].
