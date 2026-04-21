---
type: entity
title: MSR-VTT
slug: msr-vtt
date: 2026-04-20
entity_type: dataset
aliases: [MSR Video to Text]
tags: []
---

## Description

MSR-VTT is a large-scale text-video retrieval benchmark used in [[jiang-2022-crossmodal-2211-09623]]. The paper uses the `9k` training split and the `1k-A` test set to study parameter-performance trade-offs in detail.

## Key Contributions

- Serves as the main benchmark for the paper's detailed comparison curves and overfitting analysis.
- Shows that Cross-Modal Adapter can reach text-to-video `R@1 = 45.4` with only `0.52M` trainable parameters.

## Related Concepts

- [[text-video-retrieval]]
- [[overfitting]]
- [[parameter-efficient-fine-tuning]]

## Sources

- [[jiang-2022-crossmodal-2211-09623]]
