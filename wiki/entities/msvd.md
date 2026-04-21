---
type: entity
title: MSVD
slug: msvd
date: 2026-04-20
entity_type: dataset
aliases: [Microsoft Video Description Dataset]
tags: []
---

## Description

MSVD is a text-video retrieval benchmark used in [[jiang-2022-crossmodal-2211-09623]], with standard train, validation, and test splits of `1200/100/670` videos. The paper reports both text-to-video and video-to-text retrieval metrics on this dataset.

## Key Contributions

- Demonstrates that Cross-Modal Adapter reaches text-to-video `R@1 = 47.4` with about `1.00M` trainable parameters.
- Helps show that the method remains competitive with fully fine-tuned CLIP-based systems on smaller retrieval datasets.

## Related Concepts

- [[text-video-retrieval]]
- [[query-aware-video-representation]]
- [[parameter-efficient-fine-tuning]]

## Sources

- [[jiang-2022-crossmodal-2211-09623]]
