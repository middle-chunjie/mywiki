---
type: entity
title: UPMC-Food-101
slug: upmc-food-101
date: 2026-04-20
entity_type: dataset
aliases: [UPMC Food-101, Food-101 multimodal, UPMC-Food101]
tags: []
---

## Description

UPMC-Food-101 is a multimodal image-text classification dataset containing food images paired with textual recipe descriptions for 101 food categories. Introduced by Wang et al. (2015), it is derived from UPMC (ISIR, Paris) and widely used as a benchmark for vision-language classification. The full dataset contains ~87K samples; UniS-MMC uses 60,085 train, 5,000 validation (split from training), and 21,683 test samples.

## Key Contributions

- Standard benchmark for multimodal image-text classification with 101 fine-grained food categories.
- Distinguishes methods that exploit complementary vision-language information versus those that rely primarily on text (BERT text-only achieves 86.8% vs. ViT image-only 73.1%).

## Related Concepts

- [[image-text-classification]]
- [[multi-modal-learning]]

## Sources

- [[zou-2023-unismmc-2305-09299]]
