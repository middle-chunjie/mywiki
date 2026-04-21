---
type: entity
title: N24News
slug: n24news
date: 2026-04-20
entity_type: dataset
aliases: [N24News dataset, N24 News]
tags: []
---

## Description

N24News is a multimodal news classification dataset introduced by Wang et al. (LREC 2022) containing 24 news categories with paired images and four text types per article: Heading, Caption, Abstract, and Body. UniS-MMC uses the first three text sources (Heading, Caption, Abstract) for experiments, with 48,988 train, 6,123 validation, and 6,124 test samples.

## Key Contributions

- Provides a multimodal news benchmark with multiple text modalities per sample, enabling fine-grained study of how text source type affects multimodal classification performance.
- Exposes the dependence of multimodal models on encoder choice (BERT vs. RoBERTa) due to its diverse text types including very short (Heading/Caption) and longer (Abstract) inputs.

## Related Concepts

- [[image-text-classification]]
- [[multi-modal-learning]]

## Sources

- [[zou-2023-unismmc-2305-09299]]
