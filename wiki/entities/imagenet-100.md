---
type: entity
title: ImageNet-100
slug: imagenet-100
date: 2026-04-20
entity_type: dataset
aliases: [ImageNet100]
tags: []
---

## Description

ImageNet-100 is the 100-class ImageNet subset used in [[wang-2021-understanding-2012-09740]] for pretraining and linear evaluation with a ResNet-50 backbone.

## Key Contributions

- Extends the paper's temperature analysis beyond small-scale datasets such as CIFAR-10, CIFAR-100, and SVHN.
- Shows that ordinary contrastive loss peaks at `75.10%` top-1 accuracy at `tau = 0.3`, while the simple objective is much weaker at `48.09%`.

## Related Concepts

- [[contrastive-learning]]
- [[representation-learning]]
- [[embedding-uniformity]]

## Sources

- [[wang-2021-understanding-2012-09740]]
