---
type: entity
title: Stacked-MNIST
slug: stacked-mnist
date: 2026-04-20
entity_type: dataset
aliases: [StackedMNIST]
tags: []
---

## Description

Stacked-MNIST is the labeled synthetic image dataset used in [[friedman-2023-vendi-2210-02410]] to study GAN mode collapse. It is formed by stacking three MNIST digits across color channels, yielding `1000` class combinations.

## Key Contributions

- Serves as the paper's controlled benchmark for comparing Vendi Score with number-of-modes and mode-distribution entropy.
- Shows that models with the same mode count can still differ materially in [[mode-collapse]] severity.

## Related Concepts

- [[mode-collapse]]
- [[vendi-score]]
- [[diversity-metric]]

## Sources

- [[friedman-2023-vendi-2210-02410]]
