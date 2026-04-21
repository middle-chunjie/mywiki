---
type: entity
title: MambaOut
slug: mambaout
date: 2026-04-20
entity_type: model
aliases: [MambaOut model]
tags: []
---

## Description

MambaOut is the family of hierarchical vision backbones proposed in [[yu-2024-mambaout-2405-07992]]. It keeps the surrounding Mamba-style block structure but removes SSM, using Gated CNN token mixing instead.

## Key Contributions

- Provides a controlled architecture-level ablation for testing whether SSM is necessary in visual backbones.
- Outperforms the compared visual Mamba models on ImageNet classification across the reported scales.
- Serves as a negative-result baseline showing that dense visual tasks remain the more plausible setting for SSM benefits.

## Related Concepts

- [[gated-cnn]]
- [[state-space-model]]
- [[image-classification]]

## Sources

- [[yu-2024-mambaout-2405-07992]]
