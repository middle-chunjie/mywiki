---
type: entity
title: DeepScaleR-1.5B-Preview
slug: deepscaler-1.5b-preview
date: 2026-04-20
entity_type: tool
aliases: [DeepScaleR-24K, DeepScaleR-4K]
tags: []
---

## Description

DeepScaleR-1.5B-Preview is the reasoning model family used as the base model and main unconstrained baseline in the paper. The authors initialize [[l1]] from this model and compare against both its original `24K` and retrained `4K` variants.

## Key Contributions

- Supplies the initialization checkpoint for LCPO training.
- Provides the unconstrained baseline that L1-Exact nearly matches and L1-Max approximately reaches under budget control.

## Related Concepts

- [[reasoning-language-model]]
- [[chain-of-thought]]
- [[test-time-scaling]]

## Sources

- [[aggarwal-2025-l-2503-04697]]
