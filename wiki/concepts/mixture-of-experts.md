---
type: concept
title: Mixture-of-Experts
slug: mixture-of-experts
date: 2026-04-20
updated: 2026-04-20
aliases: [MoE, mixture of experts, 专家混合]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Mixture-of-Experts** (专家混合) — a sparse neural architecture in which each token is routed to only a small subset of specialized feed-forward experts instead of activating the full parameter set.

## Key Points

- DeepSeek-V2 uses a sparse MoE backbone with `236B` total parameters but only `21B` activated parameters per token.
- All FFN layers except the first are replaced by DeepSeekMoE layers containing `2` shared experts and `160` routed experts.
- Each token activates `6` routed experts, allowing the model to scale total capacity without paying dense-compute cost at every layer.
- The paper argues that fine-grained expert segmentation plus shared expert isolation improves expert specialization relative to conventional MoE designs.
- Device-limited routing, auxiliary balance losses, and token dropping are added to keep MoE training computationally stable and communication-efficient.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deepseek-ai-2024-deepseekv-2405-04434]].
