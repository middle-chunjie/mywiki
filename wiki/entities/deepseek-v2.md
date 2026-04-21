---
type: entity
title: DeepSeek-V2
slug: deepseek-v2
date: 2026-04-20
entity_type: tool
aliases: [DeepSeek V2]
tags: []
---

## Description

DeepSeek-V2 is the base model introduced in the paper: a `236B`-parameter Mixture-of-Experts language model with `21B` activated parameters per token and support for `128K` context. It serves as the central artifact through which the paper evaluates MLA, DeepSeekMoE, and the associated training stack.

## Key Contributions

- Demonstrates that a sparse open-source model can remain top-tier while materially reducing training and serving costs.
- Combines MLA, DeepSeekMoE, YaRN-based long-context extension, and large-scale bilingual pretraining in one model.
- Provides the base checkpoint from which the SFT and RL chat models are derived.

## Related Concepts

- [[mixture-of-experts]]
- [[multi-head-latent-attention]]
- [[long-context-training]]

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]
