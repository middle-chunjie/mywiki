---
type: entity
title: Skywork-Reward-Llama-3.1-8B
slug: skywork-reward-llama-3.1-8b
date: 2026-04-20
entity_type: tool
aliases: [Skywork Reward Llama 3.1 8B]
tags: []
---

## Description

Skywork-Reward-Llama-3.1-8B is the reward-model backbone fine-tuned with LoRA to instantiate the critic models in [[li-2024-can-2410-01428]]. The paper uses its first classifier logit as the scalar reward score for planning decisions.

## Key Contributions

- Serves as the shared base model for sub-goal and execution critics.
- Provides compact, domain-adaptable scoring models that are cheaper to tune than the main generator.

## Related Concepts

- [[critic-model]]
- [[large-language-model]]
- [[pairwise-ranking-loss]]

## Sources

- [[li-2024-can-2410-01428]]
