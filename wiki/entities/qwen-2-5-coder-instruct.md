---
type: entity
title: Qwen-2.5-Coder-Instruct
slug: qwen-2-5-coder-instruct
date: 2026-04-20
entity_type: tool
aliases: [Qwen2.5-Coder-Instruct, Qwen 2.5 Coder Instruct]
tags: [language-model, code]
---

## Description

Qwen-2.5-Coder-Instruct is the base model family fine-tuned in [[pan-2024-training-2412-21139]] for both software engineering agents and verifier models. The paper studies `7B`, `14B`, and `32B` variants.

## Key Contributions

- Serves as the open-weight backbone for OpenHands and MoatlessTools experiments.
- Shows large gains after training on SWE-Gym trajectories, especially in the 32B setting.
- Also functions as the verifier backbone for trajectory reranking.

## Related Concepts

- [[software-engineering-agent]]
- [[rejection-sampling-fine-tuning]]
- [[outcome-supervised-reward-model]]

## Sources

- [[pan-2024-training-2412-21139]]
