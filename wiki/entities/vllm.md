---
type: entity
title: vLLM
slug: vllm
date: 2026-04-20
entity_type: tool
aliases: [vLLM]
tags: []
---

## Description

vLLM is the inference backend used during DeepSeek-V2 reinforcement-learning training. The paper leverages it to accelerate large-batch rollout generation inside the hybrid RL engine.

## Key Contributions

- Serves as the inference component in the paper's hybrid RL training system.
- Helps keep RL rollout generation fast enough for online preference optimization at large model scale.
- Connects the paper's alignment section to practical systems engineering rather than optimizer design alone.

## Related Concepts

- [[group-relative-policy-optimization]]
- [[key-value-cache]]
- [[large-language-model]]

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]
