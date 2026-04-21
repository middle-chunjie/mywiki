---
type: entity
title: RLHFlow PRM
slug: rlhflow-prm
date: 2026-04-20
entity_type: tool
aliases: [RLHFlow-PRM, RLHF-Reward-Modeling, rlhflow-reward-modeling]
tags: [process-reward-model, open-source, llm]
---

## Description

RLHFlow PRM is a family of open-source process reward models released by the RLHFlow project (Xiong et al., 2024), including Mistral-8B-based and DeepSeek-8B-based variants trained following the Math-Shepherd methodology with different solution generators and optimization objectives.

## Key Contributions

- RLHFlow-PRM-Mistral-8B achieves F1 = 28.4 and RLHFlow-PRM-Deepseek-8B achieves F1 = 26.6 on [[processbench]] (average), revealing poor generalization beyond GSM8K/MATH.
- Illustrates the brittleness of auto-labeled PRM training: on OlympiadBench and Omni-MATH both models drop below 17 F1, compared to 56.5 for the human-annotated Qwen2.5-Math-7B-PRM800K baseline.

## Related Concepts

- [[process-reward-model]]
- [[step-level-verification]]
- [[mathematical-reasoning]]

## Sources

- [[zheng-2024-processbench-2412-06559]]
