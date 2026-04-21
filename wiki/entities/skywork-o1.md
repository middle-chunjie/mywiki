---
type: entity
title: Skywork-o1
slug: skywork-o1
date: 2026-04-20
entity_type: tool
aliases: [Skywork o1, Skywork-PRM, skywork-o1-open]
tags: [process-reward-model, open-source, llm]
---

## Description

Skywork-o1 is an open-source project from Skywork that releases process reward models (PRMs) in 1.5B and 7B parameter sizes, producing scalar scores for each reasoning step. Used as baselines in [[processbench]] evaluation.

## Key Contributions

- Skywork-PRM-7B achieves F1 = 42.1 average on [[processbench]]—the best among auto-labeled open-source PRMs—but still far below the human-annotated Qwen2.5-Math-7B-PRM800K (56.5).
- Performance degrades sharply on harder subsets: 70.8 F1 on GSM8K vs. 21.0 on Omni-MATH, illustrating the difficulty-generalization gap in current PRM training pipelines.
- Skywork PRMs require a score threshold to convert continuous outputs to binary step-correctness predictions; threshold selected by maximising F1 on the GSM8K subset.

## Related Concepts

- [[process-reward-model]]
- [[step-level-verification]]
- [[mathematical-reasoning]]

## Sources

- [[zheng-2024-processbench-2412-06559]]
