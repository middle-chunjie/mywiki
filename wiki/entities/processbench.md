---
type: entity
title: ProcessBench
slug: processbench
date: 2026-04-20
entity_type: benchmark
aliases: [PROCESSBENCH, Process Bench]
tags: [benchmark, math, reasoning, evaluation]
---

## Description

ProcessBench is a step-level evaluation benchmark introduced in Zheng et al. (arXiv:2412.06559) for measuring the ability of models to identify the first erroneous step in mathematical reasoning traces, as opposed to only checking final answer correctness.

## Key Contributions

- Defines step-level error identification as the primary PRM evaluation task: given a multi-step solution, locate the first wrong step or confirm all steps are correct.
- Covers four difficulty levels: GSM8K, MATH, OlympiadBench, and Omni-MATH.
- Used in [[zhang-2025-lessons-2501-07301]] to expose the inverse correlation between BoN performance and genuine process-verification capability.
- Demonstrates that human-annotated PRMs and LLM-as-a-judge methods substantially outperform MC-trained PRMs on step-level accuracy.

## Related Concepts

- [[step-level-evaluation]]
- [[process-reward-model]]
- [[bon-prm-misalignment]]

## Sources

- [[zhang-2025-lessons-2501-07301]]
