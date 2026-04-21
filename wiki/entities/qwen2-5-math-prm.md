---
type: entity
title: Qwen2.5-Math-PRM
slug: qwen2-5-math-prm
date: 2026-04-20
entity_type: tool
aliases: [Qwen2.5-Math-PRM-7B, Qwen2.5-Math-PRM-72B]
tags: [model, math, reasoning]
---

## Description

Qwen2.5-Math-PRM is the family of Process Reward Models (7B and 72B parameter variants) released by the Qwen Team in [[zhang-2025-lessons-2501-07301]], trained using consensus filtering of MC estimation and LLM-as-a-judge annotations initialized from Qwen2.5-Math-Instruct.

## Key Contributions

- Sets a new open-source state-of-the-art on both Best-of-8 (`67.6` avg on 7 benchmarks for 7B) and PROCESSBENCH (`73.5` avg F1 for 7B; `78.3` for 72B).
- Demonstrates that consensus-filtered hard-label training with both BoN and step-level evaluation closes the gap between automated and human-annotated PRM quality.
- Released publicly at `https://hf.co/Qwen/Qwen2.5-Math-PRM-7B` and `https://hf.co/Qwen/Qwen2.5-Math-PRM-72B`.

## Related Concepts

- [[process-reward-model]]
- [[consensus-filtering]]
- [[mathematical-reasoning]]

## Sources

- [[zhang-2025-lessons-2501-07301]]
