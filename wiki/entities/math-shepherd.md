---
type: entity
title: Math-Shepherd
slug: math-shepherd
date: 2026-04-20
entity_type: tool
aliases: [MATH-SHEPHERD, Math Shepherd]
tags: []
---

## Description

Math-Shepherd is the process reward model introduced in [[wang-2024-mathshepherd-2312-08935]] for mathematical reasoning. It scores each intermediate reasoning step and is used both for verification and for step-level reinforcement learning.

## Key Contributions

- Introduces an automatically trained [[process-reward-model]] that scores reasoning steps instead of only final answers.
- Improves best-of-`256` verification on GSM8K and MATH over self-consistency and [[outcome-reward-model]] baselines.
- Supplies step-level rewards for [[step-by-step-ppo]] training.

## Related Concepts

- [[process-reward-model]]
- [[automatic-process-annotation]]
- [[step-by-step-ppo]]

## Sources

- [[wang-2024-mathshepherd-2312-08935]]
