---
type: concept
title: Step-by-Step PPO
slug: step-by-step-ppo
date: 2026-04-20
updated: 2026-04-20
aliases: [step by step PPO, step-level PPO, 逐步PPO]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Step-by-Step PPO** (逐步PPO) — a reinforcement-learning setup that assigns rewards at intermediate reasoning steps instead of only at the end of a full response.

## Key Points

- [[wang-2024-mathshepherd-2312-08935]] uses MATH-SHEPHERD as a step-level reward model during PPO training for math LLMs.
- Unlike ORM-based PPO, the method gives a reward signal after each reasoning step, improving credit assignment for multi-step solutions.
- The paper trains LLaMA2-7B and Mistral-7B with this setup using learning rates `` `4e-7` `` and `` `1e-7` `` respectively, with KL coefficient `` `0.04` ``.
- Step-by-step PPO outperforms both rejective fine-tuning and ORM-PPO on GSM8K and MATH.
- The best reported combination is Mistral-7B plus step-by-step PPO plus verification, reaching `89.1%` on GSM8K and `43.5%` on MATH.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-mathshepherd-2312-08935]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-mathshepherd-2312-08935]].
