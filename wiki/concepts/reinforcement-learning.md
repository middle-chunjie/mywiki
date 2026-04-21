---
type: concept
title: Reinforcement Learning
slug: reinforcement-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [RL, policy optimization]
tags: [llm, optimization]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reinforcement Learning** (强化学习) — an optimization framework that improves a policy by rewarding desirable outcomes rather than directly imitating labeled outputs.

## Key Points

- LCPO uses RL because the target behavior depends jointly on answer correctness and generated length.
- The training data contains only questions and final answers, so RL avoids requiring curated reasoning traces.
- The reward functions explicitly encode token-budget adherence, which lets the model learn adaptive reasoning lengths.
- In the paper's comparison, supervised fine-tuning on relabeled outputs fails to learn effective length control.
- Additional RL training improves adherence metrics, indicating the capability continues to sharpen after initial convergence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
