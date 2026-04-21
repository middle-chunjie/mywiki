---
type: entity
title: MiniGrid
slug: minigrid
date: 2026-04-20
entity_type: tool
aliases: [MiniGrid]
tags: []
---

## Description

MiniGrid is the language-conditioned gridworld benchmark suite used in [[tang-2024-worldcoder-2402-12275]] to study transfer learning, goal-conditioned reward adaptation, and sparse-reward exploration. It supplies the paper's main curriculum-learning evaluation.

## Key Contributions

- Provides a sequence of environments that isolate transfer over new objects, new dynamics, and new natural-language goals.
- Shows that optimism under uncertainty is necessary for solving harder non-transfer settings and for zero-shot adaptation to new goals.
- Includes the UnlockPickup setting where the paper reports learning the correct model in `<=100` actions.

## Related Concepts

- [[transfer-learning]]
- [[contextual-markov-decision-process]]
- [[goal-directed-exploration]]

## Sources

- [[tang-2024-worldcoder-2402-12275]]
