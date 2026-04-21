---
type: entity
title: Sokoban
slug: sokoban
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

Sokoban is the grid-based puzzle environment used in [[tang-2024-worldcoder-2402-12275]] to test whether a learned code world model can rapidly acquire and transfer basic pushing dynamics. It serves as the paper's dense-reward planning benchmark.

## Key Contributions

- Demonstrates that WorldCoder can build a usable world model within the first `50` actions.
- Supports the comparison showing deep RL baselines need `>1` million interactions for basic `2`-box competence.
- Provides the setting where ReAct achieves only `15% ± 8%` success despite having pretrained priors about the game.

## Related Concepts

- [[world-model]]
- [[goal-directed-exploration]]
- [[model-based-reinforcement-learning]]

## Sources

- [[tang-2024-worldcoder-2402-12275]]
