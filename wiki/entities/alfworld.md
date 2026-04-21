---
type: entity
title: AlfWorld
slug: alfworld
date: 2026-04-20
entity_type: tool
aliases: [ALFWorld]
tags: []
---

## Description

AlfWorld is the household robot task-planning environment used in [[tang-2024-worldcoder-2402-12275]] to stress-test the scalability of code world models. The paper converts it into a symbolic MDP over fluents and plans in it with MCTS plus a BM25-based heuristic.

## Key Contributions

- Demonstrates that WorldCoder can synthesize `250+` lines of environment code for a richer long-horizon domain.
- Motivates the use of a modified `[[monte-carlo-tree-search]]` planner for deterministic sparse-reward environments.
- Shows that the agent often reaches reward in the first episode after about `20` exploratory steps.

## Related Concepts

- [[monte-carlo-tree-search]]
- [[model-based-reinforcement-learning]]
- [[transition-function]]

## Sources

- [[tang-2024-worldcoder-2402-12275]]
