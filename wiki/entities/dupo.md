---
type: entity
title: Duplicating Sampling Policy Optimization
slug: dupo
date: 2026-04-20
entity_type: tool
aliases: [DUPO]
tags: []
---

## Description

Duplicating Sampling Policy Optimization is the reinforcement-learning algorithm proposed in [[li-2025-websailor-2507-02592]] to improve the efficiency of agent RL for long-horizon web tasks.

## Key Contributions

- Replaces DAPO-style cross-case dynamic sampling with within-batch duplication of informative cases.
- Uses group-relative advantage estimation and a clipped token-level policy objective.
- Delivers an estimated `2-3x` training speedup according to the paper.

## Related Concepts

- [[reinforcement-learning]]
- [[sample-efficiency]]
- [[long-horizon-reasoning]]

## Sources

- [[li-2025-websailor-2507-02592]]
