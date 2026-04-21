---
type: concept
title: Collaborative Simulation
slug: collaborative-simulation
date: 2026-04-20
updated: 2026-04-20
aliases: [协作仿真]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Collaborative Simulation** (协作仿真) — a training and evaluation procedure that rolls a conversation forward with simulated user responses in order to estimate how current model actions affect future collaboration quality.

## Key Points

- CollabLLM uses collaborative simulation as the mechanism for approximating future conversation trajectories needed by MR.
- The simulator conditions on conversation history and an implicit goal, so the same current response can be evaluated under different plausible futures.
- The paper uses the same simulation idea both for evaluation environments and for computing training signals.
- This design avoids fitting a separate reward model, but shifts difficulty to realistic user simulation and rollout cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-collabllm-2502-00640]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-collabllm-2502-00640]].
