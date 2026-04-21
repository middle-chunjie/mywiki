---
type: concept
title: World Model
slug: world-model
date: 2026-04-20
updated: 2026-04-20
aliases: [world model]
tags: [llm, planning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**World Model** (世界模型) — an internal or external representation of how states evolve under actions, enabling planning, value estimation, and long-horizon decision making.

## Key Points

- The paper argues that MCTS plus the learned policy/value module gives the LLM access to an external world model for search-heavy tasks.
- This world model injects domain knowledge about valid transitions and action quality, especially for tasks where the LLM alone struggles with long-term planning.
- XoT uses the world model to supply thought trajectories that the LLM can review instead of inventing search traces from scratch.
- The discussion section notes that changes in the environment can make the world model inaccurate and thereby degrade the generated thoughts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-everything-2311-04254]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-everything-2311-04254]].
