---
type: concept
title: User Simulation
slug: user-simulation
date: 2026-04-20
updated: 2026-04-20
aliases: [simulated users, user simulator, 用户模拟]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**User Simulation** (用户模拟) — an offline procedure that approximates user behavior with synthetic agents or scoring models so systems can be developed and evaluated without live traffic.

## Key Points

- The paper uses GPT-4 prompts plus handcrafted personas to simulate feedback for generated questions.
- Simulation is split into relevance scoring and action sampling, rather than asking the LLM to produce clicks directly.
- Each iteration runs `5000` simulated interactions, adding controlled noise to estimated CTR values.
- The simulator is central to the paper's offline development story but is also one of the main validity limitations acknowledged by the authors.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[senel-2024-generative-2406-05255]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[senel-2024-generative-2406-05255]].
