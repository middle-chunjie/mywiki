---
type: concept
title: Critic Agent
slug: critic-agent
date: 2026-04-20
updated: 2026-04-20
aliases: [critic agent, 批评者智能体]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Critic Agent** (批评者智能体) — an auxiliary agent that evaluates intermediate observations, updates memory, and decides whether enough evidence has been gathered to answer a query.

## Key Points

- In WebWalker, the critic runs after each explorer action and consumes the query together with the latest observation-action pair `(O_t, A_t)`.
- The critic incrementally updates memory `M`, filtering observations for usefulness before deciding whether the current evidence is sufficient.
- This separation is motivated by the large history `H_t` and the difficulty of keeping long reasoning traces coherent inside a single implicit policy.
- The paper treats the critic as the module that converts raw traversal traces into answer-ready evidence for final generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-webwalker-2501-07572]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-webwalker-2501-07572]].
