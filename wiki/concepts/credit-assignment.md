---
type: concept
title: Credit Assignment
slug: credit-assignment
date: 2026-04-20
updated: 2026-04-20
aliases: [reward attribution, 信用分配]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Credit Assignment** (信用分配) — the problem of determining which earlier decisions in a multi-step process should receive positive or negative learning signal from a delayed final outcome.

## Key Points

- AgentFlow's planner makes interdependent decisions whose value may only become clear after later tool results and verification steps.
- The paper treats long-horizon credit assignment as the core obstacle to training agentic systems from final-answer supervision.
- Flow-GRPO avoids handcrafted intermediate heuristics by assigning the same trajectory-level reward to every turn.
- The authors argue that this converts multi-turn optimization into a set of easier single-turn policy updates conditioned on full memory state.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-intheflow-2510-05592]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-intheflow-2510-05592]].
