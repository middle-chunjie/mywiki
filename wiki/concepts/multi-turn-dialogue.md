---
type: concept
title: Multi-turn Dialogue
slug: multi-turn-dialogue
date: 2026-04-20
updated: 2026-04-20
aliases: [多轮对话]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multi-turn Dialogue** (多轮对话) — an interaction setting in which user requests and assistant responses unfold across multiple turns, making current decisions depend on accumulated conversational history.

## Key Points

- WildToolBench formalizes a session as `D = {u_1, a_1, ..., u_N, a_N}` with multiple tasks scattered across turns.
- The benchmark uses four-task scenarios instead of isolated single-turn requests.
- Later tasks often depend on earlier turns, and task accuracy drops as task order advances.
- The paper frames the full dialogue history, tool calls, and tool feedback as the state of an MDP.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2026-benchmarking-2604-06185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2026-benchmarking-2604-06185]].
