---
type: concept
title: Thought Revision
slug: thought-revision
date: 2026-04-20
updated: 2026-04-20
aliases: [thought revision, reasoning revision]
tags: [llm, reasoning, search]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Thought Revision** — an iterative process that identifies and repairs errors in intermediate reasoning trajectories before producing a final answer.

## Key Points

- In XoT, the LLM first checks an MCTS-generated thought trajectory and flags an error state `s_e` when the trajectory appears wrong.
- MCTS then reruns simulations from the parent of `s_e` to generate a revised trajectory, which is returned to the LLM for final solving.
- The revision loop can be repeated multiple times, and the paper reports clear gains from `1` to `3` revisions across all three benchmark tasks.
- Because the LLM is used mainly for error detection rather than full branch evaluation, revision improves reliability without exploding LLM inference cost.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-everything-2311-04254]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-everything-2311-04254]].
