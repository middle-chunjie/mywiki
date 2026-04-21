---
type: concept
title: Program State Tracking
slug: program-state-tracking
date: 2026-04-20
updated: 2026-04-20
aliases: [state tracing, program state tracking]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Program State Tracking** (程序状态跟踪) — explicit maintenance of variable bindings and intermediate execution states throughout a reasoning trace.

## Key Points

- CoC records delta states after each line so both Python and the LMulator operate over a shared execution context.
- The state trace is central to handling loops, conditionals, and semantic helper functions in one unified trajectory.
- Ablations in the paper show that LM variants with explicit state traces outperform LM variants that only emit final answers.
- The implementation stores state in simple serializable forms, which makes the approach workable but limits support for complex custom objects.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-chain-2312-04474]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-chain-2312-04474]].
