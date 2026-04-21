---
type: concept
title: Loop Invariant Synthesis
slug: loop-invariant-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [循环不变式综合, invariant generation]
tags: [formal-methods, program-analysis]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Loop Invariant Synthesis** (循环不变式综合) — the task of generating logical predicates that remain true across loop iterations and are strong enough to prove desired program properties.

## Key Points

- [[tang-2024-code-2405-17503]] uses nonlinear loop invariant synthesis as one of its three main evaluation domains for refinement search.
- The paper evaluates on `38` benchmark tasks and checks whether candidate invariants are established, preserved, and sufficient for the postcondition.
- REx treats solver feedback and failed verification conditions as refinement signals for the LLM.
- On this benchmark, the paper reports `73.7%` solved, outperforming a prior specialized solver that reaches `60.5%`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tang-2024-code-2405-17503]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tang-2024-code-2405-17503]].
