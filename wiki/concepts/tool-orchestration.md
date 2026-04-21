---
type: concept
title: Tool Orchestration
slug: tool-orchestration
date: 2026-04-20
updated: 2026-04-20
aliases: [工具编排]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Tool Orchestration** (工具编排) — the capability to choose, order, and coordinate single, sequential, parallel, or mixed tool calls so an agent completes a user task correctly and efficiently.

## Key Points

- WildToolBench argues that realistic user requests often require tool topologies that are trees rather than simple chains.
- The benchmark separates orchestration into sequential, parallel, and mixed multi-tool settings and evaluates each separately.
- Evaluation does not only ask whether a model finished the task; it also measures acceptable-path and optimal-path rates.
- Even the best model in the paper reaches only `43.75%` orchestration task accuracy and `42.74%` optimal-path rate.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2026-benchmarking-2604-06185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2026-benchmarking-2604-06185]].
