---
type: concept
title: Planner-Executor Architecture
slug: planner-executor-architecture
date: 2026-04-20
updated: 2026-04-20
aliases: [planner-executor framework, 规划器-执行器架构]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Planner-Executor Architecture** (规划器-执行器架构) — an agent design that separates high-level decision-making about what to do next from low-level generation and execution of the concrete commands that implement those decisions.

## Key Points

- In OctoTools, the planner produces both a high-level initial plan and step-level actions `a_t` that specify sub-goals, tool choice, and contextual focus.
- A distinct executor converts each action into executable code `o_t`, runs it, and stores the resulting output `r_t` in the trajectory state `s_t := (a_t, o_t, r_t)`.
- The paper argues that this separation reduces command-generation errors by preventing one model component from simultaneously handling strategy and environment-specific execution details.
- Context verification and final solution summarization sit on top of the planner-executor loop, making the full reasoning trajectory explicit and inspectable.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lu-2025-octotools-2502-11271]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lu-2025-octotools-2502-11271]].
