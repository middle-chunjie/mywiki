---
type: concept
title: Task-Agnostic Scaffolding
slug: task-agnostic-scaffolding
date: 2026-04-20
updated: 2026-04-20
aliases: [task agnostic scaffolding]
tags: [llm, prompting, orchestration]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Task-Agnostic Scaffolding** — a reusable inference-time control structure for language models that applies the same high-level procedure across many tasks without task-specific exemplars or custom pipelines.

## Key Points

- The paper keeps the high-level meta instructions fixed while varying only the user query and, when enabled, access to the Python interpreter.
- The scaffold combines decomposition, expert assignment, context routing, answer extraction, and verification under one reusable prompting interface.
- It is evaluated on heterogeneous tasks including Game of 24, BIG-Bench Hard tasks, Python Programming Puzzles, MGSM, and sonnet writing.
- The paper argues that this task-agnostic design reduces the burden on users who would otherwise need bespoke prompting recipes for each new task.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[suzgun-2024-metaprompting-2401-12954]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[suzgun-2024-metaprompting-2401-12954]].
