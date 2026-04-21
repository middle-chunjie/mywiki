---
type: concept
title: Runnable Level
slug: runnable-level
date: 2026-04-20
updated: 2026-04-20
aliases: [dependency level, 可运行级别]
tags: [benchmark, code-generation, evaluation]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Runnable level** (可运行级别) — a benchmark label describing how much external code context a function requires before it can execute successfully.

## Key Points

- CoderEval defines six runnable levels: `self-contained`, `slib-runnable`, `plib-runnable`, `class-runnable`, `file-runnable`, and `project-runnable`.
- Each higher runnable level may depend on lower levels but must not require dependencies from later levels.
- The taxonomy provides a structured way to measure how code generation degrades as contextual dependency increases.
- In CoderEval for Python, `84%` of tasks are `file_runnable`, showing that same-file context is often critical in real projects.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-codereval-2302-00288]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-codereval-2302-00288]].
