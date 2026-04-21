---
type: concept
title: Task Decomposition
slug: task-decomposition
date: 2026-04-20
updated: 2026-04-20
aliases: [problem decomposition, 任务分解]
tags: [reasoning, planning, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Task Decomposition** (任务分解) — the process of splitting a complex problem into smaller subtasks that can be solved, checked, or delegated separately before being recombined.

## Key Points

- Meta-prompting begins by instructing the Meta Model to break complex user queries into smaller manageable pieces.
- The decomposed subtasks are routed to specialized experts under fresh instructions rather than solved in one monolithic generation.
- The paper treats decomposition as a generic mechanism that works across symbolic reasoning, coding, and constrained generation tasks.
- Decomposition is coupled with centralized verification and synthesis, so the Meta Model can refine or reject partial outputs before returning a final answer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[suzgun-2024-metaprompting-2401-12954]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[suzgun-2024-metaprompting-2401-12954]].
