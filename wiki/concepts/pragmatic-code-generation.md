---
type: concept
title: Pragmatic Code Generation
slug: pragmatic-code-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [practical code generation, 实用代码生成]
tags: [code-generation, benchmark, software-engineering]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pragmatic code generation** (实用代码生成) — code generation for realistic repository settings where a target function depends on surrounding project context rather than only built-in language features.

## Key Points

- CoderEval defines pragmatic code generation as a stricter target than standalone function synthesis in HumanEval-like benchmarks.
- The paper motivates the concept with an empirical finding that more than `70%` of functions in popular open-source Java and Python projects are non-standalone.
- In this setting, useful models must resolve dependencies on project-local types, APIs, variables, and constants.
- The benchmark operationalizes pragmatic code generation through runnable levels ranging from `self-contained` to `project-runnable`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2024-codereval-2302-00288]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2024-codereval-2302-00288]].
