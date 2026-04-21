---
type: entity
title: ClassEval
slug: class-eval
date: 2026-04-20
entity_type: tool
aliases: [ClassEval benchmark]
tags: []
---

## Description

ClassEval is a manually crafted benchmark for class-level Python code generation introduced in [[du-2023-classeval-2308-01861]]. It contains `100` class-generation tasks with class skeletons, test suites, and canonical solutions.

## Key Contributions

- Establishes a benchmark focused on interdependent class-level code generation rather than standalone function synthesis.
- Provides strong test sufficiency with `99.7%` statement coverage and `98.2%` branch coverage on canonical solutions.
- Enables comparison of holistic, incremental, and compositional generation strategies across `11` LLMs.

## Related Concepts

- [[class-level-code-generation]]
- [[benchmark]]
- [[execution-based-evaluation]]

## Sources

- [[du-2023-classeval-2308-01861]]
