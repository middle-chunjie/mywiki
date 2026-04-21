---
type: concept
title: Class-level Code Generation
slug: class-level-code-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [class generation, 类级代码生成]
tags: [code-generation, benchmark, software-engineering]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Class-level Code Generation** (类级代码生成) — the task of generating an entire class, including multiple potentially interdependent methods and fields, from natural-language and structural specifications.

## Key Points

- [[du-2023-classeval-2308-01861]] frames class-level generation as a harder setting than standalone function generation because methods may share fields and invoke one another.
- ClassEval operationalizes the task with class skeletons that specify class-level information plus method-level contracts.
- The benchmark contains `100` Python class-generation tasks with `412` methods, making generated artifacts substantially longer than HumanEval or MBPP tasks.
- The paper shows that strong function-level performance does not reliably transfer to class-level performance.
- Holistic generation works best only for GPT-4 and GPT-3.5, while most other models do better with method-by-method generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2023-classeval-2308-01861]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2023-classeval-2308-01861]].
