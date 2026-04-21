---
type: concept
title: Class Skeleton
slug: class-skeleton
date: 2026-04-20
updated: 2026-04-20
aliases: [class specification, 类骨架]
tags: [code-generation, specification, benchmark]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Class Skeleton** (类骨架) — a structured specification for a target class that defines its class-level metadata and method-level contracts without providing the implementation bodies.

## Key Points

- In [[du-2023-classeval-2308-01861]], the class skeleton is the primary input representation for each ClassEval task.
- Mandatory elements include class name, class description, method signature, and functional description.
- Optional elements include import statements, constructor details, parameter/return descriptions, and example input/output pairs.
- The format is inspired by contract programming and is intended to make evaluation interfaces explicit and testable.
- Class skeletons let the benchmark support holistic, incremental, and compositional prompting with a consistent task specification.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2023-classeval-2308-01861]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2023-classeval-2308-01861]].
