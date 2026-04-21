---
type: concept
title: Method Dependency
slug: method-dependency
date: 2026-04-20
updated: 2026-04-20
aliases: [inter-method dependency, 方法依赖]
tags: [code-generation, dependency, software-engineering]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Method Dependency** (方法依赖) — a dependency pattern in which a method invokes other methods in the same class and therefore must coordinate with class-internal behavior rather than act as a standalone routine.

## Key Points

- [[du-2023-classeval-2308-01861]] treats method dependency as a key source of difficulty in class-level code generation.
- `107` methods in ClassEval, or `26.0%` of the benchmark, contain method dependencies.
- The benchmark includes class-level tests to verify interactions among multiple methods, not only isolated method behavior.
- The paper's `DEP(M)` metric explicitly measures how often generated code reproduces the canonical method dependencies.
- Across models, generating method-invoking code is harder than generating field-accessing code, and standalone methods are easiest.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2023-classeval-2308-01861]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2023-classeval-2308-01861]].
