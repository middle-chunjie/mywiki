---
type: concept
title: Field Dependency
slug: field-dependency
date: 2026-04-20
updated: 2026-04-20
aliases: [attribute dependency, 字段依赖]
tags: [code-generation, dependency, software-engineering]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Field Dependency** (字段依赖) — a code dependency pattern in which a method reads or writes class instance variables and therefore relies on class state beyond its explicit parameters.

## Key Points

- [[du-2023-classeval-2308-01861]] defines field dependency as one of the core dependency types that distinguish ClassEval from function-level benchmarks.
- `269` methods in ClassEval, or `65.5%` of the benchmark, contain field dependencies.
- Method-level tests in ClassEval can check not only return values but also mutations to class fields.
- The paper finds that models generate field-accessing code more reliably than method-invoking code.
- Errors in handling field dependencies contribute to failures such as `AttributeError`, `TypeError`, and some `KeyError` cases.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2023-classeval-2308-01861]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2023-classeval-2308-01861]].
