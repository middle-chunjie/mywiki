---
type: concept
title: Formal Specification
slug: formal-specification
date: 2026-04-20
updated: 2026-04-20
aliases: [formal specification, 形式化规格]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Formal Specification** (形式化规格) — an explicit, machine-checkable description of desired program behavior, typically written as constraints such as pre-conditions and post-conditions rather than as executable implementations.

## Key Points

- [[unknown-nd-enhancing-2309-17272]] uses specifications as one of three reasoning perspectives alongside solutions and test cases.
- The paper instantiates specifications as Python functions `preconditions(input)` and `postconditions(input, output)` rather than external proof languages such as Coq or Dafny.
- Specification quality is imperfect: on HumanEval the generated specification accuracy is reported as `45.93`, below the solution accuracy of `68.38`.
- Despite that noise, specifications still improve MPSC because cross-perspective agreement adds useful constraints during reranking.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-enhancing-2309-17272]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-enhancing-2309-17272]].
