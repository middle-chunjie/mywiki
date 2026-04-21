---
type: concept
title: Programming Concept Predicate
slug: programming-concept-predicate
date: 2026-04-20
updated: 2026-04-20
aliases: [PCP, 程序概念谓词]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Programming Concept Predicate** (程序概念谓词) — a predicate over program elements or their relations that captures whether a specific programming concept, such as control flow or variable scope, holds for a program.

## Key Points

- [[hooda-2024-do-2402-05980]] uses PCPs as the abstraction layer for evaluating concept understanding in code LLMs.
- The paper instantiates PCPs for [[control-flow]], [[data-flow]], [[data-types]], and [[identifier-naming]].
- PCPs are evaluated through semantics-preserving mutations rather than through task-level accuracy alone.
- The authors argue that PCP-level analysis enables more precise attribution of model failures than aggregate benchmark scores.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
