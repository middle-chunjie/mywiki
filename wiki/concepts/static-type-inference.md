---
type: concept
title: Static Type Inference
slug: static-type-inference
date: 2026-04-20
updated: 2026-04-20
aliases: [rule-based type inference, 静态类型推断]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Static Type Inference** (静态类型推断) — the derivation of variable and expression types from source code structure and typing rules without executing the program.

## Key Points

- HiTYPER encodes typing judgments such as `` `pi |- e : theta` `` in expression nodes and activates them during graph traversal.
- Forward inference starts from nodes whose inputs are already solved and propagates types through operations, calls, containers, and comprehensions.
- The paper emphasizes that static inference is frequency-insensitive, which makes it especially useful for rare and user-defined types.
- Compared with Pytype and Pyre Infer, HiTYPER's static component infers substantially more correct annotations while maintaining high precision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
