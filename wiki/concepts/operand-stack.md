---
type: concept
title: Operand Stack
slug: operand-stack
date: 2026-04-20
updated: 2026-04-20
aliases: [stack machine operand stack, 操作数栈]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Operand Stack** (操作数栈) — a last-in-first-out runtime stack used by stack-based virtual machines to pass intermediate values between instructions.

## Key Points

- TranCS simulates instruction execution by maintaining an operand stack while traversing bytecode from top to bottom.
- Data dependencies are inferred by tracking which instruction pushes a value later popped by another instruction.
- Translation rules use stack-derived values to fill placeholders such as `[ps]` in instruction templates.
- The algorithm initializes an empty operand stack with compile-time depth `d` before processing each instruction sequence.
- Recovering stack behavior is essential for making bytecode translations short but semantically informative.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-code-2202-08029]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-code-2202-08029]].
