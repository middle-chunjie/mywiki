---
type: concept
title: Type Rejection
slug: type-rejection
date: 2026-04-20
updated: 2026-04-20
aliases: [backward type rejection, 类型拒绝]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Type Rejection** (类型拒绝) — a backward validation process that removes candidate types violating operation-specific validity constraints or cross-input consistency constraints.

## Key Points

- HiTYPER uses rejection rules to sanitize neural recommendations instead of throwing away the entire inference problem when a conflict appears.
- Each rule checks both whether an input type belongs to the valid type set for an operation and whether it is compatible with the other inputs.
- Rejected candidates are propagated backward through earlier nodes so the effects of invalid recommendations are removed transitively.
- This mechanism is central to the paper's claim that final accepted assignments remain statically justified.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
