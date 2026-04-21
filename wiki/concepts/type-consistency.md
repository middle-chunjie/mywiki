---
type: concept
title: Type Consistency
slug: type-consistency
date: 2026-04-20
updated: 2026-04-20
aliases: [type correctness]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Type consistency** (类型一致性) — the property that identifiers, expressions, and operations respect the types inferred or declared in the surrounding program context.

## Key Points

- The main instantiation of MGD in the paper enforces type-consistent identifier dereferences.
- Type-consistent suggestions are computed from the inferred type of the dereferenced object and the accessible members of that type.
- Improvements in compilation rate are partly attributed to avoiding symbol-not-found errors caused by type-inconsistent member names.
- The paper extends the same style of guidance beyond dereferences to other semantically typed actions such as enum cases and valid method signatures.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[agrawal-nd-monitorguided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[agrawal-nd-monitorguided]].
