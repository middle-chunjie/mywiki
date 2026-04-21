---
type: concept
title: Heap Search
slug: heap-search
date: 2026-04-20
updated: 2026-04-20
aliases: [Heap Search, heap search]
tags: [search, enumeration]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Heap Search** — a bottom-up enumerative search algorithm that uses per-nonterminal heaps and cached successors to enumerate grammar-generated programs in descending probability order.

## Key Points

- [[fijalkow-2022-scaling]] proves Heap Search is loss optimal because it emits each program exactly once and in non-increasing PCFG probability order.
- For each non-terminal, the algorithm maintains `Heap_T`, `Succ_T`, and `Seen_T`, and computes successors through a recursive `Query(T, x)` procedure.
- Its bottom-up construction enables cached evaluation of partial programs, avoiding repeated evaluation that slows top-down methods such as `A*`.
- In the random-PCFG experiments, Heap Search generates `2.35x` more programs than `A*`, and on learned PCFGs it reaches `38,735` programs/s while solving `97/137` tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fijalkow-2022-scaling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fijalkow-2022-scaling]].
