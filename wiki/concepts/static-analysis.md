---
type: concept
title: Static Analysis
slug: static-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [program analysis]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Static analysis** (静态分析) — the computation of syntactic or semantic properties of programs without executing them, often by reasoning over partial or complete program structure.

## Key Points

- The paper relies on IDE-style static analysis over incomplete code, not only over complete compilable programs.
- Static analysis is used to infer types, resolve accessible identifiers, and expose semantic constraints from repository context and dependencies.
- The authors implement analysis access through language servers, treating them as backends for monitor queries.
- The paper also notes that static analysis on partial programs is heuristic and can be imprecise or incomplete.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[agrawal-nd-monitorguided]]
- [[ding-2023-cocomic-2212-10007]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[agrawal-nd-monitorguided]].
