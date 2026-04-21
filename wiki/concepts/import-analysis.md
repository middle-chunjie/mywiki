---
type: concept
title: Import Analysis
slug: import-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [import resolution, 导入分析]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Import Analysis** (导入分析) — static resolution of imported modules and classes to recover the valid user-defined type space available in a program.

## Key Points

- HiTYPER collects in-file classes first and then augments them with locally and globally imported classes.
- The recovered import graph lets the system distinguish user-defined class instantiation from ordinary function calls.
- Import analysis also checks imported classes for operator overloading behavior, which affects applicable typing rules.
- The paper credits import analysis as one reason the static component can infer more user-defined types than prior tools.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[peng-2022-static-2105-03595]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[peng-2022-static-2105-03595]].
