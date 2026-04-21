---
type: concept
title: Inductive Program Synthesis
slug: inductive-program-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [IPS, inductive program synthesis, 归纳式程序综合]
tags: [program-synthesis, search]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Inductive Program Synthesis** (归纳式程序综合) — the task of recovering a program that is consistent with observed input-output examples rather than an explicit formal specification.

## Key Points

- [[balog-2017-deepcoder-1611-01989]] defines IPS as producing a program consistent with input-output examples and separates it into a search problem and a ranking problem.
- The paper argues that prior differentiable-interpreter methods solve each synthesis task largely independently, limiting transfer across problems.
- DeepCoder reframes IPS as a data-driven problem by generating large synthetic corpora of programs and learning search guidance from them.
- The paper focuses on the search side of IPS by predicting DSL function usage and using those predictions to prioritize symbolic solvers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[balog-2017-deepcoder-1611-01989]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[balog-2017-deepcoder-1611-01989]].
