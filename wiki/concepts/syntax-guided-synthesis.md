---
type: concept
title: Syntax-Guided Synthesis
slug: syntax-guided-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [SyGuS, syntax-guided synthesis, 语法引导合成]
tags: [program-synthesis, grammar]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Syntax-Guided Synthesis** (语法引导合成) — a program synthesis formulation in which the valid candidate space is explicitly constrained by a grammar or DSL together with syntactic typing rules.

## Key Points

- [[fijalkow-2022-scaling]] adopts the SyGuS setting where a DSL and syntactic constraints are compiled into a CFG that generates only admissible programs.
- The grammar may encode contextual information such as the last `n` primitives, semantic properties, and the requested program type.
- This formulation lets the authors separate learning from search: the model predicts production probabilities, and search algorithms operate over the resulting PCFG.
- The paper's theoretical analysis of loss and optimality is built directly on this grammar-constrained search space.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fijalkow-2022-scaling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fijalkow-2022-scaling]].
