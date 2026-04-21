---
type: concept
title: Algebraic Data Type
slug: algebraic-data-type
date: 2026-04-20
updated: 2026-04-20
aliases: [ADT, algebraic data types, 代数数据类型]
tags: [formal-methods, program-analysis]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Algebraic Data Type** (代数数据类型) - a tree-oriented type system built from constructors and selectors that can represent structured objects such as abstract syntax trees.

## Key Points

- The background section models programs as ADT instances so that repair can reason over finite trees instead of raw text.
- Constructors define node kinds, selectors define typed outgoing edges, and testers identify constructor membership.
- The paper uses ADTs to formalize AST structure in the SMT and MAX-SMT repair stages.
- Valid ADT instances must be acyclic, which matches the intended structure of syntax trees.
- Abstract syntax trees are presented as concrete instances of ADTs for the target programming setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[mora-2024-synthetic-2406-03636]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[mora-2024-synthetic-2406-03636]].
