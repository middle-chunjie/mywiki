---
type: concept
title: Variable Misuse Bug
slug: variable-misuse-bug
date: 2026-04-20
updated: 2026-04-20
aliases: [VARMISUSE, variable misuse, 变量误用缺陷]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Variable Misuse Bug** (变量误用缺陷) — a defect in which a program uses an in-scope variable at some location even though a different in-scope variable is the correct choice there.

## Key Points

- The paper assumes the correct repair variable already appears somewhere else in the same program, so repair can be cast as pointing rather than generation.
- In the authors' formulation, the target output is the faulty variable-use location plus any occurrence of the correct variable.
- The main experiments study Python functions, while a comparison to prior work uses the C# MSR-VarMisuse dataset with type-compatible candidates.
- Enumerative slot-wise repair is shown to be especially brittle on this bug class because most queried slots are not the true bug location.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[vasic-2019-neural-1904-01720]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[vasic-2019-neural-1904-01720]].
