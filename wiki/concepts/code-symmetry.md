---
type: concept
title: Code Symmetry
slug: code-symmetry
date: 2026-04-20
updated: 2026-04-20
aliases: [code symmetry, 代码对称性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Symmetry** (代码对称性) — a semantics-preserving transformation of program structure whose application leaves the program's input-output behavior unchanged.

## Key Points

- The paper treats code reordering, variable renaming, and loop-level rewrites as motivating examples of symmetries for program analysis.
- Its central claim is that code-analysis models should be invariant to such transformations whenever the downstream task depends only on semantics.
- The work elevates these transformations from ad hoc augmentations to algebraic objects that can be reasoned about as groups.
- SymC is designed to preserve a symmetry group by construction instead of trying to memorize transformed variants during training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pei-2024-exploiting-2308-03312]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pei-2024-exploiting-2308-03312]].
