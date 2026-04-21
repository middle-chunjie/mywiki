---
type: concept
title: Identifier Naming
slug: identifier-naming
date: 2026-04-20
updated: 2026-04-20
aliases: [variable naming, 标识符命名]
tags: []
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Identifier Naming** (标识符命名) — the choice of names for variables or functions, ideally without changing program semantics when renamings are consistent.

## Key Points

- [[hooda-2024-do-2402-05980]] evaluates identifier naming using random renaming and shuffled-name mutations.
- The mutations rename only user-defined variables and exclude reserved keywords and function parameters.
- Variable-name perturbations are among the most damaging, with AME often in the `15%` to `29%` range across models and datasets.
- The paper interprets this as evidence that current code LLMs rely heavily on lexical naming cues.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
