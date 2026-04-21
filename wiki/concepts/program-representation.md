---
type: concept
title: Program Representation
slug: program-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [program embedding, 程序表示]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Program Representation** (程序表示) — a mapping from source code into vectors or structured states that preserve information useful for downstream software-engineering or code-understanding tasks.

## Key Points

- [[long-2022-multiview]] argues that strong program representations should separate complementary semantic signals instead of collapsing them into one syntax-heavy structure.
- The paper instantiates program representation with four graph views: `DFG`, `CFG`, `RWG`, and `CG`.
- Each view is encoded by a GGNN and then fused by concatenation `z = z_DFG ⊕ z_CFG ⊕ z_RWG ⊕ z_CG`.
- The resulting representation improves both single-label and multi-label algorithm classification benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[long-2022-multiview]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[long-2022-multiview]].
