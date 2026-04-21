---
type: concept
title: Structural Consistency
slug: structural-consistency
date: 2026-04-20
updated: 2026-04-20
aliases: [logical consistency, 结构一致性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Structural Consistency** (结构一致性) — the property that successive reasoning steps form a logically continuous, mutually compatible sequence rather than an isolated collection of locally plausible fragments.

## Key Points

- The central empirical claim of the paper is that structural consistency matters more than local content correctness for learning long-CoT reasoning.
- Shuffling, deleting, or inserting reasoning steps degrades accuracy much more than wrong answers, corrupted digits, or removed reasoning keywords.
- Preserving individual step plausibility is insufficient if the sequence no longer supports coherent references, revisions, and case analysis.
- The paper treats reflection, backtracking, and self-validation as useful only when they appear in a logically connected order.
- High output length and frequent reasoning keywords do not guarantee structural consistency or strong benchmark accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-llms-2502-07374]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-llms-2502-07374]].
