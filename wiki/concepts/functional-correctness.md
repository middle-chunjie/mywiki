---
type: concept
title: Functional Correctness
slug: functional-correctness
date: 2026-04-20
updated: 2026-04-20
aliases: [behavioral correctness, 功能正确性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Functional Correctness** (功能正确性) — the degree to which code satisfies the intended task behavior, typically judged by whether it produces correct outputs on representative inputs.

## Key Points

- [[dong-2023-codescore-2301-09043]] treats functional correctness as the central target for evaluating generated code, not token overlap with a reference.
- The paper operationalizes this notion through execution-derived supervision, especially [[pass-ratio]] over large test sets.
- Functionally equivalent programs may differ substantially in syntax, which is why the paper criticizes BLEU-style metrics for code evaluation.
- CodeScore is designed to approximate functional correctness without re-running full execution at evaluation time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2023-codescore-2301-09043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2023-codescore-2301-09043]].
