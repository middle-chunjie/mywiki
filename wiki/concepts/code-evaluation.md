---
type: concept
title: Code Evaluation
slug: code-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [program evaluation, 代码评估]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Evaluation** (代码评估) — the task of estimating the quality or correctness of generated code with respect to its intended behavior, requirements, or references.

## Key Points

- [[dong-2023-codescore-2301-09043]] argues that code evaluation should prioritize functional correctness rather than surface similarity to a single reference implementation.
- The paper formulates evaluation under three input regimes: generated code paired with reference code, natural-language requirements, or both.
- It positions learned evaluation as a practical middle ground between cheap but weak match-based metrics and expensive execution-based testing.
- CodeScore is trained to output continuous scores aligned with execution-derived correctness signals instead of hand-designed overlap heuristics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2023-codescore-2301-09043]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2023-codescore-2301-09043]].
