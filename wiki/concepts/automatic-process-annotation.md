---
type: concept
title: Automatic Process Annotation
slug: automatic-process-annotation
date: 2026-04-20
updated: 2026-04-20
aliases: [automatic process annotation, 自动过程标注]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Automatic Process Annotation** (自动过程标注) — a method for labeling intermediate reasoning steps automatically by estimating whether they can still be completed into a correct solution.

## Key Points

- [[wang-2024-mathshepherd-2312-08935]] defines the quality of a step by its potential to reach the gold final answer.
- The paper uses a completer to sample `` `N = 8` `` continuations from an intermediate step and derives labels from the correctness of their final answers.
- It studies both hard estimation, where any successful completion makes the step positive, and soft estimation, where the label is the success frequency.
- Manual inspection on GSM8K shows the automatic labels can be fairly accurate, with hard estimation reaching `86%` at `N = 4`.
- The approach replaces expensive human process labels and scales PRM data construction to roughly `170k` GSM8K solutions and `270k` MATH solutions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-mathshepherd-2312-08935]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-mathshepherd-2312-08935]].
