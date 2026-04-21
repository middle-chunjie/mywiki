---
type: concept
title: Controlled Generation
slug: controlled-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [controllable generation, 可控生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Controlled Generation** (可控生成) — text generation conditioned on an explicit target signal so that produced outputs satisfy a desired property or constraint.

## Key Points

- The paper reframes embedding inversion as controlled generation rather than one-shot decoding from a vector.
- Vec2Text conditions on the target embedding `e`, the current hypothesis embedding `ê^(t)`, and the residual `e - ê^(t)` to guide each correction round.
- Encoder feedback matters substantially: after `50` greedy correction rounds, the feedback-enabled model gets `52.0%` exact matches, versus `4.2%` without feedback.
- Sequence-level beam search on top of the corrective generator further raises exact recovery, indicating that control quality depends on both model conditioning and search.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[morris-2023-text-2310-06816]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[morris-2023-text-2310-06816]].
