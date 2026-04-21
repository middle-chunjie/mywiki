---
type: concept
title: Self-Correction
slug: self-correction
date: 2026-04-20
updated: 2026-04-20
aliases: [self correction, 自我纠错]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Correction** (自我纠错) — the ability of a model to inspect its own earlier response and revise it into a more accurate later response without receiving external corrective feedback.

## Key Points

- The paper studies intrinsic self-correction, where the model must infer its own mistakes rather than rely on tools, judges, or revealed answers.
- SCoRe frames self-correction as a two-turn policy optimization problem instead of a pure imitation or prompting problem.
- The authors show that positive self-correction should increase incorrect-to-correct transitions while avoiding correct-to-incorrect regressions.
- Reward shaping is used to explicitly favor revision trajectories that improve correctness between the first and second attempts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kumar-2024-training-2409-12917]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kumar-2024-training-2409-12917]].
