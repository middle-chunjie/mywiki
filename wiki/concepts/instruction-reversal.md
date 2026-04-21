---
type: concept
title: Instruction Reversal
slug: instruction-reversal
date: 2026-04-20
updated: 2026-04-20
aliases: [reverse instruction generation, 指令反演]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Instruction Reversal** (指令反演) — a data-generation technique that starts from an output artifact such as code and synthesizes the corresponding instruction or task description that could have produced it.

## Key Points

- Phi-4 uses instruction reversal mainly for code and related tasks to create high-fidelity instruction-output pairs.
- The pipeline retains only cases where regenerated code remains faithful to the original artifact, improving instruction-to-output alignment.
- The paper presents instruction reversal as one component of a broader synthetic-data generation strategy rather than a standalone training method.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
