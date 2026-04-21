---
type: concept
title: Constitutional AI
slug: constitutional-ai
date: 2026-04-20
updated: 2026-04-20
aliases: [CAI, constitutional ai, 宪法式AI]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Constitutional AI** (宪法式AI) — a rule-based alignment approach in which a model critiques and revises its own outputs using a constitution-like list of principles instead of relying only on human preference labels.

## Key Points

- This paper treats Constitutional AI as the closest rule-based reference point for SELF-ALIGN, but distinguishes it from SELF-ALIGN's direct principle-conditioned answer generation.
- According to the comparison in the paper, Constitutional AI performs post-generation self-critique on an existing answer, whereas SELF-ALIGN has the model select and apply rules while producing the answer.
- The authors argue that Constitutional AI still depends on an RLHF warm-up stage, while SELF-ALIGN targets alignment from scratch with minimal human supervision.
- The paper also notes a practical difference: SELF-ALIGN must fit all rules into the prompt context during self-alignment, whereas Constitutional AI is less constrained by that token-budget issue because it operates as a later critique step.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2023-principledriven-2305-03047]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2023-principledriven-2305-03047]].
