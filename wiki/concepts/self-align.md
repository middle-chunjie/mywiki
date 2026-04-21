---
type: concept
title: Self-Align
slug: self-align
date: 2026-04-20
updated: 2026-04-20
aliases: [SELF-ALIGN, self alignment, 自对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Align** (自对齐) — a rule-guided alignment framework in which a language model generates its own aligned supervision from a small human-authored principle set and then fine-tunes on those self-produced responses.

## Key Points

- The paper instantiates SELF-ALIGN as a four-stage pipeline: topic-guided self-instruct, principle-driven self-alignment, principle engraving, and verbose cloning.
- Human supervision is deliberately tiny: `195` seed prompts, `16` principles, and `5` in-context exemplars, totaling fewer than `300` written lines.
- After filtering, the method yields `258,878` self-aligned query-response pairs for principle engraving and `358,777` verbose pairs for the final distillation stage.
- SELF-ALIGN is positioned as an alternative to heavy SFT + RLHF pipelines because it does not require a large annotated preference dataset or a proprietary teacher assistant.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2023-principledriven-2305-03047]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2023-principledriven-2305-03047]].
