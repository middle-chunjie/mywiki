---
type: concept
title: Human Evaluation
slug: human-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [manual evaluation, 人工评测]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Human Evaluation** (人工评测) — assessment by human annotators who directly judge the quality of model outputs against task-specific criteria.

## Key Points

- [[cui-2022-codeexp-2211-15395]] uses four aspects: general adequacy, coverage, coherence, and fluency.
- Annotators score each aspect on a `0-4` Likert scale and evaluate model outputs alongside hidden reference docstrings.
- The study reports `180` sampled test examples, `9` systems per example, `4` aspects, and `6,480` total scores from `10` annotators.
- Human evaluation reveals that refined-data fine-tuning is substantially better than raw-only training and that coverage remains challenging even for humans.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cui-2022-codeexp-2211-15395]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cui-2022-codeexp-2211-15395]].
