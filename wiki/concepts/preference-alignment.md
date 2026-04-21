---
type: concept
title: Preference Alignment
slug: preference-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [preference tuning, alignment by preferences, 偏好对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Preference Alignment** (偏好对齐) — optimization that adapts a model's outputs to match the preferences of a downstream evaluator or generator.

## Key Points

- BIDER uses preference alignment after supervised distillation so the refiner better matches a downstream LLM's information acquisition preferences.
- The reward compares answer quality with refined evidence against answer quality with the original retrieved documents.
- Alignment improves both effective content and ordering: gold answers appear more often and earlier in the generated evidence after this stage.
- The paper argues this stage is helpful but still less important than constructing a strong synthesis target in the first place.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-bider-2402-12174]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-bider-2402-12174]].
