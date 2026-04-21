---
type: concept
title: In-Context Few-Shot Learning
slug: in-context-few-shot-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [multimodal in-context few-shot learning, 上下文少样本学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**In-Context Few-Shot Learning** (上下文少样本学习) — a test-time learning pattern where a model infers how to solve a task from a small number of examples placed in the prompt without updating parameters.

## Key Points

- The paper studies few-shot behavior in multimodal prompts where the demonstrations contain both images and text.
- Hard tasks such as speedometer reading and line-chart reasoning fail in `0-shot` and remain brittle in `1-shot`, but improve noticeably in `2-shot`.
- The authors note that few-shot examples usually need to match the query format closely, making them longer and more cumbersome than plain instructions.
- Despite its benefits, the report keeps zero-shot prompting as the default presentation mode to reduce potential information leakage from demonstrations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-dawn-2309-17421]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-dawn-2309-17421]].
