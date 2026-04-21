---
type: concept
title: Model Imitation
slug: model-imitation
date: 2026-04-20
updated: 2026-04-20
aliases: [proprietary model imitation, model stealing, 模型模仿]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Model Imitation** (模型模仿) — training a student model on the observable outputs of a stronger black-box model in order to approximate its behavior without access to its internals.

## Key Points

- The paper distinguishes broad imitation of ChatGPT's general behavior from local imitation of a specific capability such as Natural Questions answering.
- Broad imitation is implemented with public conversation traces rather than teacher logits or internal representations, making it weaker than classical white-box distillation.
- Scaling imitation data from `20M` to `150M` tokens substantially improves stylistic similarity to ChatGPT but not broad benchmark performance.
- The paper argues that current imitation methods do not erase the capabilities gap between weaker open models and stronger proprietary systems.
- Task-specific imitation can still work well when the target behavior is narrow and directly represented in the imitation corpus.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gudibande-2023-false-2305-15717]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gudibande-2023-false-2305-15717]].
