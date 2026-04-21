---
type: concept
title: Black-Box Model
slug: black-box-model
date: 2026-04-20
updated: 2026-04-20
aliases: [black-box LLM, API-only model]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Black-Box Model** (黑盒模型) — a model that can be queried for outputs but does not expose internals such as parameters, gradients, or token-level logits needed for direct fine-tuning.

## Key Points

- [[yang-2023-prca]] treats GPT-style API-served generators as black-box models that cannot be updated with standard backpropagation.
- The paper argues that black-box access breaks prior retrieval-augmented training recipes that depend on final-layer logits or frequent reward calls.
- PRCA addresses this constraint by moving trainable adaptation into a separate module between retriever and generator.
- The reward-driven stage uses only the generated answer to compute reward, avoiding any requirement for hidden-state or gradient access.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2023-prca]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2023-prca]].
