---
type: concept
title: Prompt tuning
slug: prompt-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt-based learning, 提示调优]
tags: [prompting, adaptation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt tuning** (提示调优) — adapting a frozen pre-trained model to a downstream task by learning small task-specific prompts instead of updating the full model parameters.

## Key Points

- The paper imports the prompt-tuning idea from NLP and vision-language learning into fine-grained visual retrieval.
- FRPT implements prompting in image space rather than text space by perturbing discriminative regions of the input image.
- The authors position prompt tuning as a way to reformulate the downstream task so it more closely matches the original pre-training regime.
- In this work, prompt tuning is paired with a frozen backbone and a lightweight adaptation head rather than used alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-finegrained-2207-14465]].
