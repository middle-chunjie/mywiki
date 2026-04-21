---
type: concept
title: Training-Free Adaptation
slug: training-free-adaptation
date: 2026-04-20
updated: 2026-04-20
aliases: [training-free, zero-shot adaptation, inference-only adaptation, 免训练适配]
tags: [foundation-model, prompting, zero-shot]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Training-Free Adaptation** (免训练适配) — the use of a pretrained model for a new task without any gradient updates, relying solely on prompt engineering, retrieval augmentation, or in-context demonstrations to steer model behavior.

## Key Points

- Img2Loc demonstrates training-free adaptation for image geolocation: CLIP and the LMM (GPT-4V / LLaVA) are used as-is; no task-specific fine-tuning is performed on any geotagged data.
- Training-free methods significantly reduce computational overhead and eliminate the need for labeled training sets, which are expensive to curate for geolocation at global scale.
- The approach relies on the emergent capability of LMMs to interpret spatial context from image content and coordinate hints in a structured prompt.
- Training-free adaptation trades peak performance for generality; in Img2Loc, the gap is visible in the weaker LLaVA variant, which underperforms supervised methods.
- The paradigm generalizes naturally as newer, stronger LMMs become available — the system benefits from model improvements without re-training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2024-imgloc-2403-19584]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2024-imgloc-2403-19584]].
