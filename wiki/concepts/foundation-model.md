---
type: concept
title: Foundation Model
slug: foundation-model
date: 2026-04-20
updated: 2026-04-20
aliases: [large-scale model, pretrained model, base model, 基础模型]
tags: [large-models, pretraining, transfer-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Foundation Model** (基础模型) — a large-scale model trained on broad data that can be adapted to a wide range of downstream tasks, typically without task-specific training from scratch (Bommasani et al., 2021).

## Key Points

- Foundation models are characterized by enormous parameter counts, broad pre-training data, and strong zero-shot or few-shot generalization.
- Due to scale and proprietary training data, they are typically deployed under Model-as-a-Service constraints — users interact via API only.
- In computer vision, they enable in-context learning: the model generalizes to unseen tasks by conditioning on prompt examples rather than weight updates.
- The term encompasses multimodal models (Flamingo, CLIP) as well as vision-only and language-only pre-trained backbones.
- Adaptation mechanisms under frozen parameters — including in-context learning and prompt retrieval — are key research challenges for foundation models.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-nd-what]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-nd-what]].
