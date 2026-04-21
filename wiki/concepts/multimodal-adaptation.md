---
type: concept
title: Multimodal Adaptation
slug: multimodal-adaptation
date: 2026-04-20
updated: 2026-04-20
aliases: [multimodal adaptation, multi-modal adaptation, 多模态适配]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multimodal Adaptation** (多模态适配) — adapting a pretrained multimodal or dual-encoder system to a target task by adding or tuning modules that explicitly model interactions between modalities.

## Key Points

- [[ray-nd-cola]] studies multimodal adaptation as a cheaper alternative to retraining a vision-language model from scratch.
- Its best method adds a lightweight multimodal encoder-decoder on top of frozen CLIP or FLAVA features instead of tuning only separate image and text branches.
- The paper distinguishes MM-Pred from MM-Adapter, with MM-Adapter aligning the multimodal output to frozen text features as a regularized adaptor.
- On `COLA`, multimodal adaptation consistently outperforms prompt tuning, linear probes, and late unimodal fine-tuning.
- The results suggest that training multimodal layers on compositional attribute-object data is more important than merely inheriting pretrained multimodal layers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ray-nd-cola]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ray-nd-cola]].
