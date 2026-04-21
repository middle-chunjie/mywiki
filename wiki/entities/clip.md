---
type: entity
title: CLIP
slug: clip
date: 2026-04-20
entity_type: tool
aliases: [Contrastive Language-Image Pretraining]
tags: []
---

## Description

CLIP is the pretrained image-text dual-encoder backbone used by [[jin-2024-mvadapter-2301-07868]]. MV-Adapter freezes most CLIP weights and adds small task-specific branches to transfer it from image-text pretraining to video-text retrieval.

## Key Contributions

- Supplies the vision and text encoders that define the paper's retrieval backbone.
- Provides the pretrained cross-modal embedding space that MV-Adapter adapts rather than retraining from scratch.
- Serves as the reference point for comparing full fine-tuning against parameter-efficient transfer.

## Related Concepts

- [[dual-encoder]]
- [[cross-modal-alignment]]
- [[transfer-learning]]

## Sources

- [[jin-2024-mvadapter-2301-07868]]
