---
type: entity
title: FLAVA
slug: flava
date: 2026-04-20
entity_type: tool
aliases: [Foundational Language And Vision Alignment]
tags: []
---

## Description

FLAVA is the larger pretrained vision-language model used in [[ray-nd-cola]] as a baseline and adaptation target with existing multimodal layers. The paper uses it to test whether pretrained multimodal attention alone is sufficient for attribute-object compositional retrieval.

## Key Contributions

- Serves as the main comparison point against CLIP-based multimodal adaptation.
- Shows that replacing or retraining multimodal layers on compositional data can outperform simply tuning pretrained multimodal layers.

## Related Concepts

- [[vision-language-model]]
- [[cross-modal-alignment]]
- [[multimodal-fusion]]

## Sources

- [[ray-nd-cola]]
