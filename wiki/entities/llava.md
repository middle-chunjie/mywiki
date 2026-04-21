---
type: entity
title: LLaVA
slug: llava
date: 2026-04-20
entity_type: tool
aliases: [Large Language and Vision Assistant, LLaVA-1]
tags: []
---

## Description

LLaVA is an open-source large multimodal model that connects a visual encoder (CLIP ViT-L/14) with a language decoder (LLaMA/Vicuna) via a linear projection, trained with visual instruction tuning on GPT-4-generated instruction-following data.

## Key Contributions

- Provides an open-source LMM backbone for visual instruction following evaluated in Img2Loc as a cheaper alternative to GPT-4V.
- In Img2Loc experiments, LLaVA underperforms prior supervised baselines on both Im2GPS3k and YFCC4k, highlighting the performance gap between open and proprietary LMMs on spatial reasoning.

## Related Concepts

- [[large-multimodal-model]]
- [[contrastive-language-image-pretraining]]
- [[training-free-adaptation]]

## Sources

- [[zhou-2024-imgloc-2403-19584]]
