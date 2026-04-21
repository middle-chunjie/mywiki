---
type: concept
title: Text-to-Image Generation
slug: text-to-image-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [T2I, text to image generation, 文本到图像生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Text-to-Image Generation** (文本到图像生成) — the task of synthesizing an image that matches a natural-language prompt, including both object content and structured constraints such as counts, relations, scale, or text.

## Key Points

- The paper treats T2I generation as more than visual plausibility and emphasizes alignment with prompt structure.
- It identifies count, spatial relation, relative scale, and text rendering as persistent failure modes for current T2I systems.
- `VPGen` reframes T2I generation as a staged process with an explicit intermediate layout rather than a pure end-to-end mapping.
- The experiments compare diffusion-based and multimodal autoregressive baselines, with diffusion models generally stronger overall.
- The open-ended prompt analysis shows that many prompts are dominated by objects and attributes, which partially explains why layout-focused gains are clearer on skill-based prompts than on general prompts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cho-2023-visual-2305-15328]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cho-2023-visual-2305-15328]].
