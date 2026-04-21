---
type: concept
title: Video Captioning
slug: video-captioning
date: 2026-04-20
updated: 2026-04-20
aliases: [视频字幕生成, video description generation, dense video captioning]
tags: [multimodal, generation, video, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Video Captioning** (视频字幕生成) — the task of generating natural language descriptions for video clips, typically conditioned on visual features extracted from the video and optionally on temporal segment boundaries.

## Key Points

- Standard evaluation metrics include BLEU@3/4, ROUGE-L, METEOR, CIDEr-D; the repetition metric RE@4 (lower is better) measures caption diversity.
- CrossCLR improves video captioning by feeding richer joint embeddings to the MART captioning model: CIDEr-D improves from 54.07 (COOT) to 58.65 (CrossCLR), and to 61.10 with additional video-level embeddings.
- Performance depends on the quality of the visual encoder features; using both clip-level and video-level hierarchical embeddings consistently improves captioning.
- Human performance on Youcook2 is BLEU@4 = 15.20, ROUGE-L = 45.10, METEOR = 25.90 — CrossCLR (12.04 / 38.63 / 20.17) remains well below human performance.
- Captioning performance serves as a proxy for the semantic richness of learned embeddings: better joint embeddings enable more accurate and diverse generated descriptions.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zolfaghari-2021-crossclr-2109-14910]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zolfaghari-2021-crossclr-2109-14910]].
