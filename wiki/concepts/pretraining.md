---
type: concept
title: Pretraining
slug: pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [pre-training, 预训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pretraining** (预训练) — large-scale self-supervised training on broad text corpora that establishes a language model's core knowledge and capabilities before task-specific adaptation.

## Key Points

- The paper argues that ChatGPT's advantage primarily comes from stronger base-model pretraining rather than from proprietary supervised fine-tuning traces alone.
- The authors note that their imitation corpora are about `1000x` smaller than pretraining scale, making broad capability transfer unrealistic.
- Scaling the base LM yields stronger benchmark gains than adding more imitation data, reinforcing the importance of pretraining quality and scale.
- The discussion section frames fine-tuning as a lightweight knowledge extractor rather than a replacement for weak pretraining.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gudibande-2023-false-2305-15717]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gudibande-2023-false-2305-15717]].
