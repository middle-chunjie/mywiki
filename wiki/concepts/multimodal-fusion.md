---
type: concept
title: Multimodal Fusion
slug: multimodal-fusion
date: 2026-04-20
updated: 2026-04-20
aliases: [multimodal fusion, multi-modal fusion, 多模态融合]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multimodal Fusion** (多模态融合) — combining representations from different modalities into a shared interaction space so downstream predictions depend on cross-modal structure rather than on separate unimodal scores alone.

## Key Points

- [[ray-nd-cola]] compares unimodal adaptation against multimodal fusion for attribute-object retrieval and finds fusion to be consistently stronger.
- The paper's main fusion design follows an MDETR-style pattern: self-attention over concatenated image and text tokens, then `[CLS]` cross-attends to the fused sequence.
- Supplemental experiments show that several fusion styles inspired by FLAVA, ALBEF, MDETR, and FIBER outperform similarly sized unimodal tuning baselines.
- The paper reports that MM-Adapter is generally stronger than MM-Pred across different fusion architectures.
- The gains are especially large on the harder `COLA` benchmark, where fine-grained binding requires explicit interaction between image regions and text tokens.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ray-nd-cola]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ray-nd-cola]].
