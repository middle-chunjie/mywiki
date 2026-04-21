---
type: concept
title: Encoder-Only Model
slug: encoder-only-model
date: 2026-04-20
updated: 2026-04-20
aliases: [encoder-only transformer, 编码器模型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Encoder-Only Model** (仅编码器模型) — a model architecture that maps an input sequence to contextual representations without an autoregressive decoder.

## Key Points

- The survey identifies Transformer-Encoder (`TE`) models as one of the main architecture families used by CodePTMs.
- Encoder-only models are reported to work well for understanding tasks such as bug detection, code search, and variable-misuse prediction.
- The paper argues that encoder-only models are disadvantaged for generation because output length is not known a priori and decoding must be added indirectly.
- CuBERT, C-BERT, JavaBERT, CugLM, CodeBERT, GraphCodeBERT, and SynCoBERT are grouped as encoder-side-centric examples in the survey tables.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[niu-2022-deep-2205-11739]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[niu-2022-deep-2205-11739]].
