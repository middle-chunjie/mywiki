---
type: concept
title: Decoder-Only Model
slug: decoder-only-model
date: 2026-04-20
updated: 2026-04-20
aliases: [decoder-only transformer, 解码器模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Decoder-Only Model** (仅解码器模型) — an autoregressive architecture that predicts the next token conditioned on previous context without a separate encoder stack.

## Key Points

- The survey treats Transformer-Decoder (`TD`) models as a distinct CodePTM family and lists GPT-C as a representative example.
- Decoder-only models are naturally aligned with generation tasks such as code completion because they model left-to-right continuation directly.
- The paper notes the complementary weakness of decoder-only models on classification-style workloads, where encoder-side bidirectional context is often more suitable.
- This architecture-level mismatch is part of the paper's broader claim that model family should be chosen according to downstream task type.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[niu-2022-deep-2205-11739]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[niu-2022-deep-2205-11739]].
