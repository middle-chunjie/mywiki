---
type: concept
title: Transformer Encoder
slug: transformer-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [bidirectional transformer encoder, Transformer 编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Transformer Encoder** (Transformer 编码器) — a stack of self-attention and feed-forward layers that maps an input sequence to contextualized hidden states suitable for sequence representation learning.

## Key Points

- CoCoSoDa uses a bidirectional Transformer encoder as the architecture for both the main encoders and the momentum encoders.
- The reported setup follows UniXcoder with `12` layers, hidden size `768`, and `12` attention heads.
- Code and query encoders share parameters, reducing parameter count while still enabling a shared retrieval space.
- Sequence representations are formed by averaging the last-layer hidden states before contrastive pre-training and retrieval fine-tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2023-cocosoda-2204-03293]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2023-cocosoda-2204-03293]].
