---
type: concept
title: Uptraining
slug: uptraining
date: 2026-04-20
updated: 2026-04-20
aliases: [continued adaptation training, 增量改造训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Uptraining** (增量改造训练) — continuing training from a pretrained model after a structural modification so the new architecture can recover performance with much less budget than training from scratch.

## Key Points

- The paper shows that pretrained vanilla transformers can be converted into Block Transformers instead of always training the hierarchical model from scratch.
- Its uptraining setup uses only `10%` of the original training steps, or about `30B` tokens, yet approaches the quality of fully pretrained Block Transformers.
- Initialization splits pretrained vanilla layers across the block and token decoders and adds a fully connected projection from concatenated token embeddings into block-decoder hidden space.
- The paper reports that this strategy substantially outperforms random initialization and suggests even better recovery may come from more deliberate weight-initialization schemes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ho-2024-block-2406-02657]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ho-2024-block-2406-02657]].
