---
type: concept
title: Masked Auto-Encoder
slug: masked-auto-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [MAE, 掩码自编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Masked Auto-Encoder** (掩码自编码器) — a self-supervised architecture that learns representations by encoding a corrupted input and reconstructing the original content from the resulting latent representation.

## Key Points

- RetroMAE adapts masked auto-encoding to dense retrieval rather than vision or generic language modeling.
- The encoder sees a moderately masked sentence, while the decoder sees a separately and more aggressively masked view of the same sentence.
- Reconstruction is intentionally made dependent on the sentence embedding so that the encoder learns stronger retrieval-oriented semantics.
- The paper combines the decoder reconstruction loss with the encoder MLM loss as ``L = L_enc + L_dec``.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2022-retromae-2205-12035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2022-retromae-2205-12035]].
