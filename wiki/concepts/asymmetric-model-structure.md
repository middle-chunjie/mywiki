---
type: concept
title: Asymmetric Model Structure
slug: asymmetric-model-structure
date: 2026-04-20
updated: 2026-04-20
aliases: [非对称模型结构]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Asymmetric Model Structure** (非对称模型结构) — an encoder-decoder design in which the encoder and decoder have intentionally different capacities so the representation burden falls mainly on one side.

## Key Points

- RetroMAE uses a full BERT-base encoder but only a `1`-layer transformer decoder.
- The small decoder is meant to stop local decoder computation from replacing the need for a strong sentence embedding.
- Larger decoders with `2` or `3` layers do not improve retrieval quality in the paper's ablations.
- The `1`-layer choice is also required for the paper's enhanced decoding mechanism.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2022-retromae-2205-12035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2022-retromae-2205-12035]].
