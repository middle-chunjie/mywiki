---
type: concept
title: Quantization
slug: quantization
date: 2026-04-20
updated: 2026-04-20
aliases: [k-bit quantization, 量化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Quantization** (量化) — the process of mapping higher-precision numerical values to a lower-precision representation in order to reduce storage, bandwidth, or compute cost.

## Key Points

- QLoRA applies block-wise `4-bit` quantization to the frozen pretrained backbone while keeping computation in BF16 after dequantization.
- The paper emphasizes that earlier low-bit approaches were effective mainly for inference, whereas QLoRA demonstrates stable training through quantized weights.
- Independent block quantization is used to mitigate outlier effects that would otherwise waste representational bins.
- The work argues that low-bit quantization can preserve downstream quality when paired with adapter-based recovery.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dettmers-2023-qlora-2305-14314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dettmers-2023-qlora-2305-14314]].
