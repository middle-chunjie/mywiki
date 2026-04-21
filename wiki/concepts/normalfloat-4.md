---
type: concept
title: NormalFloat-4
slug: normalfloat-4
date: 2026-04-20
updated: 2026-04-20
aliases: [NF4, 4-bit NormalFloat]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**NormalFloat-4** — a 4-bit quantization data type designed to be information-theoretically well matched to normally distributed neural network weights.

## Key Points

- QLoRA uses NF4 as its main storage data type for pretrained backbone weights.
- The paper motivates NF4 by observing that pretrained weights are approximately normally distributed and then testing this assumption empirically.
- On mean Pile Common Crawl perplexity, `NF4 + DQ` outperforms both `Int4` and ordinary `FP4` variants.
- The authors report that NF4 fully recovers 16-bit LoRA performance in their LLaMA MMLU experiments when paired with double quantization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dettmers-2023-qlora-2305-14314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dettmers-2023-qlora-2305-14314]].
