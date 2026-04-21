---
type: concept
title: Double Quantization
slug: double-quantization
date: 2026-04-20
updated: 2026-04-20
aliases: [DQ, double quantization, 双重量化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Double Quantization** (双重量化) — a quantization scheme that additionally quantizes the quantization constants themselves to reduce metadata overhead.

## Key Points

- QLoRA stores the primary weight blocks in `NF4` and then compresses their scale constants through a second quantization stage.
- The paper defines this through `doubleDequant(c1, c2, W)`, where `c2` is stored in low precision and dequantized before reconstructing the BF16 weights.
- The method saves about `0.37 bits` per parameter, which the paper describes as roughly `3 GB` for a `65B` model.
- Empirically, the paper reports that double quantization improves memory efficiency without degrading the quality recovered by QLoRA.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dettmers-2023-qlora-2305-14314]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dettmers-2023-qlora-2305-14314]].
