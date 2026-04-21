---
type: concept
title: Floating-Point Compression
slug: floating-point-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [浮点压缩]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Floating-Point Compression** (浮点压缩) — compression methods designed specifically for arrays of floating-point values by exploiting structure in sign, exponent, and mantissa representations.

## Key Points

- The paper frames embedding storage as a floating-point compression problem rather than only a retrieval or quantization problem.
- Prior baselines mainly exploit exponent regularities after transposition and byte shuffling, but leave mantissa entropy largely unchanged.
- The spherical reparameterization changes the value distribution itself, lowering both exponent entropy and part of the mantissa entropy.
- The paper shows that format-specific behavior matters: the method helps float32 but is counterproductive on BF16, FP16, FP8, and Int8.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-00079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-00079]].
