---
type: concept
title: Byte Shuffling
slug: byte-shuffling
date: 2026-04-20
updated: 2026-04-20
aliases: [字节重排, byte shuffle]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Byte Shuffling** (字节重排) — a layout transformation that reorders bytes so corresponding byte positions from many floating-point values are grouped together before compression.

## Key Points

- The paper uses byte shuffling after transposing the angle matrix so exponent bytes and mantissa bytes form separate low-entropy streams.
- This operation is inherited from prior floating-point compressors, but the paper combines it with a spherical transform that makes the grouped bytes much more regular.
- Byte shuffling is essential to realize the entropy reduction from exponent concentration in practice.
- The baseline compared in the paper is effectively transpose plus byte shuffle plus `zstd`, making the contribution of the spherical transform easy to isolate.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-00079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-00079]].
