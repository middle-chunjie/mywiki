---
type: concept
title: Entropy Coding
slug: entropy-coding
date: 2026-04-20
updated: 2026-04-20
aliases: [熵编码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Entropy Coding** (熵编码) — a compression stage that assigns shorter codes to more probable symbols based on their empirical distribution.

## Key Points

- The paper relies on entropy coding after spherical transformation, transposition, and byte shuffling.
- Its central claim is that geometry makes the byte distribution easier for entropy coders to compress, especially the exponent stream.
- Compression quality depends on context size: large batches or chunks let the entropy coder fit the concentrated angular distribution better.
- The work leaves open whether fixed probability tables from known angular laws could remove some batch-size dependence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2026-embedding-2602-00079]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2026-embedding-2602-00079]].
