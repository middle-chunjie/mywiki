---
type: concept
title: Enhanced Decoding
slug: enhanced-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [增强解码]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Enhanced Decoding** (增强解码) — a reconstruction design that increases training signal and context diversity by letting each target token decode from a position-specific visible context rather than one shared masked input.

## Key Points

- RetroMAE builds a query stream `H_1` and a context stream `H_2` instead of using only a single masked decoder input.
- A position-specific attention mask prevents each token from attending to itself while exposing a different visible context to each row.
- This lets the decoder reconstruct all tokens in the sequence, not just the originally masked subset.
- In ablations, enhanced decoding improves MS MARCO DPR `MRR@10` from `0.3462` to `0.3553`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2022-retromae-2205-12035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2022-retromae-2205-12035]].
