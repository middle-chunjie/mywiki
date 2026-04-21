---
type: concept
title: Byte-level Tokenization
slug: byte-level-tokenization
date: 2026-04-20
updated: 2026-04-20
aliases: [byte-level modeling, 字节级分词]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Byte-level Tokenization** (字节级分词) — a text representation scheme that models raw byte sequences directly instead of subword or wordpiece tokens.

## Key Points

- The paper studies a `7B` byte-level Transformer trained on `314.6B` bytes of code, roughly equivalent to `116B` tokens.
- Multi-byte prediction works especially well in this regime: the `n = 8` model strongly improves MBPP, HumanEval, and APPS/Intro over the next-byte baseline.
- The byte-level setup uses a longer `8192` context length, reflecting the higher sequence length cost of byte representations.
- The authors argue that self-speculative decoding can offset the longer byte sequences at inference time and make byte-level models practically competitive.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gloeckle-2024-better-2404-19737]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gloeckle-2024-better-2404-19737]].
