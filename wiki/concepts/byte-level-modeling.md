---
type: concept
title: Byte-Level Modeling
slug: byte-level-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [byte-level language modeling, 字节级建模]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Byte-Level Modeling** (字节级建模) — sequence modeling that operates directly on raw byte tokens rather than on subwords, words, or learned discrete codes.

## Key Points

- MEGABYTE fixes the vocabulary at `256` byte values and applies the same representation to text, images, and raw audio.
- On long-context text, the paper reports that byte-level modeling can approach strong subword systems when the architecture is designed to scale efficiently.
- Modeling bytes lets the same model family handle ImageNet sequences up to `1,228,800` bytes and raw `16 kHz` audio without a modality-specific tokenizer.
- The paper positions byte-level modeling as a way to avoid language-specific tokenization heuristics and lossy discrete compression.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yu-2023-megabyte-2305-07185]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yu-2023-megabyte-2305-07185]].
