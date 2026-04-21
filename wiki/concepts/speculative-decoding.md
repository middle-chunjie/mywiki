---
type: concept
title: Speculative Decoding
slug: speculative-decoding
date: 2026-04-20
updated: 2026-04-20
aliases: [投机解码]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Speculative Decoding** (投机解码) — a decoding strategy that drafts multiple candidate future tokens and verifies them in parallel to accelerate exact autoregressive generation.

## Key Points

- The paper uses the extra MTP heads for self-speculative decoding, avoiding the need for a separate draft model.
- On a `7B` 4-token model, using four heads yields relative speedups of `3.05x` on code, `2.74x` on Wikipedia, and `2.67x` on books.
- For byte-level MTP, the same idea scales to much larger speedups, reaching `6.39x` with `n = 8`.
- The decoding gains depend on head accuracy and acceptance rate; in the main code setting, the model retrieves about `3.50` tokens per forward pass.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gloeckle-2024-better-2404-19737]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gloeckle-2024-better-2404-19737]].
