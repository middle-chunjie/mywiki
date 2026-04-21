---
type: concept
title: Binary Token Representation
slug: binary-token-representation
date: 2026-04-20
updated: 2026-04-20
aliases: [BTR, binary token representations, 二值令牌表示]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Binary Token Representation** (二值令牌表示) — a token-level passage encoding that stores each hidden-state dimension as a binary sign bit so representations can be cached compactly for later reuse.

## Key Points

- [[unknown-nd-btrbinary-2310-01329]] applies binary encoding to passage tokens rather than whole passages, preserving finer-grained information for reader models.
- The paper defines token binarization as `` `b_k = sign(h_k)` ``, mapping each dimension to `1` or `-1`.
- These binary token states let the reader precompute passage-side computation offline and avoid re-encoding retrieved passages at inference time.
- The paper reports that binary token caching is the main reason BTR reduces storage by over `100x` relative to continuous passage caches.
- The approach remains useful for both Atlas-style encoder-decoder readers and BERT-based encoder-only readers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-btrbinary-2310-01329]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-btrbinary-2310-01329]].
