---
type: concept
title: KV Cache Compression
slug: kv-cache-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [KV compression, activation compression, key-value cache compression]
tags: [long-context, efficient-inference, kv-cache, memory-efficiency]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**KV Cache Compression** (KV缓存压缩) — A class of methods that reduce the memory footprint of the key-value cache in transformer self-attention by condensing multiple raw token representations into fewer, denser representations, enabling processing of longer contexts within a fixed memory budget.

## Key Points

- Distinct from [[kv-cache-eviction]] (which discards tokens based on importance scores) and [[kv-cache]] management: compression retains information from all tokens by encoding them into fewer aggregate representations.
- Condensing `l` token activations into `k` compressed activations (ratio `α = l/k`) trades memory reduction against potential information loss; higher `α` enables longer contexts but coarser representations.
- Methods differ in how compression is implemented: soft-prompt tokens (Gist Tokens), segment summarization (AutoCompressor, RMT), special beacon tokens (Activation Beacon), or recurrent state (Mamba-style).
- Training objective matters: Activation Beacon trains via next-token prediction (auto-regression) over compressed representations, coupling compression quality directly to generation quality.
- Compatible with retrieval augmentation: compressed activations provide coarse context coverage, while retrieval of lightly compressed intervals can restore fine-grained accuracy for specific queries.

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-long-2401-03462]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-long-2401-03462]].
