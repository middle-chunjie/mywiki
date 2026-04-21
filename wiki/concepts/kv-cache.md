---
type: concept
title: KV Cache
slug: kv-cache
date: 2026-04-20
updated: 2026-04-20
aliases: [key-value cache, 键值缓存]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**KV Cache** (键值缓存) — the stored key and value activations from previous decoding steps that let autoregressive transformers reuse past attention states instead of recomputing them from scratch.

## Key Points

- The paper identifies KV-cache retrieval as a dominant decode-time bottleneck for long-context autoregressive transformers.
- Its primer estimates that Llama `7B` uses about `512 KB` of KV cache per token, reaching roughly `16 GB` total KV memory at sequence length `2048` and batch size `16`.
- In Block Transformer, the block decoder reduces KV-cache size by about `L_B` and KV-cache access by `L_B^2` because it operates on block embeddings.
- The token decoder keeps only a tiny local cache for the current block, reducing cache size/access by `R = L / L_B` relative to vanilla global attention.
- Lower KV-cache pressure lets the model run with larger batch sizes, which is a main source of the reported throughput gains.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ho-2024-block-2406-02657]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ho-2024-block-2406-02657]].
