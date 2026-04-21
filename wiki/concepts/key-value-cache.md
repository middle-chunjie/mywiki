---
type: concept
title: Key-Value Cache
slug: key-value-cache
date: 2026-04-20
updated: 2026-04-20
aliases: [KV cache, key value cache, 键值缓存]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Key-Value Cache** (键值缓存) — the stored key and value states from prior tokens that accelerate autoregressive decoding by avoiding recomputation of attention over the full prefix.

## Key Points

- The paper identifies KV cache size as the main deployment bottleneck for long-context LLM inference, because it constrains both batch size and sequence length.
- Standard MHA requires cache proportional to `2 n_h d_h l` per token, which becomes expensive at large head counts and many layers.
- MLA compresses this cache to `(d_c + d_h^R) l` elements per token, replacing full keys and values with a low-rank latent state plus a small RoPE-carrying key.
- After additional deployment-time quantization, DeepSeek-V2 reports a `93.3%` KV-cache reduction relative to DeepSeek 67B.
- The smaller cache is a major factor behind the reported `5.76x` maximum generation throughput gain.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deepseek-ai-2024-deepseekv-2405-04434]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deepseek-ai-2024-deepseekv-2405-04434]].
