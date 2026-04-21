---
type: concept
title: KV Cache Eviction
slug: kv-cache-eviction
date: 2026-04-20
updated: 2026-04-20
aliases: [KV cache eviction policy, key-value cache eviction, KV eviction, 键值缓存淘汰]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**KV Cache Eviction** (键值缓存淘汰) — a policy that selectively removes past key-value pairs from the fixed-size attention cache during autoregressive LLM decoding to keep memory bounded without recomputing the full prefix.

## Key Points

- H2O formalizes eviction as a dynamic submodular maximization problem: at each step, the token whose removal least reduces the accumulated attention score is discarded.
- The eviction rule is: `u = argmax_{v∈G_i} F_score(G_i \ {v})`, where `F_score(T) = Σ_{s∈T} o_s` and `o_s` is the normalized cumulative attention score of token `s`.
- A near-optimal theoretical guarantee follows from submodularity: the greedy eviction achieves `f(S̃_i) ≥ (1−α)(1−1/e) max_{|S|=k} f(S) − β`.
- Simple recency-only ("Local") eviction fails dramatically because it discards heavy-hitter tokens that absorb large cumulative attention mass.
- H2O evenly splits the cache budget `k` between heavy hitters (H2) and the most recent tokens, combining global importance with local context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-ho-2306-14048]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-ho-2306-14048]].
