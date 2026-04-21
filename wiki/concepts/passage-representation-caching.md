---
type: concept
title: Passage Representation Caching
slug: passage-representation-caching
date: 2026-04-20
updated: 2026-04-20
aliases: [cacheable passage representations, 段落表示缓存]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Passage Representation Caching** (段落表示缓存) — the practice of precomputing passage-side model states offline so a retrieval reader can reuse them at inference time instead of re-encoding each passage.

## Key Points

- [[unknown-nd-btrbinary-2310-01329]] frames reader-side passage encoding as the dominant inference bottleneck in retrieval-augmented models.
- BTR caches passage token representations at an intermediate encoder layer and loads them during inference after computing the query-side states online.
- The paper positions DeFormer as a continuous-valued variant of the same decomposition idea, but with much worse storage costs.
- BTR shows that caching remains practical at Wikipedia scale only after replacing continuous states with binary token representations and adding compression.
- The cached passage states are retrieved from a key-value store and converted back to float values before upper-layer encoder processing.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-btrbinary-2310-01329]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-btrbinary-2310-01329]].
