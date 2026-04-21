---
type: concept
title: Token Compression
slug: token-compression
date: 2026-04-20
updated: 2026-04-20
aliases: [token merging, 令牌压缩]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Token Compression** (令牌压缩) — a procedure that merges semantically similar token representations to reduce storage or computation while trying to preserve task-relevant information.

## Key Points

- [[unknown-nd-btrbinary-2310-01329]] uses offline token compression on cached binary passage states and runtime token compression on continuous query-passage states.
- Offline compression merges non-stopword binary vectors with Hamming-distance similarity and collapses stopwords to shared mean vectors.
- Runtime compression operates both within each query-passage pair and across passages after encoder fusion, using cosine similarity for continuous states.
- The paper chooses a compression ratio of `0.2` for both offline and runtime compression as the best trade-off on NaturalQuestions.
- Ablations show offline compression cuts cache size from `159 GB` to `127 GB`, while runtime compression raises throughput from `24.6` or `28.1` QPS to `31.5` QPS on NaturalQuestions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-btrbinary-2310-01329]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-btrbinary-2310-01329]].
