---
type: concept
title: Network Embedding
slug: network-embedding
date: 2026-04-20
updated: 2026-04-20
aliases: [网络嵌入, graph embedding]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Network Embedding** (网络嵌入) — a representation-learning approach that maps graph nodes into dense vectors while preserving structural context.

## Key Points

- The paper builds a patent citation graph and learns patent representations from graph neighborhoods.
- DeepWalk-style path sampling provides local context sequences for each patent node.
- The learned network embedding serves as the meta-feature branch inside NCNN rather than as a standalone predictor.
- Structural citation information is used because litigation risk correlates with patent relationships, not only raw text.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2018-patent]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2018-patent]].
