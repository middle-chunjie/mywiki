---
type: concept
title: Dual-Level Retrieval
slug: dual-level-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [two-level retrieval, 双层检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Dual-Level Retrieval** (双层检索) — a retrieval design that separately handles fine-grained entity-specific evidence and broader topic-level evidence, then combines them for answer generation.

## Key Points

- LightRAG explicitly splits retrieval into low-level retrieval for specific entities and relations and high-level retrieval for broader themes.
- The distinction is motivated by the need to answer both detail-oriented queries and abstract sensemaking queries within one system.
- Query processing extracts local and global keywords so the two retrieval levels can target different parts of the graph index.
- Ablation results show that removing either the high-level or low-level branch reduces answer quality across multiple datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-lightrag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-lightrag]].
