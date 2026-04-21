---
type: concept
title: Any-Granularity Ranking
slug: any-granularity-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [arbitrary-granularity ranking, 任意粒度排序]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Any-Granularity Ranking** (任意粒度排序) — a ranking setup where the system can score retrieval units or query fragments at multiple granularities without rebuilding the underlying representation index for each granularity.

## Key Points

- AGRaME defines any-granularity ranking as reusing a coarser-granularity encoding, such as passages or sentences, while scoring finer units like sentences or propositions.
- The main motivation is to avoid training specialized dense encoders and maintaining separate indexes for each target granularity.
- The paper shows both `query -> sub-retrieval-unit` ranking and `sub-query -> retrieval-unit` ranking under the same multi-vector framework.
- The approach relies on token-level interactions so that sub-spans inside the encoded text can be scored directly at inference time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[reddy-2024-agrame-2405-15028]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[reddy-2024-agrame-2405-15028]].
