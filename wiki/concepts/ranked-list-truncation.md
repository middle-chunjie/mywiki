---
type: concept
title: Ranked-list truncation
slug: ranked-list-truncation
date: 2026-04-20
updated: 2026-04-20
aliases: [dynamic cutoff prediction, ranking list truncation, 排名列表截断]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Ranked-list truncation** (排名列表截断) — the task of deciding where to cut a ranked list so that relevant items are retained while low-value or misleading items are excluded.

## Key Points

- The paper treats truncation as essential for balancing relevance against noise in both web search and retrieval-augmented generation.
- GenRT reformulates truncation as step-wise binary classification performed concurrently with reranking.
- Its truncation decision uses both forward sequential context and a local backward window around the current item.
- The paper optimizes truncation with RAML-style soft labels derived from TDCG-based rewards.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-listaware-2402-02764]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-listaware-2402-02764]].
