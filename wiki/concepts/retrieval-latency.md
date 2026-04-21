---
type: concept
title: Retrieval Latency
slug: retrieval-latency
date: 2026-04-20
updated: 2026-04-20
aliases: [latency in retrieval, 检索延迟]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Latency** (检索延迟) — the end-to-end time required for a retrieval system to return ranked answers for a query.

## Key Points

- STaRK treats latency as a first-class evaluation dimension because practical retrieval systems must answer quickly.
- The paper reports latency on `1 × NVIDIA A100-SXM4-80GB`, making model families directly comparable.
- First-stage retrievers such as DPR and QAGNN are much faster (`1.40s` and `1.65s` average) than LLM rerankers (`26.33s` for Claude3 and `25.05s` for GPT-4).
- The benchmark shows a clear accuracy-latency trade-off: stronger reasoning during reranking can improve top-rank quality but substantially hurts response time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-stark-2404-13207]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-stark-2404-13207]].
