---
type: concept
title: Proactive Retrieval
slug: proactive-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 主动检索
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Proactive Retrieval** (主动检索) — a retrieval setting in which a system monitors an interaction and decides whether to surface documents before any explicit user request is issued.

## Key Points

- ProCIS defines proactive retrieval at the utterance level: after each turn, the system can either wait or return a ranked list of documents.
- The task is harder than standard conversational retrieval because the model must jointly solve timing and ranking.
- The paper introduces `npDCG` to reward timely interventions and penalize late or repeated document suggestions.
- Benchmarking combines a DeBERTa-based engagement classifier with downstream retrievers such as BM25, SPLADE, ANCE, ColBERT, and LMGR.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[samarinas-2024-procis]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[samarinas-2024-procis]].
