---
type: concept
title: Retrieval Evaluation
slug: retrieval-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [retriever evaluation, 检索评测]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Evaluation** (检索评测) — the process of scoring a retrieval system by comparing its ranked outputs against relevance labels or downstream task outcomes.

## Key Points

- The paper argues that standard end-to-end RAG evaluation is a poor retrieval evaluation signal because it only gives list-level feedback.
- It shows that KILT provenance labels and generic LLM relevance judgments correlate weakly with downstream RAG performance.
- eRAG reframes retrieval evaluation around per-document downstream utility rather than human-style query-document relevance alone.
- The paper evaluates retrieval with MAP, MRR, NDCG, Precision, Recall, and Hit Ratio after constructing document-level labels from downstream task scores.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[salemi-2024-evaluating]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[salemi-2024-evaluating]].
