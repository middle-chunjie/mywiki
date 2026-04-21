---
type: concept
title: Query-Document Matching
slug: query-document-matching
date: 2026-04-20
updated: 2026-04-20
aliases: [QDM]
tags: [retrieval, supervision, rag]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Query-Document Matching** (查询文档匹配) — the task of estimating whether each candidate document is relevant to a query, often used as an auxiliary supervision signal for retrieval models or retrieval-aware modules.

## Key Points

- In [[ye-2024-rag-2406-13249]], QDM is used to train R2-Former to interpret list-wise retrieval features before they are passed to the LLM.
- The model predicts a binary relevance score for each document in the retrieved list with `\hat{s} = f_{\to 1}(H)`.
- The auxiliary loss is binary cross-entropy over per-document labels and is combined with language-modeling loss as `L = L_QDM + L_LM`.
- Ablation shows removing the QDM objective degrades NQ-10 accuracy from `0.6930` to `0.6441`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2024-rag-2406-13249]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2024-rag-2406-13249]].
