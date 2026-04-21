---
type: concept
title: Document-Level Relevance
slug: document-level-relevance
date: 2026-04-20
updated: 2026-04-20
aliases: [per-document relevance, 文档级相关性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Document-Level Relevance** (文档级相关性) — the notion of assigning each retrieved document its own usefulness score for a query or downstream task instead of only scoring the retrieved list as a whole.

## Key Points

- eRAG produces document-level relevance by running the downstream generator on each retrieved document separately.
- The label for each document is the downstream task score of the generated output against gold supervision, not a direct human relevance judgment.
- These per-document labels make it possible to optimize or compare retrievers with standard ranking metrics.
- The paper argues document-level relevance is necessary for interpretability and for retriever optimization settings such as interleaving.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[salemi-2024-evaluating]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[salemi-2024-evaluating]].
