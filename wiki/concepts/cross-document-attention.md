---
type: concept
title: Cross-Document Attention
slug: cross-document-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [document-to-document attention, 跨文档注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Cross-Document Attention** (跨文档注意力) — an attention operation in which representations from one document use another document's keys and values as context.

## Key Points

- In REVELA, document `D_i` attends to other in-batch documents `D_j` through `softmax(Q_i^h K_j^{eT} / sqrt(d_H)) V_j^e`.
- The contribution from each external document is modulated by retriever similarity rather than treated uniformly.
- This mechanism is the bridge that turns retrieval quality into language-modeling utility.
- The paper uses it to capture semantic relations among chunks without explicit query-document supervision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cai-2026-revela-2506-16552]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cai-2026-revela-2506-16552]].
