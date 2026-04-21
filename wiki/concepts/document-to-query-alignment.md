---
type: concept
title: Document-to-Query Alignment
slug: document-to-query-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [doc-to-query alignment, 文档到查询对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Document-to-Query Alignment** (文档到查询对齐) — an alignment direction in which each document token selects or softly attends to the most relevant query token when computing relevance.

## Key Points

- The paper identifies this as the natural alignment direction induced by generative retrieval's decoder attention.
- In the GR formulation, each predicted document token contributes `rel(d_i, q)` after attending over all query tokens with a row-wise softmax.
- This direction enables autoregressive token-by-token generation, because the model can pre-compute how the next document token should align using earlier generated tokens.
- Empirically, document-to-query alignment is weaker than standard MVDR query-to-document alignment in reranking, though stronger document encoding can partially compensate.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-generative-2404-00684]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-generative-2404-00684]].
