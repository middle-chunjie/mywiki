---
type: concept
title: Multimodal Retrieval
slug: multimodal-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [multimodal retrieval, cross-modal retrieval, 多模态检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multimodal Retrieval** (多模态检索) — retrieval over mixed text and image evidence by mapping queries and documents into compatible embedding spaces for relevance matching.

## Key Points

- [[dong-2024-progressive-2412-14835]] uses a unified retrieval module that combines text retrieval and cross-modal retrieval over a hybrid-modal corpus.
- The method encodes image-text pairs with CLIP and averages image and text embeddings as `E_x(x,t) = (E_I(x) + E_T(t)) / 2` when both modalities are available.
- Retrieval is not a one-shot preprocessing step: AR-MCTS re-retrieves step-specific evidence during search expansion.
- FAISS indexing is used to make dense retrieval over the large hybrid corpus practical at inference time.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2024-progressive-2412-14835]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2024-progressive-2412-14835]].
