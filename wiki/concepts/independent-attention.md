---
type: concept
title: Independent Attention
slug: independent-attention
date: 2026-04-20
updated: 2026-04-20
aliases: [independent attention, 独立注意力]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Independent Attention** (独立注意力) — an attention masking scheme in which retrieved document chunks attend within themselves but not across chunks, while query and generated tokens can still attend over the assembled document context.

## Key Points

- TurboRAG uses independent attention to make offline per-chunk KV-cache computation compatible with online cache concatenation.
- The paper argues cross-document attention among retrieved chunks is empirically sparse in multi-document RAG, so masking it out often preserves answer quality.
- Under this masking scheme, each chunk can be prefetched and cached separately, eliminating redundant document-side prefill at inference time.
- The method still allows the query and generated answer tokens to attend to all retrieved chunks after the caches are assembled online.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-turborag]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-turborag]].
