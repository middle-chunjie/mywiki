---
type: concept
title: Generative Information Retrieval
slug: generative-information-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [GenIR, 生成式信息检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Generative Information Retrieval** (生成式信息检索) — an information retrieval paradigm that satisfies information needs by generating document identifiers or direct responses instead of relying only on similarity-based ranking over an explicit index.

## Key Points

- The survey divides GenIR into two major forms: generative document retrieval and reliable response generation.
- In the retrieval form, models map queries to valid DocID sequences and rank documents by generation probability.
- In the response-generation form, language models directly answer users while improving factuality, grounding, citation, and personalization.
- The paper argues that GenIR better matches user intent because users often want an answer or evidence rather than a ranked list alone.
- Central bottlenecks include scalability, dynamic corpora, inference latency, factuality, safety, and unified retrieval-generation training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-matching-2404-14851]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-matching-2404-14851]].
